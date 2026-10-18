"""
build_splits.py - turn the raw downloaded datasets into train / validation / test folders.

What it does, in order:
  1. Read configs/label_map.yaml to learn which raw folder -> which training class.
  2. Collect every image from those folders under data/raw/.
  3. Skip folders marked VERIFY (we have not decided what they are yet).
  4. Skip folders whose class is excluded (hazardous when num_classes is 3).
  5. Drop exact duplicate images (same bytes -> same SHA-256 hash) and corrupted files.
  6. Shuffle with seed 42 and split each class 70 / 15 / 15.
  7. Copy files into data/train, data/validation, data/test (one sub-folder per class).
  8. Re-hash the copied files to confirm no image is in more than one split.
  9. Print how many images each class has in each split.

Run from the repo root:
    python src/data/build_splits.py                    # uses num_classes from label_map.yaml
    python src/data/build_splits.py --num-classes 3    # one-off override without editing the yaml
    python src/data/build_splits.py --dry-run          # count only, copy nothing

Re-running deletes and rebuilds data/train, data/validation and data/test,
so the result is always the same for the same raw data and seed.
Nothing under data/ is committed to git.
"""

import argparse
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

# --- Paths. Everything is relative to the repo root, two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"
SPLIT_NAMES = ["train", "validation", "test"]
SPLIT_DIRS = {name: REPO_ROOT / "data" / name for name in SPLIT_NAMES}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(num_classes_override=None):
    """Read label_map.yaml and return (config, active class set, num_classes).

    The yaml has one entry under `class_sets` per class count (3 or 4).
    `num_classes` at the top picks which one is active. That is the
    "one line config change".
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    num_classes = num_classes_override or config["num_classes"]
    if num_classes not in config["class_sets"]:
        raise SystemExit(f"label_map.yaml has no class_sets entry for {num_classes}. Options: 3 or 4.")

    active = config["class_sets"][num_classes]
    print(f"Using {num_classes} classes: {active['classes']}  (excluded: {active['exclude'] or 'none'})")
    return config, active, num_classes


def file_hash(path):
    """SHA-256 of the file bytes. Two files with the same hash are exact duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read 1 MB at a time so big files never load fully into memory.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path):
    """True if Pillow can open the file. Catches corrupted or truncated images."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def folders_for_dataset(name, dataset_cfg):
    """Return a list of (folder_path, target_class) pairs for one dataset.

    Two shapes are allowed in the yaml:
      - `root` + `folders`: a dict of raw folder name -> target class
      - `label`: every image directly inside data/raw/<name>/ gets this class
    Returns an empty list if data/raw/<name>/ does not exist (not downloaded
    yet, or the dataset key is misspelled; main() prints a [skip] line for it).
    A missing `root` or `folders` entry is an error, so a typo there can never
    silently drop a class.
    """
    dataset_dir = RAW_DIR / name
    if not dataset_dir.exists():
        return []

    if "label" in dataset_cfg:
        return [(dataset_dir, dataset_cfg["label"])]

    dataset_root = dataset_dir / (dataset_cfg.get("root") or "")
    if not dataset_root.is_dir():
        raise SystemExit(f"label_map.yaml root '{dataset_cfg.get('root')}' has no folder at {dataset_root.relative_to(REPO_ROOT)}")

    pairs = []
    for folder_name, target in dataset_cfg["folders"].items():
        folder = dataset_root / folder_name
        # Every key in label_map.yaml must match a real folder. Stop if not,
        # so a typo never silently drops a whole class.
        if not folder.is_dir():
            raise SystemExit(f"label_map.yaml key '{folder_name}' has no folder at {folder.relative_to(REPO_ROOT)}")
        pairs.append((folder, target))
    return pairs


def collect_images(config, active):
    """Return {class_name: [(path, hash), ...]} with duplicates and bad files removed.

    Rules for each raw folder:
      - target VERIFY               -> skip (not decided yet)
      - target in active `exclude`  -> skip (e.g. hazardous in 3-class mode)
      - target not a known class    -> stop with an error (catches typos)
    """
    images_by_class = defaultdict(list)
    seen_hashes = set()
    stats = Counter()

    for dataset_name, dataset_cfg in config["datasets"].items():
        pairs = folders_for_dataset(dataset_name, dataset_cfg)
        if not pairs:
            print(f"  [skip] data/raw/{dataset_name}/ not found (not downloaded yet?)")
            continue

        for folder, target in pairs:
            if target == "VERIFY":
                stats["skipped_verify_folders"] += 1
                continue
            if target in active["exclude"]:
                stats["skipped_excluded_folders"] += 1
                continue
            if target not in active["classes"]:
                raise SystemExit(f"Unknown class '{target}' for {folder} in label_map.yaml")

            # A `label:` dataset only reads its own top level, so the README and
            # any sub-folders are ignored. `folders:` datasets read recursively.
            files = folder.iterdir() if "label" in dataset_cfg else folder.rglob("*")
            for path in sorted(files):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                stats["found"] += 1

                if not is_valid_image(path):
                    stats["corrupted"] += 1
                    continue

                digest = file_hash(path)
                if digest in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(digest)

                images_by_class[target].append((path, digest))

    print(
        f"\nFound {stats['found']} images. Dropped {stats['duplicates']} exact duplicates "
        f"and {stats['corrupted']} corrupted files. "
        f"Skipped {stats['skipped_verify_folders']} VERIFY folders and "
        f"{stats['skipped_excluded_folders']} excluded folders."
    )
    return images_by_class


def split_list(items, ratios, seed):
    """Shuffle `items` with a fixed seed, then cut into train / validation / test."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],  # the remainder, so nothing is lost to rounding
    }


def copy_splits(splits_by_class):
    """Copy images into data/<split>/<class>/.

    Filenames get a dataset + folder prefix so two datasets that both have a
    file called 1.jpg never overwrite each other.
    """
    for d in SPLIT_DIRS.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for class_name, splits in splits_by_class.items():
        for split_name, items in splits.items():
            out_dir = SPLIT_DIRS[split_name] / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src, _digest in items:
                # data/raw/garbage12_enhanced/.../battery/battery7.jpg
                #   -> garbage12_enhanced__battery__battery7.jpg
                dataset = src.relative_to(RAW_DIR).parts[0]
                shutil.copy2(src, out_dir / f"{dataset}__{src.parent.name}__{src.name}")


def check_no_overlap():
    """Re-hash every copied file and confirm no image sits in two splits.

    This reads the files actually on disk in data/train, data/validation and
    data/test, so it checks the real output, not just our in-memory lists.
    """
    hashes = {}
    for split_name, split_dir in SPLIT_DIRS.items():
        hashes[split_name] = {file_hash(p) for p in split_dir.rglob("*") if p.is_file()}

    overlaps = {
        "train & validation": hashes["train"] & hashes["validation"],
        "train & test": hashes["train"] & hashes["test"],
        "validation & test": hashes["validation"] & hashes["test"],
    }
    bad = {pair: len(common) for pair, common in overlaps.items() if common}
    if bad:
        raise SystemExit(f"FAILED: images found in more than one split: {bad}")
    print("Overlap check passed: no image appears in more than one split.")


def print_table(splits_by_class):
    """Print class counts per split as a small table."""
    print(f"\n{'class':<12}{'train':>8}{'validation':>12}{'test':>8}{'total':>8}")
    print("-" * 48)
    totals = Counter()
    for class_name in sorted(splits_by_class):
        counts = {k: len(v) for k, v in splits_by_class[class_name].items()}
        total = sum(counts.values())
        totals.update(counts)
        totals["total"] += total
        print(f"{class_name:<12}{counts['train']:>8}{counts['validation']:>12}{counts['test']:>8}{total:>8}")
    print("-" * 48)
    print(f"{'TOTAL':<12}{totals['train']:>8}{totals['validation']:>12}{totals['test']:>8}{totals['total']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Build train/validation/test folders from data/raw/.")
    parser.add_argument("--num-classes", type=int, choices=[3, 4], help="override num_classes from label_map.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count and print, but do not copy files")
    args = parser.parse_args()

    config, active, num_classes = load_config(args.num_classes)
    ratios = config["split"]["ratios"]
    seed = config["split"]["seed"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit("split ratios in label_map.yaml must add up to 1.0")

    images_by_class = collect_images(config, active)
    if not images_by_class:
        raise SystemExit("No images found. Run src/data/download.py first.")

    splits_by_class = {cls: split_list(items, ratios, seed) for cls, items in images_by_class.items()}
    print_table(splits_by_class)

    if args.dry_run:
        print("\n(dry run: nothing was copied)")
        return

    copy_splits(splits_by_class)
    check_no_overlap()
    print(f"Wrote data/train, data/validation, data/test  (seed={seed}, {num_classes} classes)")


if __name__ == "__main__":
    main()
