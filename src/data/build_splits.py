"""
build_splits.py - turn the raw downloaded datasets into train / validation / test data.

What it does, in order:
  1. Read configs/label_map.yaml to learn which raw folder -> which training class.
  2. Collect every image from those folders under data/raw/.
     - Skip folders marked VERIFY (not decided yet).
     - Skip folders whose class is excluded (hazardous when num_classes is 3).
     - Skip corrupted files that Pillow cannot open.
  3. Remove EXACT duplicates (identical bytes -> identical SHA-256 hash).
  4. Remove NEAR duplicates (perceptual hash distance <= near_duplicate_max_distance
     in label_map.yaml, 5 by default), across all classes.
     If near duplicates disagree on the label, drop all of them (label conflict).
  5. Shuffle with seed 42 and split each class 70 / 15 / 15.
  6. Copy files into data/train, data/validation, data/test (one sub-folder per class).
  7. Write CSV manifests: data/splits/train.csv, validation.csv, test.csv (path, label).
  8. Re-check the copied files: no exact or near duplicate may sit in two splits.
  9. Compute class weights from the TRAIN split only -> results/class_weights.json.
 10. Write every removed file to results/removed_duplicates.csv.

Run from the repo root:
    python src/data/build_splits.py                    # uses num_classes from label_map.yaml
    python src/data/build_splits.py --num-classes 3    # one-off override without editing the yaml
    python src/data/build_splits.py --dry-run          # count only, write nothing

Re-running deletes and rebuilds the outputs, so the result is always the same
for the same raw data and seed. Images under data/ are never committed to git.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

# --- Paths. Everything is relative to the repo root, two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"
SPLIT_NAMES = ["train", "validation", "test"]
SPLIT_DIRS = {name: REPO_ROOT / "data" / name for name in SPLIT_NAMES}
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
REMOVED_CSV = RESULTS_DIR / "removed_duplicates.csv"
WEIGHTS_JSON = RESULTS_DIR / "class_weights.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def rel(path):
    """Path relative to the repo root, with forward slashes (same on Mac and Windows)."""
    return Path(path).relative_to(REPO_ROOT).as_posix()


def file_hash(path):
    """SHA-256 of the file bytes. Two files with the same hash are exact duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read 1 MB at a time so big files never load fully into memory.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path):
    """64-bit perceptual hash (pHash) as a plain integer.

    pHash shrinks the image to a tiny grayscale version and keeps only its
    overall light/dark pattern. Visually similar images get hashes that differ
    in only a few bits, even if the files are resized or re-saved.
    """
    with Image.open(path) as img:
        return int(str(imagehash.phash(img)), 16)  # hex string -> integer


def hamming_distances(one_hash, all_hashes):
    """How many bits differ between one hash and every hash in a numpy array."""
    return np.bitwise_count(np.uint64(one_hash) ^ all_hashes)


def is_valid_image(path):
    """True if Pillow can fully read the file. Catches corrupted or truncated images."""
    try:
        with Image.open(path) as img:
            img.verify()  # checks the file structure
        with Image.open(path) as img:
            img.load()    # decodes every pixel; a cut-off file fails here
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 1-2: read the config and collect images
# ---------------------------------------------------------------------------

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
        raise SystemExit(f"label_map.yaml root '{dataset_cfg.get('root')}' has no folder at {rel(dataset_root)}")

    pairs = []
    for folder_name, target in dataset_cfg["folders"].items():
        folder = dataset_root / folder_name
        if not folder.is_dir():
            raise SystemExit(f"label_map.yaml key '{folder_name}' has no folder at {rel(folder)}")
        pairs.append((folder, target))
    return pairs


def collect_images(config, active, removed_rows):
    """Return a list of images as dicts: {"path": Path, "label": str}.

    Rules for each raw folder:
      - target VERIFY               -> skip (not decided yet)
      - target in active `exclude`  -> skip (e.g. hazardous in 3-class mode)
      - target not a known class    -> stop with an error (catches typos)
    Corrupted files are skipped and logged in removed_rows.
    """
    items = []
    stats = Counter()

    for dataset_name, dataset_cfg in config["datasets"].items():
        pairs = folders_for_dataset(dataset_name, dataset_cfg)
        if not pairs:
            print(f"  [skip] data/raw/{dataset_name}/ not found (not downloaded yet?)")
            continue

        for folder, target in pairs:
            if target == "VERIFY":
                stats["verify"] += 1
                continue
            if target in active["exclude"]:
                stats["excluded"] += 1
                continue
            if target not in active["classes"]:
                raise SystemExit(f"Unknown class '{target}' for {folder} in label_map.yaml")

            # A `label:` dataset only reads its own top level, so the README and
            # any sub-folders are ignored. `folders:` datasets read recursively.
            files = folder.iterdir() if "label" in dataset_cfg else folder.rglob("*")
            for path in sorted(files):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if not is_valid_image(path):
                    removed_rows.append({"file": rel(path), "kept_copy": "", "reason": "corrupted", "distance": ""})
                    continue
                items.append({"path": path, "label": target})

    print(
        f"\nRead {len(items)} valid images. Skipped {stats['verify']} VERIFY folders "
        f"and {stats['excluded']} excluded folders."
    )
    return items


# ---------------------------------------------------------------------------
# Step 3-4: duplicate removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(items, removed_rows):
    """Group images by SHA-256. Keep one per group; a mixed-label group is a conflict.

    Returns (kept items, number of exact duplicates removed, number of conflict images).
    """
    groups = defaultdict(list)  # sha256 -> [items with those exact bytes]
    for item in items:
        item["sha256"] = file_hash(item["path"])
        groups[item["sha256"]].append(item)

    kept, n_dupes, n_conflict = [], 0, 0
    for group in groups.values():
        first = group[0]
        labels = {item["label"] for item in group}
        if len(group) == 1:
            kept.append(first)
        elif len(labels) == 1:
            # Same picture, same label: keep the first copy, drop the others.
            kept.append(first)
            for dup in group[1:]:
                removed_rows.append({"file": rel(dup["path"]), "kept_copy": rel(first["path"]),
                                     "reason": "exact_duplicate", "distance": 0})
                n_dupes += 1
        else:
            # Same picture, different labels: we cannot trust either label, drop all.
            conflict = "label_conflict (" + " vs ".join(sorted(labels)) + ")"
            for item in group:
                note = "none (first copy of conflict group)" if item is first else f"none (exact copy of {rel(first['path'])})"
                removed_rows.append({"file": rel(item["path"]), "kept_copy": note, "reason": conflict, "distance": 0})
                n_conflict += 1

    # Keep the original order so results do not depend on dictionary order.
    order = {id(item): i for i, item in enumerate(items)}
    kept.sort(key=lambda item: order[id(item)])
    return kept, n_dupes, n_conflict


def remove_near_duplicates(items, removed_rows, max_distance):
    """Remove images whose perceptual hashes are within max_distance bits of each other.

    Works across ALL classes. Images are grouped in chains: if A is close to B
    and B is close to C, then A, B and C are one group, even if A and C are
    further apart. That way the file order can never hide a match.
      - Everyone in the group has the same label -> keep the first image, drop the rest (near_duplicate).
      - Labels disagree                          -> drop the whole group (label_conflict).
    Returns (kept items, near duplicates removed, conflict groups, conflict images).
    """
    for item in items:
        item["phash"] = perceptual_hash(item["path"])
    hashes = np.array([item["phash"] for item in items], dtype=np.uint64)

    # "Union-find": every image starts as its own group. Each close pair we find
    # merges two groups. group_of[i] points toward the image that names i's group.
    group_of = list(range(len(items)))

    def find_group(i):
        while group_of[i] != i:
            group_of[i] = group_of[group_of[i]]  # shortcut so later lookups are faster
            i = group_of[i]
        return i

    for i in range(len(items)):
        distances = hamming_distances(hashes[i], hashes[i + 1:])  # only later images
        for k in np.nonzero(distances <= max_distance)[0]:
            a, b = find_group(i), find_group(i + 1 + int(k))
            if a != b:
                group_of[max(a, b)] = min(a, b)  # the earlier image names the merged group

    groups = defaultdict(list)  # group name -> image indexes, in file order
    for i in range(len(items)):
        groups[find_group(i)].append(i)

    removed = set()
    n_near, n_conflict_groups, n_conflict_images = 0, 0, 0
    for members in groups.values():
        if len(members) == 1:
            continue
        labels = {items[i]["label"] for i in members}
        first = members[0]

        if len(labels) == 1:
            # Same label: keep the first image. Distance is to that kept copy; in a
            # chain (A~B~C) it can be above max_distance because C matched via B.
            for i in members[1:]:
                removed.add(i)
                dist = int(np.bitwise_count(hashes[i] ^ hashes[first]))
                removed_rows.append({"file": rel(items[i]["path"]), "kept_copy": rel(items[first]["path"]),
                                     "reason": "near_duplicate", "distance": dist})
                n_near += 1
        else:
            # Labels disagree: we cannot trust any of them, drop the whole group.
            # Each row names its closest image in the group; that distance is <= max_distance.
            n_conflict_groups += 1
            conflict = "label_conflict (" + " vs ".join(sorted(labels)) + ")"
            for i in members:
                removed.add(i)
                others = [j for j in members if j != i]
                dists = hamming_distances(hashes[i], hashes[others])
                closest = others[int(np.argmin(dists))]
                removed_rows.append({"file": rel(items[i]["path"]),
                                     "kept_copy": f"none (matches {rel(items[closest]['path'])})",
                                     "reason": conflict, "distance": int(dists.min())})
                n_conflict_images += 1

    kept = [item for i, item in enumerate(items) if i not in removed]
    return kept, n_near, n_conflict_groups, n_conflict_images


# ---------------------------------------------------------------------------
# Step 5-7: split, copy, manifests
# ---------------------------------------------------------------------------

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


def copy_splits_and_write_manifests(splits_by_class):
    """Copy images into data/<split>/<class>/ and write data/splits/<split>.csv.

    Filenames get a dataset + folder prefix so two datasets that both have a
    file called 1.jpg never overwrite each other. The CSV paths point at the
    copied files and are relative to the repo root.
    """
    for d in list(SPLIT_DIRS.values()) + [MANIFEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for split_name in SPLIT_NAMES:
        rows = []
        for class_name in sorted(splits_by_class):
            out_dir = SPLIT_DIRS[split_name] / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in splits_by_class[class_name][split_name]:
                src = item["path"]
                # data/raw/garbage12_enhanced/.../battery/battery7.jpg
                #   -> garbage12_enhanced__battery__battery7.jpg
                dataset = src.relative_to(RAW_DIR).parts[0]
                dst = out_dir / f"{dataset}__{src.parent.name}__{src.name}"
                shutil.copy2(src, dst)
                rows.append({"path": rel(dst), "label": class_name})

        rows.sort(key=lambda row: row["path"])
        with open(MANIFEST_DIR / f"{split_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Step 8: verify the output on disk
# ---------------------------------------------------------------------------

def check_no_overlap(max_distance):
    """Re-hash every copied file and confirm no exact or near duplicate spans two splits.

    This reads the files actually on disk in data/train, data/validation and
    data/test, so it checks the real output, not just our in-memory lists.
    """
    files = []  # (split name, path)
    for split_name, split_dir in SPLIT_DIRS.items():
        files += [(split_name, p) for p in sorted(split_dir.rglob("*")) if p.is_file()]

    # Exact: the same SHA-256 must never show up in two splits.
    splits_per_sha = defaultdict(set)
    for split_name, path in files:
        splits_per_sha[file_hash(path)].add(split_name)
    exact_cross = sum(1 for s in splits_per_sha.values() if len(s) > 1)

    # Near: compare every pair of perceptual hashes; count close pairs in different splits.
    hashes = np.array([perceptual_hash(path) for _, path in files], dtype=np.uint64)
    split_of = [split_name for split_name, _ in files]
    near_cross = 0
    for i in range(len(files)):
        distances = hamming_distances(hashes[i], hashes[i + 1:])
        for k in np.nonzero(distances <= max_distance)[0]:
            if split_of[i + 1 + k] != split_of[i]:
                near_cross += 1

    if exact_cross or near_cross:
        raise SystemExit(f"FAILED: {exact_cross} exact and {near_cross} near duplicates span two splits.")
    print("Overlap check passed: no exact or near duplicate appears in more than one split.")


# ---------------------------------------------------------------------------
# Step 9-10: class weights and the removed-files log
# ---------------------------------------------------------------------------

def write_class_weights(splits_by_class, num_classes):
    """Inverse-frequency weights from the TRAIN split only.

    weight = total_train / (num_classes * class_train_count)
    Rare classes get a weight above 1, common ones below 1, so the loss
    pays more attention to the classes with fewer images.
    """
    class_order = sorted(splits_by_class)  # alphabetical = torchvision ImageFolder order
    train_counts = {c: len(splits_by_class[c]["train"]) for c in class_order}
    total_train = sum(train_counts.values())
    if 0 in train_counts.values():
        raise SystemExit(f"A class has no training images: {train_counts}")

    percents = {c: round(100 * n / total_train, 2) for c, n in train_counts.items()}
    weights = {c: round(total_train / (num_classes * n), 4) for c, n in train_counts.items()}

    data = {
        "num_classes": num_classes,
        "class_order": class_order,
        "train_counts": train_counts,
        "train_total": total_train,
        "train_percent": percents,
        "weights": weights,
        "weights_in_class_order": [weights[c] for c in class_order],
        "formula": "total_train / (num_classes * class_train_count)",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(WEIGHTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"\nTrain split share and class weights ({rel(WEIGHTS_JSON)}):")
    for c in class_order:
        print(f"  {c:<12}{train_counts[c]:>6} images  {percents[c]:>6.2f}% of train   weight {weights[c]:.4f}")


def write_removed_csv(removed_rows):
    """Write every dropped file with the reason, sorted by file path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(REMOVED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "kept_copy", "reason", "distance"])
        writer.writeheader()
        writer.writerows(sorted(removed_rows, key=lambda row: row["file"]))


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
    parser = argparse.ArgumentParser(description="Build train/validation/test data from data/raw/.")
    parser.add_argument("--num-classes", type=int, choices=[3, 4], help="override num_classes from label_map.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count and print, but write nothing")
    args = parser.parse_args()

    config, active, num_classes = load_config(args.num_classes)
    ratios = config["split"]["ratios"]
    seed = config["split"]["seed"]
    max_distance = config["near_duplicate_max_distance"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit("split ratios in label_map.yaml must add up to 1.0")

    removed_rows = []  # one dict per dropped file, written to results/removed_duplicates.csv
    items = collect_images(config, active, removed_rows)
    if not items:
        raise SystemExit("No images found. Run src/data/download.py first.")

    items, n_exact, n_exact_conflict = remove_exact_duplicates(items, removed_rows)
    print(f"Checking near duplicates (perceptual hash, distance <= {max_distance}) across all classes...")
    items, n_near, n_conflict_groups, n_near_conflict = remove_near_duplicates(items, removed_rows, max_distance)

    n_corrupted = sum(1 for row in removed_rows if row["reason"] == "corrupted")
    print(f"\nRemoved {n_exact} exact duplicates, {n_near} near duplicates, "
          f"and {n_exact_conflict + n_near_conflict} images in label conflicts "
          f"({n_conflict_groups} near-duplicate groups"
          + (f" + {n_exact_conflict} exact-duplicate images" if n_exact_conflict else "")
          + f"). Skipped {n_corrupted} corrupted files. {len(items)} images remain.")

    by_class = defaultdict(list)
    for item in items:
        by_class[item["label"]].append(item)
    # Every active class must still have images, or the class order and the
    # weights would not line up with num_classes.
    missing = set(active["classes"]) - set(by_class)
    if missing:
        raise SystemExit(f"No images left for class(es) {sorted(missing)}. Check data/raw/ and label_map.yaml.")
    splits_by_class = {cls: split_list(by_class[cls], ratios, seed) for cls in sorted(by_class)}
    print_table(splits_by_class)

    if args.dry_run:
        print("\n(dry run: nothing was written)")
        return

    copy_splits_and_write_manifests(splits_by_class)
    write_removed_csv(removed_rows)
    write_class_weights(splits_by_class, num_classes)
    print()
    check_no_overlap(max_distance)
    print(f"Wrote data/train, data/validation, data/test, data/splits/*.csv, "
          f"{rel(REMOVED_CSV)}  (seed={seed}, {num_classes} classes)")


if __name__ == "__main__":
    main()
