"""
build_splits.py - turn the raw downloaded datasets into train / validation / test data.

What it does, in order:
  1. Read configs/label_map.yaml to learn which raw folder -> which training class.
  2. Collect every image from the folders listed in label_map.yaml under data/raw/.
     Any extra source (for example amy_images) is handled the same way once listed.
     - Skip folders marked VERIFY (not decided yet) or EXCLUDE (dropped by the team).
     - Skip folders whose class is excluded (hazardous when num_classes is 3).
     - Drop corrupted files that Pillow cannot fully read.
  3. Remove EXACT duplicates (identical bytes -> identical SHA-256 hash).
  4. Find NEAR duplicates using two image fingerprints, pHash and dHash.
     A pair matches only if pHash <= max_phash AND dHash <= max_dhash
     (both under `near_duplicates:` in label_map.yaml).
     - Same label:      keep the first image, drop the copy.
     - Different label: a LABEL CONFLICT. Nothing is deleted: the whole group is
       written to results/conflicts_to_review.csv and kept out of every split
       until a person decides which label is right.
     An image only joins a group if it matches the group's kept copy directly.
  5. Shuffle with seed 42 and split each class 70 / 15 / 15.
  6. Copy files into data/train, data/validation, data/test (one sub-folder per class).
  7. Write CSV manifests: data/splits/train.csv, validation.csv, test.csv (path, label).
  8. Re-check the copied files: no exact or near duplicate may sit in two splits.
  9. Compute two sets of class weights from the TRAIN split only -> results/class_weights.json.
 10. Write every removed file to results/removed_duplicates.csv.
 11. Print the always-guess-the-biggest-class baseline on the validation split.
 12. Check the arithmetic: images found - removed - held for review = images in the splits.
 13. Stamp the build in results/class_weights.json and print both stamps:
     - split_version:   the git commit of the code and config used.
     - manifest_sha256: a fingerprint of the three split CSVs, so it also changes
       when images are added to or removed from data/raw/ (which git ignores).
     Two models only compare fairly if both stamps match.

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
import math
import random
import shutil
import subprocess
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
CONFLICTS_CSV = RESULTS_DIR / "conflicts_to_review.csv"
WEIGHTS_JSON = RESULTS_DIR / "class_weights.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REMOVED_COLUMNS = ["file", "kept_copy", "reason", "distance", "dhash_distance"]
CONFLICT_COLUMNS = ["group", "file", "label", "matches", "distance", "dhash_distance"]


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


def image_hashes(path):
    """Return (pHash, dHash) of an image, each a 64-bit number.

    Both shrink the image to a tiny grayscale version:
      - pHash keeps the overall light/dark pattern (good at spotting re-saved or resized copies).
      - dHash keeps which way brightness changes from pixel to pixel (edges).
    Requiring BOTH to be close cuts false matches, like two different bottles
    that happen to have the same outline on a white background.
    """
    with Image.open(path) as img:
        phash = int(str(imagehash.phash(img)), 16)  # hex string -> integer
        dhash = int(str(imagehash.dhash(img)), 16)
    return phash, dhash


def hamming_distances(one_hash, all_hashes):
    """How many bits differ between one hash and every hash in a numpy array."""
    return np.bitwise_count(np.uint64(one_hash) ^ all_hashes)


def is_near_match(phash_dist, dhash_dist, rules):
    """True if two images count as the same picture under the label_map.yaml rules.

    Both fingerprints must agree. Works on single numbers or on whole numpy arrays.
    """
    return (phash_dist <= rules["max_phash"]) & (dhash_dist <= rules["max_dhash"])


def manifest_fingerprint():
    """First 12 characters of a SHA-256 over data/splits/train.csv, validation.csv and test.csv.

    The manifests list every image path and label in every split, so any change
    to the splits changes this fingerprint, even when the git commit stays the same.
    """
    h = hashlib.sha256()
    for split_name in SPLIT_NAMES:
        h.update((MANIFEST_DIR / f"{split_name}.csv").read_bytes())
    return h.hexdigest()[:12]


def get_split_version():
    """Short git commit hash of the code that built these splits, for example "fa59286".

    Adds "-dirty" if src/ or configs/ have changes that are not committed yet,
    because then the commit alone does not describe how the splits were made.
    Returns "unknown" if git is not available.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = git("rev-parse", "--short=7", "HEAD")
        dirty = git("status", "--porcelain", "--", "src", "configs")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


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
    """Return (list of image dicts {"path", "label"}, number of image files found).

    Rules for each raw folder:
      - target VERIFY               -> skip (not decided yet)
      - target EXCLUDE              -> skip (dropped by team decision)
      - target in active `exclude`  -> skip (e.g. hazardous in 3-class mode)
      - target not a known class    -> stop with an error (catches typos)
    Corrupted files count as found, are dropped, and are logged in removed_rows.
    """
    items = []
    n_found = 0
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
            if target == "EXCLUDE":
                stats["team_excluded"] += 1
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
                n_found += 1
                if not is_valid_image(path):
                    removed_rows.append({"file": rel(path), "kept_copy": "", "reason": "corrupted",
                                         "distance": "", "dhash_distance": ""})
                    continue
                items.append({"path": path, "label": target})

    print(f"\nFound {n_found} images. Skipped {stats['verify']} VERIFY folders, "
          f"{stats['team_excluded']} EXCLUDE folders, and {stats['excluded']} folders "
          f"outside the active classes.")
    return items, n_found


# ---------------------------------------------------------------------------
# Step 3-4: duplicate removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(items, removed_rows, review_rows):
    """Group images by SHA-256. Keep one per group; a mixed-label group goes to review.

    Returns (kept items, exact duplicates removed, held items).
    """
    groups = defaultdict(list)  # sha256 -> [items with those exact bytes]
    for item in items:
        groups[file_hash(item["path"])].append(item)

    kept, held, n_dupes = [], [], 0
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
                                     "reason": "exact_duplicate", "distance": 0, "dhash_distance": 0,
                                     "pair": dup["path"], "pair_with": first["path"]})
                n_dupes += 1
        else:
            # Same picture, different labels: hold the whole group for a person to review.
            group_id = next_group_id(review_rows)
            for item in group:
                other = next(o for o in group if o["label"] != item["label"])
                review_rows.append({"group": group_id, "file": rel(item["path"]), "label": item["label"],
                                    "matches": rel(other["path"]), "distance": 0, "dhash_distance": 0})
                held.append(item)

    # Keep the original order so results do not depend on dictionary order.
    order = {id(item): i for i, item in enumerate(items)}
    kept.sort(key=lambda item: order[id(item)])
    return kept, n_dupes, held


def next_group_id(review_rows):
    """Next conflict group number, like G001, G002, ... (in the order groups are found)."""
    return f"G{len({row['group'] for row in review_rows}) + 1:03d}"


def remove_near_duplicates(items, removed_rows, review_rows, rules, already_held=()):
    """Remove same-label near duplicates and hold mixed-label groups for review.

    Works across ALL classes, with no chaining. Walks the images in file order.
    Each image still in play becomes an anchor (a "kept copy") and is compared
    directly with every later image still in play. A later image joins the
    anchor's group only if it matches the ANCHOR itself (see is_near_match).
    If A matches B and B matches C, C is not grouped with A unless C also matches A.
      - Every match has the anchor's label -> keep the anchor, drop the matches (near_duplicate).
      - Any match has a different label    -> label conflict: the anchor and all its
        matches go to results/conflicts_to_review.csv and into no split. Nothing is deleted.
    Follow-up pass: a kept image that directly matches an image already dropped or
    held (including `already_held`, the exact-copy conflicts) with a DIFFERENT
    label is held for review too, so file order cannot hide a conflict.
    Returns (kept items, near duplicates removed, conflict groups, images held for review).
    """
    for item in items:
        item["phash"], item["dhash"] = image_hashes(item["path"])
    phashes = np.array([item["phash"] for item in items], dtype=np.uint64)
    dhashes = np.array([item["dhash"] for item in items], dtype=np.uint64)

    out = np.zeros(len(items), dtype=bool)  # True once an image is dropped or held for review
    n_near, n_conflict_groups, n_held = 0, 0, 0

    for i, anchor in enumerate(items):
        if out[i]:
            continue
        p_dist = hamming_distances(anchor["phash"], phashes)
        d_dist = hamming_distances(anchor["dhash"], dhashes)
        match = is_near_match(p_dist, d_dist, rules)
        # Only later images still in play can join this anchor's group.
        group = [j for j in np.nonzero(match)[0] if j > i and not out[j]]
        if not group:
            continue

        if all(items[j]["label"] == anchor["label"] for j in group):
            for j in group:
                out[j] = True
                removed_rows.append({"file": rel(items[j]["path"]), "kept_copy": rel(anchor["path"]),
                                     "reason": "near_duplicate", "distance": int(p_dist[j]),
                                     "dhash_distance": int(d_dist[j]),
                                     "pair": items[j]["path"], "pair_with": anchor["path"]})
                n_near += 1
        else:
            # Labels disagree: hold the anchor and its whole group for a person to check.
            n_conflict_groups += 1
            group_id = next_group_id(review_rows)
            # The anchor's row points at its closest match with a different label.
            cross = [j for j in group if items[j]["label"] != anchor["label"]]
            closest = min(cross, key=lambda j: (int(p_dist[j]), int(d_dist[j]), j))
            out[i] = True
            review_rows.append({"group": group_id, "file": rel(anchor["path"]), "label": anchor["label"],
                                "matches": rel(items[closest]["path"]),
                                "distance": int(p_dist[closest]), "dhash_distance": int(d_dist[closest])})
            for j in group:
                out[j] = True
                review_rows.append({"group": group_id, "file": rel(items[j]["path"]), "label": items[j]["label"],
                                    "matches": rel(anchor["path"]),
                                    "distance": int(p_dist[j]), "dhash_distance": int(d_dist[j])})
            n_held += 1 + len(group)

    # Follow-up pass. Compare each kept image with every image that is out (dropped
    # or held) plus the exact-copy conflicts. Repeat until nothing new is held,
    # because a newly held image can itself match another kept image.
    extra = list(already_held)
    for item in extra:
        item["phash"], item["dhash"] = image_hashes(item["path"])
    others = [items[j] for j in np.nonzero(out)[0]] + extra
    while others:
        o_phash = np.array([o["phash"] for o in others], dtype=np.uint64)
        o_dhash = np.array([o["dhash"] for o in others], dtype=np.uint64)
        o_label = np.array([o["label"] for o in others])
        newly_held = []
        for i, item in enumerate(items):
            if out[i]:
                continue
            p_dist = hamming_distances(item["phash"], o_phash)
            d_dist = hamming_distances(item["dhash"], o_dhash)
            hits = np.nonzero(is_near_match(p_dist, d_dist, rules) & (o_label != item["label"]))[0]
            if len(hits) == 0:
                continue
            k = min(hits, key=lambda h: (int(p_dist[h]), int(d_dist[h]), int(h)))  # closest look-alike
            out[i] = True
            match_path = rel(others[k]["path"])
            # Join the look-alike's group if it is already in the review list ...
            group_id = next((r["group"] for r in review_rows if r["file"] == match_path), None)
            if group_id is None:
                # ... otherwise it was removed as a copy: start a new group and name its kept copy.
                n_conflict_groups += 1
                group_id = next_group_id(review_rows)
                kept_copy = next((r["kept_copy"] for r in removed_rows if r["file"] == match_path), "")
                if kept_copy:
                    match_path = f"{match_path} (removed as a copy of {kept_copy})"
            review_rows.append({"group": group_id, "file": rel(item["path"]),
                                "label": item["label"], "matches": match_path,
                                "distance": int(p_dist[k]), "dhash_distance": int(d_dist[k])})
            n_held += 1
            newly_held.append(item)
        others = newly_held  # only the new ones can create further matches

    kept = [item for i, item in enumerate(items) if not out[i]]
    return kept, n_near, n_conflict_groups, n_held


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

def split_files_on_disk():
    """List (split name, class name, path) for every copied image in data/<split>/<class>/."""
    files = []
    for split_name, split_dir in SPLIT_DIRS.items():
        files += [(split_name, p.parent.name, p) for p in sorted(split_dir.rglob("*")) if p.is_file()]
    return files


def check_no_overlap(rules):
    """Re-hash every copied file and confirm no exact or near duplicate spans two splits.

    This reads the files actually on disk in data/train, data/validation and
    data/test, so it checks the real output, not just our in-memory lists.
    Near duplicates use the same rules as the removal step.
    """
    files = split_files_on_disk()

    # Exact: the same SHA-256 must never show up in two splits.
    splits_per_sha = defaultdict(set)
    for split_name, _, path in files:
        splits_per_sha[file_hash(path)].add(split_name)
    exact_cross = sum(1 for s in splits_per_sha.values() if len(s) > 1)

    # Near: compare every pair; count matching pairs that sit in different splits.
    hashes = [image_hashes(path) for _, _, path in files]
    phashes = np.array([h[0] for h in hashes], dtype=np.uint64)
    dhashes = np.array([h[1] for h in hashes], dtype=np.uint64)
    splits = np.array([f[0] for f in files])
    near_cross = 0
    for i in range(len(files)):
        later = slice(i + 1, None)
        match = is_near_match(hamming_distances(phashes[i], phashes[later]),
                              hamming_distances(dhashes[i], dhashes[later]), rules)
        near_cross += int(np.sum(match & (splits[later] != splits[i])))

    if exact_cross or near_cross:
        raise SystemExit(f"FAILED: {exact_cross} exact and {near_cross} near duplicates span two splits.")
    print("Overlap check passed: no exact or near duplicate appears in more than one split.")


def check_counts(n_found, out_counts, n_final, n_removed_rows, n_review_rows):
    """Stop with a clear message unless: images found - removed - held for review = images in the splits.

    Also checks that results/removed_duplicates.csv has one row per removed image
    and results/conflicts_to_review.csv has one row per image held for review.
    (A plain `assert` would be skipped if Python runs with -O, so this uses SystemExit.)
    """
    n_out = sum(out_counts.values())
    n_held = out_counts["held for review"]
    parts = " - ".join(f"{n} {name}" for name, n in out_counts.items())
    print(f"\nCount check: {n_found} found - {parts} = {n_found - n_out}; "
          f"{n_final} in the splits; {n_removed_rows} rows in {rel(REMOVED_CSV)}; "
          f"{n_review_rows} rows in {rel(CONFLICTS_CSV)}.")
    if n_found - n_out != n_final or n_removed_rows != n_out - n_held or n_review_rows != n_held:
        raise SystemExit(f"FAILED count check: {n_found} found - {n_out} out = {n_found - n_out}, but the splits "
                         f"hold {n_final} images, the removed list has {n_removed_rows} rows (expected "
                         f"{n_out - n_held}) and the review list has {n_review_rows} rows (expected {n_held}).")
    print("Count check passed.")


# ---------------------------------------------------------------------------
# Step 9-11: class weights, removed-files log, baseline
# ---------------------------------------------------------------------------

def compute_class_weights(splits_by_class, num_classes):
    """Two sets of class weights, both from the TRAIN split only.

    inverse_frequency:      total_train / (num_classes * class_count)
        Rare classes get a weight above 1, common ones below 1.
    sqrt_inverse_frequency: sqrt(total_train / (num_classes * class_count)),
        then divided by the average so the weights average to exactly 1.
        A gentler push toward rare classes; often more stable in training.
    """
    class_order = sorted(splits_by_class)  # alphabetical = torchvision ImageFolder order
    train_counts = {c: len(splits_by_class[c]["train"]) for c in class_order}
    total_train = sum(train_counts.values())
    if 0 in train_counts.values():
        raise SystemExit(f"A class has no training images: {train_counts}")

    inverse = {c: total_train / (num_classes * n) for c, n in train_counts.items()}
    sqrt_raw = {c: math.sqrt(w) for c, w in inverse.items()}
    sqrt_mean = sum(sqrt_raw.values()) / len(sqrt_raw)
    sqrt_norm = {c: w / sqrt_mean for c, w in sqrt_raw.items()}

    def weight_set(formula, by_class):
        rounded = {c: round(by_class[c], 4) for c in class_order}
        return {"formula": formula, "by_class": rounded, "in_class_order": [rounded[c] for c in class_order]}

    return {
        "split_version": get_split_version(),
        "num_classes": num_classes,
        "class_order": class_order,
        "train_counts": train_counts,
        "train_total": total_train,
        "train_percent": {c: round(100 * n / total_train, 2) for c, n in train_counts.items()},
        "weights": {
            "inverse_frequency": weight_set(
                "total_train / (num_classes * class_train_count)", inverse),
            "sqrt_inverse_frequency": weight_set(
                "sqrt(total_train / (num_classes * class_train_count)), divided by the mean so the weights average to 1",
                sqrt_norm),
        },
    }


def print_class_weights(data):
    """Show train share and both weight sets, one line per class."""
    w = data["weights"]
    print(f"\nTrain split share and class weights ({rel(WEIGHTS_JSON)}):")
    print(f"  {'class':<12}{'train':>7}{'% train':>10}{'inverse':>10}{'sqrt inv':>10}")
    for c in data["class_order"]:
        print(f"  {c:<12}{data['train_counts'][c]:>7}{data['train_percent'][c]:>9.2f}%"
              f"{w['inverse_frequency']['by_class'][c]:>10.4f}{w['sqrt_inverse_frequency']['by_class'][c]:>10.4f}")


def print_majority_baseline(splits_by_class):
    """Accuracy and macro F1 on validation if a model always guessed the biggest train class.

    For the guessed class: precision = accuracy, recall = 1, so F1 = 2*acc/(acc+1).
    Every other class is never predicted, so its F1 is 0.
    Macro F1 is the plain average of the per-class F1 scores.
    """
    majority = max(splits_by_class, key=lambda c: len(splits_by_class[c]["train"]))
    n_val = sum(len(s["validation"]) for s in splits_by_class.values())
    accuracy = len(splits_by_class[majority]["validation"]) / n_val
    f1_majority = 2 * accuracy / (accuracy + 1)
    macro_f1 = f1_majority / len(splits_by_class)
    print(f"\nBaseline 'always guess {majority}' on validation ({n_val} images): "
          f"accuracy {100 * accuracy:.2f}%, macro F1 {macro_f1:.4f}")


def write_removed_csv(removed_rows):
    """Write every dropped file with the reason, sorted by file path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(REMOVED_CSV, "w", newline="") as f:
        # extrasaction="ignore" skips the helper keys ("pair", "pair_with") used by the spot check.
        writer = csv.DictWriter(f, fieldnames=REMOVED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(removed_rows, key=lambda row: row["file"]))


def write_conflicts_csv(review_rows):
    """Write every label-conflict group for a person to check. Header only if there are none."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(CONFLICTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONFLICT_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)  # already in group order


def print_table(splits_by_class):
    """Print class counts per split as a small table. Returns the grand total."""
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
    return totals["total"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_duplicates(config, active):
    """Steps 2-4. Returns (kept items, removed rows, review rows, images found, counts of images taken out).

    Shared with src/data/review_near_dups.py so the spot check uses the exact same rules.
    """
    rules = config["near_duplicates"]
    removed_rows = []  # one dict per dropped file  -> results/removed_duplicates.csv
    review_rows = []   # one dict per held-back file -> results/conflicts_to_review.csv
    items, n_found = collect_images(config, active, removed_rows)
    if not items:
        raise SystemExit("No images found. Run src/data/download.py first.")
    n_corrupted = len(removed_rows)

    items, n_exact, exact_held = remove_exact_duplicates(items, removed_rows, review_rows)
    n_exact_held = len(exact_held)
    n_exact_groups = len({row["group"] for row in review_rows})
    print(f"Checking near duplicates across all classes: pHash <= {rules['max_phash']} "
          f"and dHash <= {rules['max_dhash']}...")
    items, n_near, n_groups, n_near_held = remove_near_duplicates(items, removed_rows, review_rows, rules,
                                                                  already_held=exact_held)

    out_counts = {
        "corrupted": n_corrupted,
        "exact duplicates": n_exact,
        "near duplicates": n_near,
        "held for review": n_exact_held + n_near_held,
    }
    print(f"\nRemoved {n_exact} exact duplicates and {n_near} near duplicates. "
          f"Held {n_exact_held + n_near_held} images in {n_exact_groups + n_groups} label-conflict groups "
          f"for review ({rel(CONFLICTS_CSV)}). Dropped {n_corrupted} corrupted files. "
          f"{len(items)} images go into the splits.")
    return items, removed_rows, review_rows, n_found, out_counts


def main():
    parser = argparse.ArgumentParser(description="Build train/validation/test data from data/raw/.")
    parser.add_argument("--num-classes", type=int, choices=[3, 4], help="override num_classes from label_map.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count and print, but write nothing")
    args = parser.parse_args()

    config, active, num_classes = load_config(args.num_classes)
    ratios = config["split"]["ratios"]
    seed = config["split"]["seed"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit("split ratios in label_map.yaml must add up to 1.0")

    items, removed_rows, review_rows, n_found, out_counts = find_duplicates(config, active)

    by_class = defaultdict(list)
    for item in items:
        by_class[item["label"]].append(item)
    # Every active class must still have images, or the class order and the
    # weights would not line up with num_classes.
    missing = set(active["classes"]) - set(by_class)
    if missing:
        raise SystemExit(f"No images left for class(es) {sorted(missing)}. Check data/raw/ and label_map.yaml.")
    splits_by_class = {cls: split_list(by_class[cls], ratios, seed) for cls in sorted(by_class)}
    n_in_splits = print_table(splits_by_class)

    weights = compute_class_weights(splits_by_class, num_classes)
    print_class_weights(weights)
    print_majority_baseline(splits_by_class)

    if args.dry_run:
        check_counts(n_found, out_counts, n_in_splits, len(removed_rows), len(review_rows))
        print(f"\nSplit version: {weights['split_version']}  (dry run: nothing was written)")
        return

    copy_splits_and_write_manifests(splits_by_class)
    weights["manifest_sha256"] = manifest_fingerprint()
    write_removed_csv(removed_rows)
    write_conflicts_csv(review_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(WEIGHTS_JSON, "w") as f:
        json.dump(weights, f, indent=2)
        f.write("\n")
    print()
    check_no_overlap(config["near_duplicates"])

    # Final arithmetic check against what is really on disk.
    n_on_disk = len(split_files_on_disk())
    with open(REMOVED_CSV) as f:
        n_removed_rows = sum(1 for _ in csv.DictReader(f))
    with open(CONFLICTS_CSV) as f:
        n_review_rows = sum(1 for _ in csv.DictReader(f))
    check_counts(n_found, out_counts, n_on_disk, n_removed_rows, n_review_rows)
    print(f"Wrote data/train, data/validation, data/test, data/splits/*.csv, {rel(REMOVED_CSV)}, "
          f"{rel(CONFLICTS_CSV)}, {rel(WEIGHTS_JSON)}  (seed={seed}, {num_classes} classes)")
    print(f"Split version: {weights['split_version']}  manifest_sha256: {weights['manifest_sha256']}  "
          f"(both saved in {rel(WEIGHTS_JSON)})")


if __name__ == "__main__":
    main()
