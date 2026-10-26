"""
build_splits.py - turn the raw downloaded datasets into train / validation / test / wild_test data.

What it does, in order:
  1. Read configs/label_map.yaml to learn which raw folder -> which training class.
  2. Collect every image from the folders listed in label_map.yaml under data/raw/.
     Any extra source (for example amy_images) is handled the same way once listed.
     - Folder names match case-insensitively, and common variants also count
       (for example "Recycle" -> recycling), see `folder_aliases:` in label_map.yaml.
     - A dataset marked `optional: true` whose folder is missing is skipped with a
       warning; any other missing dataset folder stops the run.
     - Skip folders marked VERIFY (not decided yet) or EXCLUDE (dropped by the team).
     - Skip folders whose class is excluded (hazardous when num_classes is 3).
     - iPhone HEIC/HEIF photos are read with pillow-heif and copied as JPEG.
     - New-source photos are turned upright by their EXIF orientation tag before
       hashing and copying, and saved without the tag. Base images are copied byte for byte.
     - Files that are not images (.DS_Store, videos, README...) are skipped and counted.
     - Drop corrupted files, and image formats we cannot read, and log them.
  3. Remove EXACT duplicates (identical bytes -> identical SHA-256 hash).
  4. Find NEAR duplicates using two image fingerprints, pHash and dHash.
     A pair matches only if pHash <= max_phash AND dHash <= max_dhash
     (both under `near_duplicates:` in label_map.yaml).
     - Same label:      keep the first image, drop the copy.
     - Different label: a LABEL CONFLICT. Nothing is deleted: the whole group is
       written to results/conflicts_to_review.csv and kept out of every split
       until a person decides which label is right.
     An image only joins a group if it matches the group's kept copy directly.
  5. Split in two stages so new photos never move old ones:
     - BASE datasets (`base_datasets:` in label_map.yaml, the Kaggle set) are
       de-duplicated and shuffled 70 / 15 / 15 with seed 42 exactly as if nothing
       else existed. Their test split is the FROZEN BENCHMARK.
     - NEW sources (taco_crops, realwaste, amy_images, team_photos, personal_battery)
       are first checked against every base image and every LOCKED image
       (configs/frozen/wild_split_lock.csv): a copy is dropped (or held, if its label
       differs) and the base or locked image stays. They never enter the benchmark.
  5b. Drop the KNOWN DUPLICATES listed in label_map.yaml: copies a person confirmed by
     eye in the leakage check. The listed `file` is dropped, its `train_match` is kept.
     This happens AFTER the base split is decided, so no other image changes split.
  5c. New recycling images from the capped sources are limited to `recycling_cap`
     (`wild_split:` in label_map.yaml); locked images count first and are always kept.
  5d. New images are placed by split_wild_images: locked images keep their frozen
     split (train, validation or wild_test); every unlocked image goes ONLY to train
     or validation (70 / 15). Crops of one photo and look-alikes stay in one split,
     and an unlocked look-alike of a benchmark or wild test image is dropped.
  6. Copy files into data/train, data/validation, data/test, data/wild_test.
  7. Write CSV manifests: data/splits/train.csv, validation.csv, test.csv, wild_test.csv.
  7b. Stop unless test.csv and wild_test.csv are byte-identical to the frozen copies
     in configs/frozen/, or if any manifest path comes from the held-out data/raw/camera_test/.
  8. Re-check the copied files: no exact or near duplicate may sit in two splits.
  9. Compute two sets of class weights from the TRAIN split only -> results/class_weights.json.
 10. Write every removed file to results/removed_duplicates.csv.
 11. Print the always-guess-the-biggest-class baseline on the validation split.
 12. Check the arithmetic: images found - removed - held for review = images in the splits.
 13. Stamp the build in results/class_weights.json and print both stamps:
     - split_version:   the last git commit that changed this script or configs/.
     - manifest_sha256: a fingerprint of the split CSVs, so it also changes
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
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image, ImageOps

# iPhone photos are HEIC/HEIF. Pillow cannot open them on its own; the
# pillow-heif package adds that. If it is missing, HEIC files are reported as
# "unsupported format" with a hint, never skipped silently.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

# --- Paths. Everything is relative to the repo root, two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"
# The base (Kaggle) split. Its "test" is the frozen benchmark (label_map.yaml `frozen:`).
SPLIT_NAMES = ["train", "validation", "test"]
# Non-Kaggle images: the TACO and RealWaste images of the frozen build keep their locked
# split (train, validation or wild_test, configs/frozen/wild_split_lock.csv); every other
# image goes only to train or validation, so nothing new enters either test set.
WILD_TEST = "wild_test"
ALL_SPLITS = SPLIT_NAMES + [WILD_TEST]
SPLIT_DIRS = {name: REPO_ROOT / "data" / name for name in ALL_SPLITS}
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
# Demo-room photos (webcam / Pi camera). Held out for src/eval/score_camera_test.py
# and NEVER split or trained on: a dataset reading from it stops the build, and so
# does any manifest path that mentions it (check_no_camera_test()).
CAMERA_TEST_NAME = "camera_test"
CAMERA_TEST_DIR = RAW_DIR / CAMERA_TEST_NAME
REMOVED_CSV = RESULTS_DIR / "removed_duplicates.csv"
CONFLICTS_CSV = RESULTS_DIR / "conflicts_to_review.csv"
WEIGHTS_JSON = RESULTS_DIR / "class_weights.json"

# Images we copy as they are (torchvision's ImageFolder can read all of these).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Images we can read but copy as .jpg, because ImageFolder ignores these types.
CONVERT_TO_JPEG = {".heic", ".heif", ".avif"}
# Image types we recognise but cannot read: camera RAW, Photoshop, vector, animations.
KNOWN_UNSUPPORTED = {".dng", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf",
                     ".psd", ".svg", ".gif", ".ico", ".jxl"}
# The files that decide how splits are made. split_version only changes when one of these changes.
SPLIT_SOURCES = ["src/data/build_splits.py", "configs"]
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


def image_hashes(path, upright=False):
    """Return (pHash, dHash) of an image, each a 64-bit number.

    Both shrink the image to a tiny grayscale version:
      - pHash keeps the overall light/dark pattern (good at spotting re-saved or resized copies).
      - dHash keeps which way brightness changes from pixel to pixel (edges).
    Requiring BOTH to be close cuts false matches, like two different bottles
    that happen to have the same outline on a white background.

    upright=True first turns the photo the way its EXIF orientation tag says
    (phones store sideways pixels plus a "rotate me" tag). Used for new photo
    sources, so a sideways copy and an upright copy of one photo still match.
    """
    with Image.open(path) as img:
        if upright:
            img = ImageOps.exif_transpose(img)
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
    """First 12 characters of a SHA-256 over data/splits/train.csv, validation.csv, test.csv
    and wild_test.csv.

    The manifests list every image path and label in every split, so any change
    to the splits changes this fingerprint, even when the git commit stays the same.
    wild_test.csv is only included when it lists images, so a build without any
    new source gives the same fingerprint as before wild_test existed.
    """
    h = hashlib.sha256()
    for split_name in ALL_SPLITS:
        path = MANIFEST_DIR / f"{split_name}.csv"
        data = path.read_bytes()
        if split_name == WILD_TEST and data.count(b"\n") <= 1:  # header only
            continue
        h.update(data)
    return h.hexdigest()[:12]


def get_split_version():
    """Short git commit hash of the code that built these splits, for example "fa59286".

    It is the LAST commit that changed one of SPLIT_SOURCES (this script and
    configs/), not simply the newest commit. Docs, results or helper scripts
    (plot_counts.py, check_leakage.py, ...) do not change how splits are made,
    so they do not change the version, and rebuilding on a later commit gives
    the same stamp. Adds "-dirty" if those files have uncommitted changes.
    Returns "unknown" if git is not available.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = git("log", "-1", "--format=%h", "--abbrev=7", "--", *SPLIT_SOURCES)
        dirty = git("status", "--porcelain", "--", *SPLIT_SOURCES)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def can_read_format(suffix):
    """True if this file type can be opened on this computer right now."""
    suffix = suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    if suffix in {".heic", ".heif"}:
        return HEIF_AVAILABLE
    if suffix == ".avif":
        return ".avif" in Image.registered_extensions()
    return False


def normalise_folder_name(name):
    """Lower-case and drop spaces, dashes and underscores: "Hazardous_Waste" -> "hazardouswaste"."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


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
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as err:
        raise SystemExit(
            f"configs/label_map.yaml could not be read:\n{err}\n"
            "If you just switched on amy_images: delete only the two characters \"# \" "
            "(hash and one space) at the start of each of its 7 lines, so the spaces "
            "in front stay the same.")

    num_classes = num_classes_override or config["num_classes"]
    if num_classes not in config["class_sets"]:
        raise SystemExit(f"label_map.yaml has no class_sets entry for {num_classes}. Options: 3 or 4.")

    active = config["class_sets"][num_classes]
    print(f"Using {num_classes} classes: {active['classes']}  (excluded: {active['exclude'] or 'none'})")
    return config, active, num_classes


def missing_dataset_message(name, dataset_cfg):
    """Explain where a dataset's images should be, for the error messages below."""
    if "label" in dataset_cfg:
        where = f"data/raw/{name}/ (photos directly inside it)"
    else:
        root = f"{dataset_cfg['root']}/" if dataset_cfg.get("root") else ""
        where = f"data/raw/{name}/{root}<class folder>/, with class folders " + ", ".join(dataset_cfg["folders"])
    return (f"Put the images in {where}. If this is a Kaggle dataset, run python src/data/download.py. "
            f"To leave it out, put \"# \" back in front of its lines in configs/label_map.yaml.")


def folders_for_dataset(name, dataset_cfg, aliases, show_mapping=False):
    """Return a list of (folder_path, target_class) pairs for one dataset.

    Two shapes are allowed in the yaml:
      - `root` + `folders`: a dict of folder name -> target class
      - `label`: every image inside data/raw/<name>/ gets this class
    Folder names match case-insensitively, and the names under `folder_aliases:`
    in label_map.yaml also count ("Recycle" or "recyclables" -> the recycling
    folder). Several real folders may match the same entry.
    The order follows label_map.yaml, so the result never depends on folder names.
    Hidden folders and __MACOSX (zip leftovers) are ignored, and named in a note.
    With show_mapping=True every folder is printed with the class it was read as.
    Stops with a clear message if the dataset folder is missing, if a real folder
    matches nothing, or if a listed folder has no match. (A dataset marked
    `optional: true` with a missing folder is skipped by collect_images() before
    this function is called, so here a missing folder is always an error.)
    """
    dataset_dir = RAW_DIR / name
    if not dataset_dir.is_dir():
        raise SystemExit(f"data/raw/{name}/ not found, but \"{name}\" is switched on in "
                         f"configs/label_map.yaml. {missing_dataset_message(name, dataset_cfg)}")

    if "label" in dataset_cfg:
        return [(dataset_dir, dataset_cfg["label"])]

    dataset_root = dataset_dir / (dataset_cfg.get("root") or "")
    if not dataset_root.is_dir():
        raise SystemExit(f"label_map.yaml root '{dataset_cfg.get('root')}' has no folder at {rel(dataset_root)}")

    # Real sub-folders. Hidden ones (.Trashes) and __MACOSX (left by unzipping on a Mac) are not photos.
    folders = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    ignored = [p for p in folders if p.name.startswith(".") or p.name == "__MACOSX"]
    real = [p for p in folders if p not in ignored]
    if ignored:
        print(f"  Ignored folders in {rel(dataset_root)}/ (system or zip metadata): {', '.join(p.name for p in ignored)}")
    if not real:
        raise SystemExit(f"{rel(dataset_root)}/ has no folders inside. {missing_dataset_message(name, dataset_cfg)}")

    # Every accepted spelling (normalised) -> the folder entry it belongs to.
    accepted = {}
    for key in dataset_cfg["folders"]:
        for spelling in [key] + list(aliases.get(key, [])):
            accepted.setdefault(normalise_folder_name(spelling), key)

    matches = defaultdict(list)  # folder entry -> real folders that match it
    unknown = []
    for folder in real:
        key = accepted.get(normalise_folder_name(folder.name))
        if key is None:
            unknown.append(folder.name)
        else:
            matches[key].append(folder)

    if unknown:
        expected = "; ".join(f"{key} (or {', '.join(aliases[key])})" if aliases.get(key) else key
                             for key in dataset_cfg["folders"])
        raise SystemExit(f"{rel(dataset_root)}/ has folder(s) that match no class: {', '.join(unknown)}.\n"
                         f"  Folders found:    {', '.join(f.name for f in real)}\n"
                         f"  Folders expected: {expected}\n"
                         f"  Rename the folder, or add the name under folder_aliases: in configs/label_map.yaml.")
    missing = [key for key in dataset_cfg["folders"] if key not in matches]
    if missing:
        raise SystemExit(f"label_map.yaml folder(s) {', '.join(missing)} have no folder in {rel(dataset_root)}/.\n"
                         f"  Folders found: {', '.join(f.name for f in real)}")

    pairs = []
    for key, target in dataset_cfg["folders"].items():  # label_map.yaml order
        for folder in matches[key]:
            if show_mapping or folder.name != key:
                print(f"  {rel(folder)}/ -> {key} ({target})")
            pairs.append((folder, target))
    return pairs


def collect_images(config, active, removed_rows):
    """Return (list of image dicts {"path", "label", "folder"}, images found, skip counts).

    Rules for each raw folder:
      - target VERIFY               -> skip (not decided yet)
      - target EXCLUDE              -> skip (dropped by team decision)
      - target in active `exclude`  -> skip (e.g. hazardous in 3-class mode)
      - target not a known class    -> stop with an error (catches typos)
    Every file inside is looked at, sub-folders included, and sorted into one of:
      - not an image   (.DS_Store, ._ files, videos, README...): skipped and counted
      - outside a class folder (a photo lying loose in data/raw/<dataset>/): skipped and counted
      - unsupported    (an image type we cannot read, e.g. RAW): counted as found,
                        dropped, logged in removed_rows
      - corrupted      (readable type, but broken): counted as found, dropped, logged
      - image          kept
    """
    items = []
    n_found = 0
    stats = Counter()
    skipped = Counter()            # reason -> how many files
    examples = defaultdict(list)   # reason -> a few file paths to show
    aliases = config.get("folder_aliases") or {}
    heif_missing = []              # HEIC/HEIF photos found while pillow-heif is not installed

    def note(reason, path):
        skipped[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(rel(path))

    for dataset_name, dataset_cfg in config["datasets"].items():
        found_here = 0
        is_new_source = dataset_name not in (config.get("base_datasets") or [])
        # The camera test set is held out. Nothing may read from it, not even a
        # dataset under another name whose root or symlink points inside it.
        dataset_root = (RAW_DIR / dataset_name / (dataset_cfg.get("root") or "")).resolve()
        held_out = CAMERA_TEST_DIR.resolve()
        if dataset_name == CAMERA_TEST_NAME or dataset_root == held_out or held_out in dataset_root.parents:
            raise SystemExit(f"STOP: \"{dataset_name}\" in configs/label_map.yaml reads data/raw/{CAMERA_TEST_NAME}/. "
                             f"Those are the held-out demo-room photos and must never be split or trained on. "
                             f"Remove that entry. Score them with: python src/eval/score_camera_test.py ...")
        # `optional: true` in label_map.yaml: a machine without this folder warns
        # and builds from the other datasets. Without it, a missing folder stops the run.
        if dataset_cfg.get("optional") and not (RAW_DIR / dataset_name).is_dir():
            print(f"  WARNING: data/raw/{dataset_name}/ not found. It is marked optional in "
                  f"configs/label_map.yaml, so the build continues WITHOUT it. "
                  f"The splits will not contain its images.")
            continue
        pairs = folders_for_dataset(dataset_name, dataset_cfg, aliases, show_mapping=is_new_source)
        # Files lying loose next to the class folders belong to no class: report them.
        if "folders" in dataset_cfg:
            for path in sorted(p for p in (RAW_DIR / dataset_name / (dataset_cfg.get("root") or "")).iterdir()
                               if p.is_file()):
                looks_like_photo = path.suffix.lower() in IMAGE_EXTENSIONS | CONVERT_TO_JPEG | KNOWN_UNSUPPORTED
                note("outside a class folder" if looks_like_photo and not path.name.startswith(".") else "not an image", path)
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

            for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                suffix = path.suffix.lower()
                # Hidden files (.DS_Store) and macOS "._" copies are never photos.
                if path.name.startswith("."):
                    note("not an image", path)
                    continue
                if suffix not in IMAGE_EXTENSIONS | CONVERT_TO_JPEG | KNOWN_UNSUPPORTED:
                    note("not an image", path)
                    continue
                n_found += 1
                found_here += 1
                if suffix in {".heic", ".heif"} and not HEIF_AVAILABLE:
                    heif_missing.append(path)
                    continue
                if not can_read_format(suffix):
                    note("unsupported format", path)
                    removed_rows.append({"file": rel(path), "kept_copy": "", "reason": "unsupported_format",
                                         "distance": "", "dhash_distance": ""})
                    continue
                if not is_valid_image(path):
                    note("corrupted", path)
                    removed_rows.append({"file": rel(path), "kept_copy": "", "reason": "corrupted",
                                         "distance": "", "dhash_distance": ""})
                    continue
                # New photo sources (not in base_datasets) are turned upright by their
                # EXIF tag before hashing and copying. Base (Kaggle) images are left as they are.
                items.append({"path": path, "label": target, "folder": folder, "upright": is_new_source})

        # An enabled photo folder with nothing usable in it is almost always a mistake...
        if "folders" in dataset_cfg and found_here == 0 and any(
                t in active["classes"] for t in dataset_cfg["folders"].values()):
            # ...except for an optional one: its empty class folders are tracked in git
            # (team_photos), so most machines have the folders and no photos yet.
            if dataset_cfg.get("optional"):
                print(f"  WARNING: data/raw/{dataset_name}/ has no images yet. It is marked optional in "
                      f"configs/label_map.yaml, so the build continues WITHOUT it.")
                continue
            raise SystemExit(f"data/raw/{dataset_name}/ has no images. {missing_dataset_message(dataset_name, dataset_cfg)}")

    print(f"\nFound {n_found} images. Skipped {stats['verify']} VERIFY folders, "
          f"{stats['team_excluded']} EXCLUDE folders, and {stats['excluded']} folders "
          f"outside the active classes.")
    reasons = ["not an image", "outside a class folder", "unsupported format", "corrupted"]
    print("Files skipped: " + ", ".join(f"{skipped[r]} {r}" for r in reasons))
    for reason in reasons:
        if skipped[reason]:
            more = f" (+{skipped[reason] - len(examples[reason])} more)" if skipped[reason] > len(examples[reason]) else ""
            print(f"  {reason}: {', '.join(examples[reason])}{more}")
    # Dropping every iPhone photo would quietly ruin the dataset, so stop instead.
    if heif_missing:
        raise SystemExit(f"Found {len(heif_missing)} iPhone HEIC/HEIF photos (for example {rel(heif_missing[0])}), "
                         f"but the pillow-heif package that opens them is not installed.\n"
                         f"Run:  pip install -r requirements.txt   and then run this script again.")
    return items, n_found, skipped


# ---------------------------------------------------------------------------
# Step 3-4: duplicate removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(items, removed_rows, review_rows):
    """Group images by SHA-256. Keep one per group; a mixed-label group goes to review.

    Returns (kept items, exact duplicates removed, held items).
    """
    groups = defaultdict(list)  # sha256 -> [items with those exact bytes]
    for item in items:
        if "sha256" not in item:  # already known if this image was checked in an earlier stage
            item["sha256"] = file_hash(item["path"])
        groups[item["sha256"]].append(item)

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


def drop_copies_of_base(new_items, base_items, removed_rows, review_rows, rules, reason="copy_of_base"):
    """Keep new images away from the frozen base set: a copy of any base image is never used.

    Exact (same SHA-256) or near copies (the near_duplicates rule) of a base image:
      - same label      -> dropped, logged as "copy_of_base" in results/removed_duplicates.csv
      - different label -> held for review in results/conflicts_to_review.csv
    The BASE image always stays where it is. (Before the benchmark was frozen a
    label conflict held both images; now a new photo can never remove or move a
    base image, so the benchmark test set cannot change.)
    `base_items` includes base images already held for review in stage A.
    Returns (new images still kept, copies dropped, images held).
    """
    if not new_items or not base_items:
        return new_items, 0, []
    for item in list(new_items) + list(base_items):
        if "sha256" not in item:
            item["sha256"] = file_hash(item["path"])
        if "phash" not in item:
            item["phash"], item["dhash"] = image_hashes(item["path"], item.get("upright", False))
    base_p = np.array([b["phash"] for b in base_items], dtype=np.uint64)
    base_d = np.array([b["dhash"] for b in base_items], dtype=np.uint64)
    base_sha = {b["sha256"]: k for k, b in enumerate(base_items)}
    kept, held, n_copies = [], [], 0
    for item in new_items:
        p_dist = hamming_distances(item["phash"], base_p)
        d_dist = hamming_distances(item["dhash"], base_d)
        if item["sha256"] in base_sha:
            k = base_sha[item["sha256"]]
        else:
            hits = np.nonzero(is_near_match(p_dist, d_dist, rules))[0]
            if len(hits) == 0:
                kept.append(item)
                continue
            k = int(min(hits, key=lambda h: (int(p_dist[h]), int(d_dist[h]), int(h))))
        base = base_items[k]
        if base["label"] == item["label"]:
            removed_rows.append({"file": rel(item["path"]), "kept_copy": rel(base["path"]), "reason": reason,
                                 "distance": int(p_dist[k]), "dhash_distance": int(d_dist[k]),
                                 "pair": item["path"], "pair_with": base["path"]})
            n_copies += 1
        else:
            group_id = next((r["group"] for r in review_rows if r["file"] == rel(base["path"])), None) \
                or next_group_id(review_rows)
            review_rows.append({"group": group_id, "file": rel(item["path"]), "label": item["label"],
                                "matches": f"{rel(base['path'])} ({'base' if reason == 'copy_of_base' else 'locked'} image, kept)",
                                "distance": int(p_dist[k]), "dhash_distance": int(d_dist[k])})
            held.append(item)
    return kept, n_copies, held


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
    Returns (kept items, near duplicates removed, conflict groups, list of images held for review).
    """
    for item in items:
        if "phash" not in item:  # already known if this image was checked in an earlier stage
            item["phash"], item["dhash"] = image_hashes(item["path"], item.get("upright", False))
    phashes = np.array([item["phash"] for item in items], dtype=np.uint64)
    dhashes = np.array([item["dhash"] for item in items], dtype=np.uint64)

    out = np.zeros(len(items), dtype=bool)  # True once an image is dropped or held for review
    n_near, n_conflict_groups = 0, 0
    held = []  # images held for review, in the order they were held

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
            held += [anchor] + [items[j] for j in group]

    # Follow-up pass. Compare each kept image with every image that is out (dropped
    # or held) plus the exact-copy conflicts. Repeat until nothing new is held,
    # because a newly held image can itself match another kept image.
    extra = list(already_held)
    for item in extra:
        if "phash" not in item:
            item["phash"], item["dhash"] = image_hashes(item["path"], item.get("upright", False))
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
            held.append(item)
            newly_held.append(item)
        others = newly_held  # only the new ones can create further matches

    kept = [item for i, item in enumerate(items) if not out[i]]
    return kept, n_near, n_conflict_groups, held


# ---------------------------------------------------------------------------
# Step 5b: known duplicates (runs right after the split, see main())
# ---------------------------------------------------------------------------

def drop_known_duplicates(splits_by_class, config, removed_rows):
    """Drop the copies listed under `known_duplicates:` in label_map.yaml.

    Each entry names a `file` and its `train_match`: two images a person
    confirmed by eye are the same photo, too different for the automatic
    pHash/dHash cutoffs to catch. A copy only leaks if the two images end up in
    DIFFERENT splits, so we first look up where each one landed:
      - same split            -> no leak, keep both (a note is printed)
      - one of them in train  -> drop the one that is NOT in train, keep the train copy
      - validation vs test    -> drop `file`, keep `train_match`
    Every drop is logged in results/removed_duplicates.csv as "known_duplicate".

    Why AFTER the split: removing images before the shuffle changes every
    class's list, so seed 42 would move thousands of OTHER images to a
    different split, and new copies could end up across train and test.
    Dropping afterwards changes nothing except the listed files.
    Returns (splits_by_class without the dropped files, number dropped).
    """
    entries = config.get("known_duplicates") or []
    # Where every image landed: path -> (split name, item).
    where = {rel(item["path"]): (split_name, item)
             for splits in splits_by_class.values()
             for split_name, items in splits.items() for item in items}
    to_drop = set()

    for entry in entries:
        file, train_match = entry["file"], entry["train_match"]
        # A path that does not exist at all is a typo in the yaml: stop.
        for path in (file, train_match):
            if not (REPO_ROOT / path).is_file():
                raise SystemExit(f"known_duplicates: {path} does not exist. Check configs/label_map.yaml.")
        # Not in play: its class is not active (e.g. hazardous with 3 classes)
        # or it was already removed as a duplicate. Nothing can leak.
        if file not in where or train_match not in where:
            continue
        file_split, train_split = where[file][0], where[train_match][0]
        if file_split == train_split:
            print(f"  [keep both] {file} and its match are both in {file_split}: no leak.")
            continue
        # Keep the copy in train if there is one; otherwise keep train_match.
        drop, keep = (train_match, file) if file_split == "train" else (file, train_match)
        to_drop.add(drop)
        a, b = where[drop][1], where[keep][1]  # both already have pHash/dHash from the near-duplicate step
        removed_rows.append({"file": drop, "kept_copy": keep, "reason": "known_duplicate",
                             "distance": int(np.bitwise_count(np.uint64(a["phash"] ^ b["phash"]))),
                             "dhash_distance": int(np.bitwise_count(np.uint64(a["dhash"] ^ b["dhash"])))})

    # Rebuild every split list without the dropped files; everything else stays where it was.
    kept = {cls: {name: [item for item in items if rel(item["path"]) not in to_drop]
                  for name, items in splits.items()}
            for cls, splits in splits_by_class.items()}
    return kept, len(to_drop)


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


def split_by_source(items, ratios, seed):
    """Split one class 70/15/15, each source dataset on its own, then combine.

    Why per source: one shuffle over the whole class means adding even one new
    photo moves EVERY existing image to a possibly different split. Kaggle
    near-copies that sat together in train could then end up in train and test
    (leakage). Shuffling each dataset separately keeps every existing image
    exactly where it was when new photos arrive. With only one source, this is
    the same as one shuffle over the class.
    """
    by_source = defaultdict(list)  # dataset name -> its items, in the order they were collected
    for item in items:
        by_source[item["path"].relative_to(RAW_DIR).parts[0]].append(item)
    combined = {name: [] for name in SPLIT_NAMES}
    for source_items in by_source.values():  # label_map.yaml order
        for name, part in split_list(source_items, ratios, seed).items():
            combined[name] += part
    return combined


def source_of(item):
    """Dataset name of an image: the first folder under data/raw/."""
    return item["path"].relative_to(RAW_DIR).parts[0]


def load_split_lock(config):
    """{source path: (label, split)} from frozen/wild_split_lock.csv, or {} if there is none.

    The lock (written by src/data/freeze_wild_split.py) pins every non-Kaggle image of
    the frozen build to its split: train, validation or wild_test. Locked images keep
    that split on every build; any other non-Kaggle image goes to train or validation.
    """
    path = (config.get("frozen") or {}).get("wild_split_lock")
    if not path:
        return {}
    if not (REPO_ROOT / path).exists():
        raise SystemExit(f"{path} (frozen.wild_split_lock in configs/label_map.yaml) is missing. Without it every "
                         f"wild image would be re-split and the frozen wild test would change. Restore it with "
                         f"git checkout {path}.")
    with open(REPO_ROOT / path, newline="") as f:
        return {row["source"]: (row["label"], row["split"]) for row in csv.DictReader(f)}


def mark_locked(items, lock):
    """Set item["locked_split"] for every locked image; stop if its label changed since the freeze.

    Also warns about locked images that are missing (dataset not downloaded, file deleted,
    or category set to EXCLUDE). A missing wild_test image makes the frozen check fail.
    """
    found = set()
    for item in items:
        entry = lock.get(rel(item["path"]))
        if entry is None:
            continue
        label, split = entry
        if label != item["label"]:
            raise SystemExit(f"{rel(item['path'])} is locked as {label} ({split}) but configs/label_map.yaml now "
                             f"maps it to {item['label']}. A locked image cannot change class: that would change "
                             f"the frozen test sets. Undo the mapping change, or re-freeze on a team decision.")
        item["locked_split"] = split
        found.add(rel(item["path"]))
    missing = Counter((src.split("/")[2], lock[src][1]) for src in lock if src not in found)
    if missing:
        print("  WARNING: locked images not found on this machine (the frozen wild test cannot match without them): "
              + ", ".join(f"{n} {source} {split}" for (source, split), n in sorted(missing.items()))
              + ". Run python src/data/download.py and python src/data/crop_taco.py.")


def category_of(item):
    """Source category for the recycling cap and the reports: dataset/class folder[/first sub-folder].

    realwaste/Metal, taco_crops/recycling/drink-can, team_photos/trash, ...
    """
    sub = item["path"].relative_to(item["folder"]).parts[:-1]
    return "/".join([source_of(item), item["folder"].name, *sub[:1]])


def apply_recycling_cap(items, wild_cfg, removed_rows):
    """Keep at most `recycling_cap` new recycling images from `cap_datasets`, spread over their categories.

    Recycling is already about 68% of train, so the wild sources must not push it
    further. The cap is shared evenly by the categories ("water filling": a small
    category keeps all its images and its unused share goes to the bigger ones),
    and inside each category the images are drawn at random with the seed.
    Dropped images are logged as "recycling_cap". Returns (kept items, number dropped).
    """
    cap = wild_cfg.get("recycling_cap")
    capped_sources = set(wild_cfg.get("cap_datasets") or [])
    pool = [it for it in items if it["label"] == "recycling" and source_of(it) in capped_sources]
    if cap is None or len(pool) <= cap:
        if pool:
            print(f"Recycling cap {cap}: {len(pool)} new recycling images from {sorted(capped_sources)}, all kept.")
        return items, 0
    # Locked images (frozen split) are always kept and use up the cap first; only
    # what is left is shared by the unlocked images.
    locked = [it for it in pool if it.get("locked_split")]
    pool = [it for it in pool if not it.get("locked_split")]
    by_cat = defaultdict(list)
    for it in pool:  # file order inside each category
        by_cat[category_of(it)].append(it)
    quota, remaining = {}, max(0, cap - len(locked))
    ordered = sorted(by_cat, key=lambda c: (len(by_cat[c]), c))  # smallest first
    for k, cat in enumerate(ordered):
        quota[cat] = min(len(by_cat[cat]), remaining // (len(ordered) - k))
        remaining -= quota[cat]
    rng = random.Random(wild_cfg["seed"])
    keep = {id(it) for it in locked}
    for cat in sorted(by_cat):
        keep.update(id(it) for it in rng.sample(by_cat[cat], quota[cat]))
    dropped = 0
    for it in pool:
        if id(it) not in keep:
            removed_rows.append({"file": rel(it["path"]), "kept_copy": "", "reason": "recycling_cap",
                                 "distance": "", "dhash_distance": ""})
            dropped += 1
    print(f"Recycling cap: {len(pool) + len(locked)} new recycling images from {', '.join(sorted(capped_sources))}, "
          f"kept {len(keep)} (cap {cap}, seed {wild_cfg['seed']}; {len(locked)} locked images kept first, "
          f"{len(keep) - len(locked)} of {len(pool)} unlocked):")
    for cat in sorted(by_cat):
        print(f"  {cat:<45} {len(by_cat[cat]):>5} unlocked -> {quota[cat]:>4}")
    return [it for it in items if not (it["label"] == "recycling" and source_of(it) in capped_sources)
            or id(it) in keep], dropped


def split_wild_images(new_items, splits_by_class, config, removed_rows):
    """Place images from every non-base source in train / validation / wild_test.

    The base split, including the benchmark test set, is fixed before this runs and
    is never changed here. Two images "look alike" if pHash <= keep_together_max_phash
    AND dHash <= keep_together_max_dhash (label_map.yaml; the same rule as check_leakage.py).
      0. LOCKED images (frozen/wild_split_lock.csv, item["locked_split"]) keep their split.
         wild_test holds locked images only, so it never changes.
      1. Groups that must stay in ONE split: look-alike images (chains included,
         A~B and B~C -> one group), images from the same source photo (the dataset's
         group_pattern, e.g. all TACO crops of one photo), and runs of consecutively
         numbered files (the dataset's group_block, e.g. RealWaste files 0-19, 20-39, ...
         of one category, because it photographed each item several times in a row).
      2. Unlocked images in a group with locked images follow them: dropped if one is in
         wild_test ("lookalike_of_wild_test"), else to train if one is in train, else validation.
         An unlocked image that looks like a benchmark test image is dropped
         ("lookalike_of_benchmark"). Otherwise an unlocked group that looks like a base train
         image goes to train, one that only looks like base validation images to validation.
      3. Every other unlocked group is shuffled with the seed and handed out per class (by
         its most common label) to train or validation, toward the 70 / 15 ratio.
      4. Final check: every wild_test image is compared with ALL train images (base and
         new) and the benchmark test images; a look-alike is dropped ("wild_test_lookalike").
         With the lock in place this never fires; if it did, the frozen check would stop the build.
    Returns (splits_by_class with the new images added, {reason: number dropped}).
    """
    dropped = Counter()
    if not new_items:
        return splits_by_class, dropped
    wild = config["wild_split"]
    ratios, seed = wild["ratios"], wild["seed"]
    rules = config["near_duplicates"]
    p_max, d_max = rules["keep_together_max_phash"], rules["keep_together_max_dhash"]
    together = lambda p, d: (p <= p_max) & (d <= d_max)
    names = ["train", "validation", WILD_TEST]
    new_names = ["train", "validation"]  # where unlocked images may go

    base = [(split, item) for splits in splits_by_class.values()
            for split in SPLIT_NAMES for item in splits[split]]
    base_p = np.array([item["phash"] for _, item in base], dtype=np.uint64)
    base_d = np.array([item["dhash"] for _, item in base], dtype=np.uint64)
    base_split = np.array([split for split, _ in base])

    # 1. Groups ("union-find": merging two groups = pointing one at the other).
    new_p = np.array([item["phash"] for item in new_items], dtype=np.uint64)
    new_d = np.array([item["dhash"] for item in new_items], dtype=np.uint64)
    group_of = list(range(len(new_items)))

    def find(i):
        while group_of[i] != i:
            group_of[i] = group_of[group_of[i]]
            i = group_of[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            group_of[max(a, b)] = min(a, b)

    for i in range(len(new_items)):
        later = slice(i + 1, None)
        close = together(hamming_distances(new_p[i], new_p[later]), hamming_distances(new_d[i], new_d[later]))
        for k in np.nonzero(close)[0]:
            union(i, i + 1 + int(k))
    first_of_photo = {}
    for i, item in enumerate(new_items):
        dataset_cfg = config["datasets"][source_of(item)]
        pattern = dataset_cfg.get("group_pattern")
        match = re.match(pattern, item["path"].name) if pattern else None
        if match:  # e.g. every TACO crop of one photo
            union(i, first_of_photo.setdefault((source_of(item), match.group(1)), i))
        block = dataset_cfg.get("group_block")
        number = re.search(r"(\d+)\D*$", item["path"].stem) if block else None
        if number:  # e.g. RealWaste shot each item several times in a row: files 370-372 are one leaf
            key = (source_of(item), item["folder"].name, int(number.group(1)) // block)
            union(i, first_of_photo.setdefault(key, i))
    groups = defaultdict(list)  # group id -> member indexes, in file order
    for i in range(len(new_items)):
        groups[find(i)].append(i)

    def drop(i, reason, counter_key, other=None, p=None, d=None):
        removed_rows.append({"file": rel(new_items[i]["path"]), "kept_copy": rel(other["path"]) if other else "",
                             "reason": reason, "distance": "" if p is None else p,
                             "dhash_distance": "" if d is None else d})
        dropped[counter_key] += 1

    # 0 + 2. Locked images keep their split; unlocked ones follow locked group members or base look-alikes.
    where = {}  # image index -> split, or None if dropped
    for gid, members in groups.items():
        locked_here = {i: new_items[i]["locked_split"] for i in members if new_items[i].get("locked_split")}
        where.update(locked_here)
        unlocked = [i for i in members if i not in locked_here]
        if not unlocked:
            continue
        lsplits = set(locked_here.values())
        hits, closest = [], None
        for i in unlocked:
            p_dist, d_dist = hamming_distances(new_p[i], base_p), hamming_distances(new_d[i], base_d)
            idx = np.nonzero(together(p_dist, d_dist))[0]
            hits += list(base_split[idx])
            for j in idx:
                if base_split[j] == "test" and closest is None:
                    closest = (i, int(j), int(p_dist[j]), int(d_dist[j]))
        if WILD_TEST in lsplits:
            anchor = next(new_items[i] for i in locked_here if locked_here[i] == WILD_TEST)
            for i in unlocked:
                drop(i, "lookalike_of_wild_test", "look-alikes of wild test", anchor)
            for i in unlocked:
                where[i] = None
        elif "test" in hits:
            b_item = base[closest[1]][1]
            for i in unlocked:
                drop(i, "lookalike_of_benchmark", "look-alikes of benchmark", b_item,
                     closest[2] if i == closest[0] else None, closest[3] if i == closest[0] else None)
                where[i] = None
        elif lsplits:  # follow the locked group members, so one group stays in one split
            where.update({i: ("train" if "train" in lsplits else "validation") for i in unlocked})
        elif "train" in hits:
            where.update({i: "train" for i in unlocked})
        elif hits:
            where.update({i: "validation" for i in unlocked})

    # 3. Unlocked groups not placed yet: seeded shuffle, then fill each class toward
    # the train / validation ratio. Only unlocked images are counted, so new sources
    # split about 82 / 18 on their own.
    free = [gid for gid, members in groups.items() if all(i not in where for i in members)]
    random.Random(seed).shuffle(free)
    label_of = lambda gid: Counter(new_items[i]["label"] for i in groups[gid]).most_common(1)[0][0]
    unlocked_ids = [i for i in range(len(new_items)) if not new_items[i].get("locked_split")]
    class_total = Counter(new_items[i]["label"] for i in unlocked_ids if where.get(i, "") is not None)
    count = defaultdict(Counter)  # class -> split -> unlocked images so far
    for i in unlocked_ids:
        if where.get(i):
            count[new_items[i]["label"]][where[i]] += 1
    share = {name: ratios[name] / sum(ratios[n] for n in new_names) for name in new_names}
    for gid in free:
        cls = label_of(gid)
        room = {name: share[name] * class_total[cls] - count[cls][name] for name in new_names}
        split = max(new_names, key=lambda name: room[name])  # ties -> train
        for i in groups[gid]:
            where[i] = split
            count[new_items[i]["label"]][split] += 1

    result = {cls: {name: list(splits.get(name, [])) for name in ALL_SPLITS} for cls, splits in splits_by_class.items()}
    for i, item in enumerate(new_items):
        if where[i] is not None:
            result.setdefault(item["label"], {name: [] for name in ALL_SPLITS})[where[i]].append(item)

    # 4. Final check: no wild_test image may look like a train or benchmark image.
    reference = [item for splits in result.values() for item in splits["train"] + splits["test"]]
    ref_p = np.array([item["phash"] for item in reference], dtype=np.uint64)
    ref_d = np.array([item["dhash"] for item in reference], dtype=np.uint64)
    for cls, splits in result.items():
        keep = []
        for item in splits[WILD_TEST]:
            p_dist, d_dist = hamming_distances(item["phash"], ref_p), hamming_distances(item["dhash"], ref_d)
            hits = np.nonzero(together(p_dist, d_dist))[0]
            if len(hits) == 0:
                keep.append(item)
                continue
            k = int(min(hits, key=lambda h: (int(p_dist[h]) + int(d_dist[h]), int(h))))
            removed_rows.append({"file": rel(item["path"]), "kept_copy": rel(reference[k]["path"]),
                                 "reason": "wild_test_lookalike", "distance": int(p_dist[k]),
                                 "dhash_distance": int(d_dist[k])})
            dropped["wild_test look-alikes"] += 1
        splits[WILD_TEST] = keep

    # Report: new images per source, class and split, locked and unlocked.
    n_groups = sum(len(m) > 1 for m in groups.values())
    n_locked = sum(1 for it in new_items if it.get("locked_split"))
    print(f"\nWild split ({len(new_items)} new images: {n_locked} locked, {len(new_items) - n_locked} unlocked; "
          f"{len(groups)} groups, {n_groups} with more than one image; seed {seed}): "
          f"{dropped['look-alikes of benchmark']} dropped as look-alikes of benchmark test images, "
          f"{dropped['look-alikes of wild test']} as look-alikes of wild test images, "
          f"{dropped['wild_test look-alikes']} wild_test images dropped as look-alikes of train/benchmark images.")
    table = Counter()
    for cls, splits in result.items():
        for name in names:
            for item in splits[name]:
                if source_of(item) not in (config.get("base_datasets") or []):
                    kind = "locked" if item.get("locked_split") else "new"
                    table[(source_of(item), cls, kind, name)] += 1
    print(f"  {'source':<14}{'class':<11}{'':<8}{'train':>7}{'validation':>12}{'wild_test':>11}{'total':>7}")
    for source, cls, kind in sorted({(s, c, k) for s, c, k, _ in table}):
        row = [table[(source, cls, kind, name)] for name in names]
        print(f"  {source:<14}{cls:<11}{kind:<8}{row[0]:>7}{row[1]:>12}{row[2]:>11}{sum(row):>7}")
    return result, dropped


def copy_name(item):
    """Base file name for the copy in data/<split>/<class>/.

    dataset + the class folder + any sub-folders + the file name, joined with "__":
      data/raw/garbage12_enhanced/.../battery/battery7.jpg -> garbage12_enhanced__battery__battery7.jpg
      data/raw/amy_images/Recycle/2024/IMG_0001.HEIC     -> amy_images__Recycle__2024__IMG_0001.HEIC.jpg
    HEIC/HEIF/AVIF photos get ".jpg" added because they are saved as JPEG.
    In rare cases two different files still get the same name (for example
    "a__b.jpg" next to a sub-folder "a" holding "b.jpg"); unique_copy_names()
    then adds "__2", "__3", ... so no file is ever overwritten.
    """
    src, folder = item["path"], item["folder"]
    dataset = src.relative_to(RAW_DIR).parts[0]
    parts = [dataset, folder.name, *src.relative_to(folder).parts]
    name = "__".join(parts)
    if src.suffix.lower() in CONVERT_TO_JPEG:
        name += ".jpg"
    return name


def unique_copy_names(splits_by_class):
    """Give every image a copy name that is unique in its split/class folder.

    Names are compared ignoring upper/lower case, because macOS treats "A.jpg" and
    "a.jpg" as the same file. A clash gets "__2", "__3", ... before the extension.
    Returns {id(item): name}.
    """
    names, used, renamed = {}, set(), 0
    for class_name in sorted(splits_by_class):
        for split_name in ALL_SPLITS:
            for item in splits_by_class[class_name].get(split_name, []):
                name = copy_name(item)
                stem, ext = name.rsplit(".", 1)
                n = 1
                while (split_name, class_name, name.lower()) in used:
                    n += 1
                    name = f"{stem}__{n}.{ext}"
                if n > 1:
                    renamed += 1
                used.add((split_name, class_name, name.lower()))
                names[id(item)] = name
    if renamed:
        print(f"  {renamed} copies got a __2/__3 suffix because their names would have clashed.")
    return names


def copy_one(src, dst, upright=False):
    """Copy one image into a split folder.

    Base (Kaggle) images are copied byte for byte. New photo sources
    (upright=True) are turned the way their EXIF orientation tag says and
    saved again WITHOUT that tag, so every viewer and training loader shows
    them upright with no extra step. JPG stays JPG, PNG stays PNG, and
    HEIC/HEIF/AVIF become JPG.
    """
    if not upright and src.suffix.lower() not in CONVERT_TO_JPEG:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)  # pixels turned upright; the new image carries no EXIF
        if src.suffix.lower() in CONVERT_TO_JPEG or dst.suffix.lower() in {".jpg", ".jpeg"}:
            img.convert("RGB").save(dst, "JPEG", quality=95)
        else:
            img.save(dst)  # PNG, WEBP, BMP, TIFF keep their own format


def copy_splits_and_write_manifests(splits_by_class):
    """Copy images into data/<split>/<class>/ and write data/splits/<split>.csv.

    Every copy gets a unique name from unique_copy_names(), so no file is ever
    overwritten. The CSV paths point at the copied files and are relative to the
    repo root.
    """
    names = unique_copy_names(splits_by_class)

    for d in list(SPLIT_DIRS.values()) + [MANIFEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for split_name in ALL_SPLITS:
        rows = []
        for class_name in sorted(splits_by_class):
            items = splits_by_class[class_name].get(split_name, [])
            if not items and split_name == WILD_TEST:
                continue  # no empty wild_test/<class>/ folders when there are no new images
            out_dir = SPLIT_DIRS[split_name] / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                dst = out_dir / names[id(item)]
                copy_one(item["path"], dst, item.get("upright", False))
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

    This reads the files actually on disk in data/train, data/validation,
    data/test and data/wild_test, so it checks the real output, not just our
    in-memory lists.
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


def check_no_camera_test(manifest_dir=MANIFEST_DIR):
    """Stop if any manifest path mentions the held-out camera test folder.

    data/raw/camera_test/ holds the demo-room photos that score_camera_test.py
    uses as a final, never-trained-on check. A copy of one of them in train,
    validation or test would make that check meaningless, so the build stops here
    with a FAILED message (the manifests are already on disk at this point; fix
    the config and build again). The copy names keep the source folder
    (camera_test__<class>__<file>), so a plain text match is enough.
    Shared with the tests, which plant a camera_test path in a scratch manifest.
    """
    bad = []
    for split_name in ALL_SPLITS:
        if not (manifest_dir / f"{split_name}.csv").exists():
            continue
        with open(manifest_dir / f"{split_name}.csv", newline="") as f:
            bad += [f"{split_name}: {row['path']}" for row in csv.DictReader(f)
                    if CAMERA_TEST_NAME in row["path"].lower()]
    if bad:
        shown = "\n  ".join(bad[:5]) + (f"\n  (+{len(bad) - 5} more)" if len(bad) > 5 else "")
        raise SystemExit(f"FAILED: {len(bad)} manifest path(s) come from data/raw/{CAMERA_TEST_NAME}/, "
                         f"the held-out demo-room photos that must never be split or trained on:\n  {shown}\n"
                         f"Remove {CAMERA_TEST_NAME} from configs/label_map.yaml (or move the photos out of any "
                         f"other source) and run this script again.")
    print(f"Camera test check passed: no data/raw/{CAMERA_TEST_NAME}/ photo is in any split.")


def check_frozen_tests(config, active):
    """Stop unless data/splits/test.csv and wild_test.csv are byte-identical to the frozen copies.

    The copies live in configs/frozen/ (written by src/data/freeze_wild_split.py) and
    their sha256 is in label_map.yaml under `frozen:`, so an edit to a copy is caught too.
    Keeping both test sets identical is what makes every model's scores comparable.
    A build with another class set (--num-classes 3) cannot match them, so it only warns.
    """
    frozen = config.get("frozen") or {}
    if not frozen:
        print("No frozen test sets in configs/label_map.yaml (frozen:), not checked.")
        return
    other_classes = len(active["classes"]) != config["num_classes"] or active["exclude"]
    for name, key in [("test.csv", "test_csv"), (f"{WILD_TEST}.csv", "wild_test_csv")]:
        copy = REPO_ROOT / frozen[key]
        expected = frozen[f"{key}_sha256"]
        if hashlib.sha256(copy.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"FAILED: the frozen copy {frozen[key]} does not match its sha256 in "
                             f"configs/label_map.yaml. The frozen copies must never be edited by hand.")
        now = (MANIFEST_DIR / name).read_bytes()
        if now == copy.read_bytes():
            print(f"Frozen check passed: data/splits/{name} is byte-identical to {frozen[key]} (sha256 {expected[:12]}).")
            continue
        if other_classes:
            print(f"WARNING: data/splits/{name} differs from the frozen 4-class copy because this build uses "
                  f"classes {active['classes']}. Scores from this build are not comparable.")
            continue
        old_rows = set(copy.read_text().splitlines()[1:])
        new_rows = set(now.decode().splitlines()[1:])
        gone, added = sorted(old_rows - new_rows), sorted(new_rows - old_rows)
        raise SystemExit(f"FAILED: data/splits/{name} differs from the frozen copy {frozen[key]}: "
                         f"{len(gone)} rows missing (for example {gone[:2]}), {len(added)} rows added "
                         f"(for example {added[:2]}).\nThe frozen test sets must not change. Missing rows usually "
                         f"mean a dataset is not downloaded or cropped on this machine (run download.py and "
                         f"crop_taco.py); otherwise undo the change to the data or the config.")


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
    print(f"\n{'class':<12}{'train':>8}{'validation':>12}{'test':>8}{'wild_test':>11}{'total':>8}")
    print("-" * 59)
    totals = Counter()
    for class_name in sorted(splits_by_class):
        counts = Counter({k: len(v) for k, v in splits_by_class[class_name].items()})
        total = sum(counts.values())
        totals.update(counts)
        totals["total"] += total
        print(f"{class_name:<12}{counts['train']:>8}{counts['validation']:>12}{counts['test']:>8}"
              f"{counts[WILD_TEST]:>11}{total:>8}")
    print("-" * 59)
    print(f"{'TOTAL':<12}{totals['train']:>8}{totals['validation']:>12}{totals['test']:>8}"
          f"{totals[WILD_TEST]:>11}{totals['total']:>8}")
    print("(test = the frozen benchmark; wild_test = new sources only)")
    return totals["total"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_duplicates(config, active):
    """Steps 2-4 in two stages. Returns a dict with:
      base_kept   base images to split (exactly as if no new source existed)
      new_kept    new-source images for split_wild_images(); copies of base
                  images are already gone, so a new image can never remove or
                  move a base image
      removed_rows, review_rows, n_found, out_counts
    Shared with src/data/review_near_dups.py so the spot checks use the same rules.
    """
    rules = config["near_duplicates"]
    removed_rows = []  # one dict per dropped file  -> results/removed_duplicates.csv
    review_rows = []   # one dict per held-back file -> results/conflicts_to_review.csv
    items, n_found, skipped = collect_images(config, active, removed_rows)
    if not items:
        raise SystemExit("No images found. Run src/data/download.py first.")

    base_names = set(config.get("base_datasets") or [])
    is_base = lambda item: item["path"].relative_to(RAW_DIR).parts[0] in base_names
    base_items = [item for item in items if is_base(item)]
    new_items = [item for item in items if not is_base(item)]
    # Locked images (frozen split) behave like the base: they were already checked for
    # duplicates when they were frozen, and a new image can never remove or move one.
    mark_locked(new_items, load_split_lock(config))
    locked_items = [item for item in new_items if item.get("locked_split")]
    unlocked_items = [item for item in new_items if not item.get("locked_split")]

    print(f"Checking duplicates: pHash <= {rules['max_phash']} and dHash <= {rules['max_dhash']}...")
    # Stage A: the base on its own (this is what keeps its split identical).
    base_kept, n_exact, held_a = remove_exact_duplicates(base_items, removed_rows, review_rows)
    base_kept, n_near, _, near_held_a = remove_near_duplicates(base_kept, removed_rows, review_rows, rules,
                                                               already_held=held_a)
    n_held = len(held_a) + len(near_held_a)

    # Stage B: unlocked new images. First against EVERY base image (kept or held in
    # stage A), then against every locked image: a copy is dropped, or held if its label
    # differs, and the base or locked image always stays. Then the unlocked images among
    # themselves with the normal rules. Locked images pass through unchanged.
    new_kept, n_copies_of_base = [], 0
    if unlocked_items:
        kept, n_copies_of_base, held_b0 = drop_copies_of_base(unlocked_items, base_kept + held_a + near_held_a,
                                                              removed_rows, review_rows, rules)
        kept, n_copies_of_locked, held_b1 = drop_copies_of_base(kept, locked_items, removed_rows, review_rows,
                                                                rules, reason="copy_of_locked")
        kept, n_exact_b, held_b = remove_exact_duplicates(kept, removed_rows, review_rows)
        kept, n_near_b, _, near_held_b = remove_near_duplicates(kept, removed_rows, review_rows, rules,
                                                                already_held=held_b)
        n_exact += n_exact_b
        n_near += n_near_b
        n_copies_of_base += n_copies_of_locked
        n_held += len(held_b0) + len(held_b1) + len(held_b) + len(near_held_b)
        still = {id(item) for item in kept}
    else:
        still = set()
    # Locked and surviving unlocked images, in file order. The wild split needs every
    # image's fingerprints, and locked images are not hashed above when nothing is unlocked.
    new_kept = [item for item in new_items if item.get("locked_split") or id(item) in still]
    for item in new_kept:
        if "phash" not in item:
            item["phash"], item["dhash"] = image_hashes(item["path"], item.get("upright", False))

    n_groups = len({row["group"] for row in review_rows})
    out_counts = {
        "corrupted": skipped["corrupted"],
        "unsupported format": skipped["unsupported format"],
        "exact duplicates": n_exact,
        "near duplicates": n_near,
        "copies of base or locked images": n_copies_of_base,
        "known duplicates": 0,  # filled in by main() after the split
        "recycling cap": 0,                # filled in by main()
        "look-alikes of benchmark": 0,     # filled in by main()
        "look-alikes of wild test": 0,     # filled in by main()
        "wild_test look-alikes": 0,        # filled in by main()
        "held for review": n_held,
    }
    print(f"\nRemoved {n_exact} exact duplicates and {n_near} near duplicates, and {n_copies_of_base} new images "
          f"that copy a base or locked image. Held {n_held} images in {n_groups} label-conflict groups for review "
          f"({rel(CONFLICTS_CSV)}). Dropped {skipped['corrupted']} corrupted and {skipped['unsupported format']} "
          f"unsupported files. {len(base_kept) + len(new_kept)} images go on to the split "
          f"({len(base_kept)} base, {len(new_kept)} new).")
    return {"base_kept": base_kept, "new_kept": new_kept,
            "removed_rows": removed_rows, "review_rows": review_rows,
            "n_found": n_found, "out_counts": out_counts}


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

    found = find_duplicates(config, active)
    removed_rows, review_rows = found["removed_rows"], found["review_rows"]
    n_found, out_counts = found["n_found"], found["out_counts"]

    # Base split: exactly as if no new source existed.
    by_class = defaultdict(list)
    for item in found["base_kept"]:
        by_class[item["label"]].append(item)
    splits_by_class = {cls: {**split_by_source(by_class[cls], ratios, seed), WILD_TEST: []} for cls in sorted(by_class)}
    splits_by_class, n_known = drop_known_duplicates(splits_by_class, config, removed_rows)
    out_counts["known duplicates"] = n_known
    print(f"Dropped {n_known} known duplicates listed in {rel(CONFIG_PATH)} "
          f"(the copy outside train is dropped; see {rel(REMOVED_CSV)}).")
    # The base split, benchmark test included, is final from here on.
    # New sources: recycling cap (locked images first), then locked images keep their
    # split and unlocked ones go to train or validation.
    new_items, n_capped = apply_recycling_cap(found["new_kept"], config["wild_split"], removed_rows)
    out_counts["recycling cap"] = n_capped
    splits_by_class, wild_dropped = split_wild_images(new_items, splits_by_class, config, removed_rows)
    out_counts["look-alikes of benchmark"] = wild_dropped["look-alikes of benchmark"]
    out_counts["look-alikes of wild test"] = wild_dropped["look-alikes of wild test"]
    out_counts["wild_test look-alikes"] = wild_dropped["wild_test look-alikes"]
    splits_by_class = dict(sorted(splits_by_class.items()))
    # Every active class must still have images, or the class order and the
    # weights would not line up with num_classes.
    missing = set(active["classes"]) - {cls for cls, splits in splits_by_class.items() if any(splits.values())}
    if missing:
        raise SystemExit(f"No images left for class(es) {sorted(missing)}. Check data/raw/ and label_map.yaml.")
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
    check_frozen_tests(config, active)
    check_no_camera_test()
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
