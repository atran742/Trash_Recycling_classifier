"""
download.py - download the public Kaggle datasets into data/raw/<key>/.

Run from the repo root:
    python src/data/download.py                            # default: garbage12_enhanced, plus garbage_v2 if needed
    python src/data/download.py --only garbage12_enhanced  # just one dataset
    python src/data/download.py --list                     # show the datasets and exit
    python src/data/download.py --force                    # re-download even if the folder exists

After each download it prints every image folder it found and how many images
are inside. Those folder names are what configs/label_map.yaml must use.

Needs a Kaggle API token (see check_kaggle_token below).
Images land in data/, which is git-ignored. Never commit them.
"""

import argparse
import os
import sys
from pathlib import Path

# REPO_ROOT is the folder two levels above this file (src/data/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Short local name (the "key") -> Kaggle dataset slug, the part after kaggle.com/datasets/.
# Licenses, counts and overlap notes live in docs/dataset_inventory.md.
DATASETS = {
    # Primary set: 12 classes, deduplicated, labels corrected, includes battery.
    "garbage12_enhanced": "huberthamelin/garbage-classification-labels-corrections",
    # Backup battery source. Only downloaded when garbage12_enhanced is short on batteries.
    # If it is ever downloaded, add a garbage_v2 entry to configs/label_map.yaml,
    # otherwise build_splits.py will not use it. (Its glass is one "glass" folder.)
    "garbage_v2": "sumn2u/garbage-classification-v2",
}
# Not downloaded on purpose: mostafaabla/garbage-classification. It is the original
# version of garbage12_enhanced, so downloading both would just add duplicates.

# If garbage12_enhanced has fewer battery images than this, also fetch garbage_v2.
MIN_BATTERY_IMAGES = 500


def check_requirements():
    """Stop with one clear instruction if a package from requirements.txt is missing.

    This runs first, so a computer set up before a new package was added (for
    example pillow-heif for iPhone photos) learns it right away, not halfway
    through build_splits.py.
    """
    import importlib.util

    needed = {"kaggle": "kaggle", "yaml": "pyyaml", "PIL": "pillow", "pillow_heif": "pillow-heif",
              "imagehash": "imagehash", "numpy": "numpy", "matplotlib": "matplotlib"}
    missing = [pip_name for module, pip_name in needed.items() if importlib.util.find_spec(module) is None]
    if missing:
        print(f"ERROR: missing Python packages: {', '.join(missing)}\n"
              f"Run:  pip install -r requirements.txt   and then run this script again.", file=sys.stderr)
        sys.exit(1)


def check_kaggle_token():
    """Stop early with clear instructions if no Kaggle credentials are set up.

    The kaggle library accepts any one of these:
      - ~/.kaggle/kaggle.json     (classic file: {"username": ..., "key": ...})
      - ~/.kaggle/access_token    (newer token string)
      - KAGGLE_API_TOKEN environment variable
      - KAGGLE_USERNAME + KAGGLE_KEY environment variables
    """
    kaggle_dir = Path.home() / ".kaggle"
    found = (
        (kaggle_dir / "kaggle.json").exists()
        or (kaggle_dir / "access_token").exists()
        or os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    if found:
        return

    print(
        "ERROR: No Kaggle token found.\n\n"
        "Fix it in 3 steps:\n"
        "  1. Sign in at https://www.kaggle.com, open Settings -> API, click 'Create New Token'.\n"
        "  2. Save it in a folder named .kaggle inside your home folder:\n"
        "       Mac/Linux: ~/.kaggle/kaggle.json (file download) or ~/.kaggle/access_token (token string),\n"
        "                  then run: chmod 600 ~/.kaggle/*\n"
        "       Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json or C:\\Users\\<you>\\.kaggle\\access_token\n"
        "  3. Re-run: python src/data/download.py",
        file=sys.stderr,
    )
    sys.exit(1)


def count_images(dataset_dir):
    """Return {folder path relative to dataset_dir: number of images directly inside}."""
    counts = {}
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            folder = path.parent.relative_to(dataset_dir)
            counts[folder] = counts.get(folder, 0) + 1
    return counts


def print_counts(dataset_dir):
    """Print each image folder and its count, then a total."""
    counts = count_images(dataset_dir)
    print(f"\nImage folders in data/raw/{dataset_dir.name}/:")
    for folder in sorted(counts, key=str):
        print(f"  {counts[folder]:>6}  {folder}")
    print(f"  {sum(counts.values()):>6}  TOTAL")
    return counts


def download(key, force=False):
    """Download one Kaggle dataset into data/raw/<key>/ and unzip it.

    Skips the download if that folder already has images (unless force=True).
    Returns the per-folder image counts.
    """
    out_dir = RAW_DIR / key
    if out_dir.exists() and count_images(out_dir) and not force:
        print(f"[skip] data/raw/{key}/ already has images. Use --force to re-download.")
        return print_counts(out_dir)

    # Only now do we need Kaggle, so only now do we need a token.
    check_kaggle_token()

    # Imported here, not at the top, so --list and the skip path work
    # without touching Kaggle at all.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = DATASETS[key]
    print(f"Downloading {slug} -> data/raw/{key}/ ...")
    # unzip=True extracts the archive and deletes the .zip afterwards.
    api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
    return print_counts(out_dir)


def battery_count(counts):
    """Largest image count among folders named 'battery' (the dataset ships two copies)."""
    return max((n for folder, n in counts.items() if folder.name == "battery"), default=0)


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets into data/raw/.")
    parser.add_argument("--only", choices=sorted(DATASETS), action="append",
                        help="download just this dataset (can be repeated)")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    check_requirements()
    if args.list:
        for key, slug in DATASETS.items():
            print(f"{key:<20} https://www.kaggle.com/datasets/{slug}")
        return

    # --only: download exactly what was asked for.
    if args.only:
        for key in args.only:
            download(key, args.force)
        return

    # Default: primary set first, then the backup only if batteries are short.
    counts = download("garbage12_enhanced", args.force)
    n_battery = battery_count(counts)
    if n_battery < MIN_BATTERY_IMAGES:
        print(f"\nOnly {n_battery} battery images (< {MIN_BATTERY_IMAGES}). Downloading garbage_v2 too.")
        download("garbage_v2", args.force)
    else:
        print(f"\n{n_battery} battery images (>= {MIN_BATTERY_IMAGES}). garbage_v2 not needed.")


if __name__ == "__main__":
    main()
