"""
download.py - download the public datasets into data/raw/<key>/.

Run from the repo root:
    python src/data/download.py                            # default: Kaggle set (+ garbage_v2 if needed), TACO, RealWaste
    python src/data/download.py --only garbage12_enhanced  # just one dataset
    python src/data/download.py --only taco --only realwaste
    python src/data/download.py --list                     # show the datasets and exit
    python src/data/download.py --force                    # re-download even if the folder exists

After each download it prints every image folder it found and how many images
are inside. Those folder names are what configs/label_map.yaml must use.

Two "wild" datasets add messy real-world photos (see docs/dataset_inventory.md):
  taco       TACO litter photos (official annotations file, pinned to one commit, plus
             the 1,500 photos it links to) -> data/raw/taco/. Run
             python src/data/crop_taco.py afterwards to cut one image per object.
  realwaste  RealWaste landfill photos from the UCI repository -> data/raw/realwaste/.

The Kaggle datasets need a Kaggle API token (see check_kaggle_token below); TACO
and RealWaste do not. Images land in data/, which is git-ignored. Never commit them.
"""

import argparse
import hashlib
import json
import os
import ssl
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ---- Open datasets (no Kaggle token needed) ----
# TACO: the official COCO annotations file in github.com/pedropro/TACO, pinned to the
# commit that last changed it, so every machine crops the same objects. The sha256
# check stops the run if the file ever differs from the one the mapping was made from.
TACO_COMMIT = "3844bcd3535cd12ec0218494d1d79f1372f553cc"  # 2026-10-20, last change to data/annotations.json
TACO_ANNOTATIONS_URL = f"https://raw.githubusercontent.com/pedropro/TACO/{TACO_COMMIT}/data/annotations.json"
TACO_ANNOTATIONS_SHA256 = "812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b"
# RealWaste: UCI Machine Learning Repository dataset 908 (CC BY 4.0), about 657 MB.
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
OPEN_DATASETS = {
    "taco": "https://github.com/pedropro/TACO (images listed in data/annotations.json)",
    "realwaste": "https://archive.ics.uci.edu/dataset/908/realwaste",
}


def check_requirements():
    """Stop with one clear instruction if a package from requirements.txt is missing.

    This runs first, so a computer set up before a new package was added (for
    example pillow-heif for iPhone photos) learns it right away, not halfway
    through build_splits.py.
    """
    import importlib

    needed = {"kaggle": "kaggle", "yaml": "pyyaml", "PIL": "pillow", "pillow_heif": "pillow-heif",
              "imagehash": "imagehash", "numpy": "numpy", "matplotlib": "matplotlib"}
    missing = []
    for module, pip_name in needed.items():
        try:
            importlib.import_module(module)
        except ImportError:  # not installed, or installed but broken
            missing.append(pip_name)
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


def https_context():
    """Certificates from certifi (in requirements.txt). The python.org installer on macOS
    ships without root certificates, which makes every HTTPS download fail otherwise."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url, out_path, tries=3):
    """Download one URL to out_path (via a .part file, so a cut-off download never looks finished)."""
    tmp = out_path.with_name(out_path.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Trash_Recycling_classifier/1.0"})
            with urllib.request.urlopen(request, timeout=120, context=https_context()) as response, \
                    open(tmp, "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
            tmp.replace(out_path)
            return
        except OSError as err:
            if attempt == tries:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"Could not download {url} after {tries} tries: {err}\n"
                                 f"Run the same command again; finished files are kept.")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_taco(force=False):
    """TACO: the pinned official annotations file, then every photo it lists.

    Photos keep the annotation file's own paths (data/raw/taco/batch_1/000006.jpg, ...),
    so crop_taco.py can find each one by its "file_name". Photos already on disk are
    skipped, so an interrupted run just continues where it stopped.
    """
    out_dir = RAW_DIR / "taco"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / "annotations.json"
    if force or not ann_path.exists():
        print(f"Downloading TACO annotations (commit {TACO_COMMIT[:7]}) ...")
        fetch(TACO_ANNOTATIONS_URL, ann_path)
    if sha256_of(ann_path) != TACO_ANNOTATIONS_SHA256:
        raise SystemExit(f"{ann_path.relative_to(REPO_ROOT)} is not the pinned TACO annotations file "
                         f"(sha256 differs). Delete it and run again.")
    with open(ann_path) as f:
        images = json.load(f)["images"]
    todo = [img for img in images if force or not (out_dir / img["file_name"]).exists()]
    print(f"TACO: {len(images)} photos listed, {len(images) - len(todo)} already here, downloading {len(todo)} ...")
    for img in todo:
        (out_dir / img["file_name"]).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    # 8 downloads at a time: about 1,500 photos of 1-5 MB each.
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, img["flickr_url"], out_dir / img["file_name"]) for img in todo]
        for job in as_completed(jobs):
            job.result()  # re-raises a failed download as a clear SystemExit
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} photos")
    # A slow server can cut a photo off early. Delete any photo that does not open
    # fully, so the next run downloads it again.
    from PIL import Image
    broken = []
    for img in images:
        path = out_dir / img["file_name"]
        try:
            with Image.open(path) as photo:
                photo.load()
        except OSError:
            broken.append(img["file_name"])
            path.unlink(missing_ok=True)
    if broken:
        print(f"WARNING: {len(broken)} photos were incomplete and were deleted: {broken[:5]}. "
              f"Run python src/data/download.py --only taco again to fetch them.")
    print("Next: python src/data/crop_taco.py   (cuts one image per mapped object into data/raw/taco_crops/)")
    return print_counts(out_dir)


def download_realwaste(force=False):
    """RealWaste: one zip from the UCI repository, unzipped into data/raw/realwaste/."""
    out_dir = RAW_DIR / "realwaste"
    if out_dir.exists() and count_images(out_dir) and not force:
        print("[skip] data/raw/realwaste/ already has images. Use --force to re-download.")
        return print_counts(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "realwaste.zip"
    print(f"Downloading RealWaste (about 657 MB) from {REALWASTE_URL} ...")
    fetch(REALWASTE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()
    # The UCI zip may contain a second zip with the images; unpack that too.
    for inner in list(out_dir.rglob("*.zip")):
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()
    return print_counts(out_dir)


def battery_count(counts):
    """Largest image count among folders named 'battery' (the dataset ships two copies)."""
    return max((n for folder, n in counts.items() if folder.name == "battery"), default=0)


def main():
    parser = argparse.ArgumentParser(description="Download the datasets into data/raw/.")
    parser.add_argument("--only", choices=sorted(DATASETS) + sorted(OPEN_DATASETS), action="append",
                        help="download just this dataset (can be repeated)")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    check_requirements()
    if args.list:
        for key, slug in DATASETS.items():
            print(f"{key:<20} https://www.kaggle.com/datasets/{slug}")
        for key, source in OPEN_DATASETS.items():
            print(f"{key:<20} {source}")
        return

    open_downloads = {"taco": download_taco, "realwaste": download_realwaste}
    # --only: download exactly what was asked for.
    if args.only:
        for key in args.only:
            if key in open_downloads:
                open_downloads[key](args.force)
            else:
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
    # The two wild datasets (no token needed).
    download_taco(args.force)
    download_realwaste(args.force)


if __name__ == "__main__":
    main()
