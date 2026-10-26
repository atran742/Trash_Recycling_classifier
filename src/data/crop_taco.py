"""
crop_taco.py - cut one image per annotated TACO object, for the classes we train on.

TACO photos show litter in context (a street, a beach, grass) with several
objects per photo, so they cannot be used whole. This script reads the official
annotations file (data/raw/taco/annotations.json, downloaded by download.py),
looks up each object's category in configs/label_map.yaml
(datasets -> taco_crops -> categories), and for every object whose category maps
to a class:
  - turns the photo upright by its EXIF tag (TACO's boxes are drawn on the upright
    photo: a sideways-stored phone photo only matches the annotation size after this),
  - adds 10% padding on every side of the box (10% of the box width left and right,
    10% of the box height top and bottom), clipped to the photo,
  - skips the crop if it is smaller than 64 pixels on either side,
  - saves it as data/raw/taco_crops/<class>/<category>/img<photo id>_ann<object id>.jpg
Objects whose category is EXCLUDE are not cropped. The file name starts with the
photo id so build_splits.py keeps all crops of one photo in the same split
(group_pattern in label_map.yaml).

The output folder is rebuilt from scratch on every run, so the crops always match
the current mapping. Images stay out of git like everything under data/.

Run from the repo root, after python src/data/download.py:
    python src/data/crop_taco.py
"""

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
TACO_DIR = REPO_ROOT / "data" / "raw" / "taco"
ANNOTATIONS = TACO_DIR / "annotations.json"
OUT_DIR = REPO_ROOT / "data" / "raw" / "taco_crops"
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"

PADDING = 0.10   # share of the box width / height added on each side
MIN_SIDE = 64    # crops smaller than this (in pixels) on either side are skipped


def slug(name):
    """Category name -> folder name: "Clear plastic bottle" -> "clear-plastic-bottle"."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def padded_box(bbox, width, height):
    """COCO box [x, y, w, h] plus PADDING on each side, clipped to the photo. Returns (left, top, right, bottom)."""
    x, y, w, h = bbox
    left = max(0, int(round(x - PADDING * w)))
    top = max(0, int(round(y - PADDING * h)))
    right = min(width, int(round(x + w + PADDING * w)))
    bottom = min(height, int(round(y + h + PADDING * h)))
    return left, top, right, bottom


def load_mapping(category_names):
    """{category name: class or EXCLUDE} from label_map.yaml. Stops if a category is missing."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    entry = (config.get("datasets") or {}).get("taco_crops")
    if not entry or "categories" not in entry:
        raise SystemExit(f"configs/label_map.yaml has no datasets -> taco_crops -> categories mapping.")
    mapping = entry["categories"]
    missing = [name for name in category_names if name not in mapping]
    if missing:
        raise SystemExit(f"These categories from {ANNOTATIONS.relative_to(REPO_ROOT)} have no line in "
                         f"configs/label_map.yaml (taco_crops -> categories): {missing}.\n"
                         f"Add each one with its class or EXCLUDE, then run again.")
    classes = set(entry["folders"].values())
    bad = {name: target for name, target in mapping.items() if target != "EXCLUDE" and target not in classes}
    if bad:
        raise SystemExit(f"taco_crops categories map to classes that are not in its folders {sorted(classes)}: {bad}")
    return mapping, sorted(classes)


def main():
    if not ANNOTATIONS.exists():
        raise SystemExit(f"{ANNOTATIONS.relative_to(REPO_ROOT)} not found. Run: python src/data/download.py --only taco")
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    mapping, classes = load_mapping(list(categories.values()))
    images = {img["id"]: img for img in coco["images"]}

    # Objects to crop, grouped by photo so each photo is opened once.
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if mapping[categories[ann["category_id"]]] != "EXCLUDE":
            by_image[ann["image_id"]].append(ann)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for cls in classes:  # every class folder exists, even if no crop lands in it
        (OUT_DIR / cls).mkdir(parents=True)

    saved, too_small, total = Counter(), Counter(), Counter()
    for ann in coco["annotations"]:
        total[categories[ann["category_id"]]] += 1
    missing_photos, size_mismatch, unreadable = [], [], []
    for image_id in sorted(by_image):
        info = images[image_id]
        path = TACO_DIR / info["file_name"]
        if not path.exists():
            missing_photos.append(info["file_name"])
            continue
        try:
            with Image.open(path) as raw:
                photo = ImageOps.exif_transpose(raw).convert("RGB")
        except OSError as err:  # cut-off download, for example
            unreadable.append(f"{info['file_name']} ({err})")
            continue
        if photo.size != (info["width"], info["height"]):
            size_mismatch.append(f"{info['file_name']} {photo.size} vs annotation {(info['width'], info['height'])}")
            continue
        for ann in sorted(by_image[image_id], key=lambda a: a["id"]):
            name = categories[ann["category_id"]]
            left, top, right, bottom = padded_box(ann["bbox"], *photo.size)
            if right - left < MIN_SIDE or bottom - top < MIN_SIDE:
                too_small[name] += 1
                continue
            out = OUT_DIR / mapping[name] / slug(name) / f"img{image_id:06d}_ann{ann['id']:06d}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            photo.crop((left, top, right, bottom)).save(out, "JPEG", quality=95)
            saved[name] += 1

    # ---- Report ----
    print(f"TACO: {len(images)} photos, {sum(total.values())} objects in {len(categories)} categories "
          f"({ANNOTATIONS.relative_to(REPO_ROOT)}).")
    print(f"Crops: box + {PADDING:.0%} padding per side, skipped if under {MIN_SIDE} px on either side.\n")
    print(f"{'category':<28}{'class':<11}{'objects':>8}{'too small':>10}{'saved':>7}")
    for name in sorted(categories.values(), key=lambda n: (mapping[n], n)):
        target = mapping[name]
        if target == "EXCLUDE":
            continue
        print(f"{name:<28}{target:<11}{total[name]:>8}{too_small[name]:>10}{saved[name]:>7}")
    excluded = sum(total[n] for n in categories.values() if mapping[n] == "EXCLUDE")
    print(f"{'(EXCLUDE categories)':<28}{'':<11}{excluded:>8}{'':>10}{0:>7}")
    print("\nPer class:")
    for cls in classes:
        n = sum(saved[name] for name in saved if mapping[name] == cls)
        print(f"  {cls:<10} {n:>5} crops  -> {OUT_DIR.relative_to(REPO_ROOT)}/{cls}/")
    print(f"  {'TOTAL':<10} {sum(saved.values()):>5} crops")
    if missing_photos:
        print(f"\nWARNING: {len(missing_photos)} photos with mapped objects are not downloaded "
              f"(for example {missing_photos[0]}). Run python src/data/download.py --only taco again.")
    if unreadable:
        print(f"\nWARNING: {len(unreadable)} photos with mapped objects could not be read and were skipped: "
              f"{unreadable[:3]}. Delete them and run python src/data/download.py --only taco again.")
    if size_mismatch:
        print(f"\nWARNING: {len(size_mismatch)} photos skipped because their upright size does not match the "
              f"annotations (boxes would be off): {size_mismatch[:3]}")
    print("\nNext: python src/data/build_splits.py")


if __name__ == "__main__":
    main()
