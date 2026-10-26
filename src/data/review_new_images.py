"""
review_new_images.py - contact sheets of new (non-Kaggle) images, for checking labels by eye.

For each class it draws N random images (seed 42) from the images the newest
sources added to train / validation / wild_test, and saves
    results/new_images_<class>.png       (git-ignored)
Every tile is numbered and captioned with its source category (realwaste/Metal,
taco_crops/recycling/drink-can, ...), so a person can write down per class how
many fit the class under the San Diego guide, how many look wrong, and how many
are too blurry or too tightly cropped to use. A category with more than 10% wrong
is dropped from the mapping in configs/label_map.yaml (EXCLUDE) and the split is rebuilt.

With --by-category it instead saves one sheet per class with a row of images for
every category (results/new_images_<class>_by_category.png), to judge each category.

Run from the repo root, after build_splits.py:
    python src/data/review_new_images.py                      # 30 per class
    python src/data/review_new_images.py --per-class 30 --sources taco_crops realwaste
    python src/data/review_new_images.py --by-category --per-category 12
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
THUMB, LABEL_H, COLS = 200, 30, 6


def category(path):
    """Copy name "taco_crops__recycling__drink-can__img000012_ann000034.jpg" -> "taco_crops/recycling/drink-can".

    realwaste copies look like "realwaste__Metal__Metal_12.jpg" -> "realwaste/Metal".
    """
    parts = Path(path).name.split("__")
    if parts[0] == "taco_crops" and len(parts) >= 4:
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def new_rows(sources):
    """{class: [(split, path)]} for every image from `sources` in train, validation and wild_test."""
    out = defaultdict(list)
    for split in ["train", "validation", "wild_test"]:
        path = MANIFEST_DIR / f"{split}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if Path(row["path"]).name.split("__")[0] in sources:
                    out[row["label"]].append((split, row["path"]))
    return out


def tile(sheet, draw, k, path, caption, cols):
    row, col = divmod(k, cols)
    x, y = col * THUMB, row * (THUMB + LABEL_H)
    with Image.open(REPO_ROOT / path) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB - 6, THUMB - 6))
        sheet.paste(img, (x + 3, y + 3))
    draw.text((x + 4, y + THUMB + 2), caption[0], fill="black")
    draw.text((x + 4, y + THUMB + 15), caption[1], fill="gray")


def per_class_sheets(rows, n):
    for cls in sorted(rows):
        # Sorted by file name (not by split folder), so the same images are drawn after a rebuild.
        ordered = sorted(rows[cls], key=lambda sp: Path(sp[1]).name)
        picks = random.Random(42).sample(ordered, min(n, len(ordered)))
        cols = COLS
        n_rows = (len(picks) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * THUMB, n_rows * (THUMB + LABEL_H)), "white")
        draw = ImageDraw.Draw(sheet)
        print(f"\n{cls}: {len(picks)} of {len(rows[cls])} new images")
        for k, (split, path) in enumerate(picks):
            cat = category(path)
            tile(sheet, draw, k, path, (f"#{k + 1} {cat.split('/', 1)[1][:28]}", f"{cat.split('/')[0]}  {split}"), cols)
            print(f"  #{k + 1:<3} {cat:<45} {split:<11} {path}")
        out = RESULTS_DIR / f"new_images_{cls}.png"
        sheet.save(out)
        print(f"Saved {out.relative_to(REPO_ROOT)}")


def by_category_sheets(rows, n):
    for cls in sorted(rows):
        by_cat = defaultdict(list)
        for split, path in rows[cls]:
            by_cat[category(path)].append(path)
        cats = sorted(by_cat)
        sheet = Image.new("RGB", (n * THUMB, len(cats) * (THUMB + LABEL_H)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, cat in enumerate(cats):
            ordered = sorted(by_cat[cat], key=lambda p: Path(p).name)
            picks = random.Random(42).sample(ordered, min(n, len(ordered)))
            for k, path in enumerate(picks):
                tile(sheet, draw, r * n + k, path, (f"{cat.split('/', 1)[1][:30]} #{k + 1}", f"({len(by_cat[cat])} images)"), n)
        out = RESULTS_DIR / f"new_images_{cls}_by_category.png"
        sheet.save(out)
        print(f"Saved {out.relative_to(REPO_ROOT)}  ({len(cats)} categories: "
              + ", ".join(f"{c} {len(by_cat[c])}" for c in cats) + ")")


def main():
    parser = argparse.ArgumentParser(description="Contact sheets of new images per class, for a label check by eye.")
    parser.add_argument("--sources", nargs="+", default=["taco_crops", "realwaste"], help="datasets to sample")
    parser.add_argument("--per-class", type=int, default=30, help="images per class sheet (default 30)")
    parser.add_argument("--by-category", action="store_true", help="one row per source category instead")
    parser.add_argument("--per-category", type=int, default=12, help="images per category row (default 12)")
    args = parser.parse_args()
    rows = new_rows(set(args.sources))
    if not rows:
        raise SystemExit(f"No images from {args.sources} in data/splits/. Run build_splits.py first.")
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.by_category:
        by_category_sheets(rows, args.per_category)
    else:
        per_class_sheets(rows, args.per_class)


if __name__ == "__main__":
    main()
