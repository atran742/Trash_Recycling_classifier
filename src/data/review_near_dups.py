"""
review_near_dups.py - spot-check the near-duplicate rules by eye.

Runs the same duplicate search as build_splits.py (nothing is copied or
written to data/) and draws pictures side by side so a person can judge
whether the matches are real copies. Two modes:

    python src/data/review_near_dups.py
        The 20 removed same-label pairs with the HIGHEST distances (the
        borderline ones) -> results/near_dup_review.png

    python src/data/review_near_dups.py --cross-label
        Every label-conflict group (look-alikes with DIFFERENT labels), one
        group per row -> results/cross_label_review.png

Add --max-dhash N to try a different dHash cutoff without editing
label_map.yaml, for example --max-dhash 64 to switch the dHash check off.
Both pictures are git-ignored, like every .png.
"""

import argparse
from collections import defaultdict

from PIL import Image, ImageDraw

# build_splits.py sits in the same folder, so Python can import it directly.
from build_splits import REPO_ROOT, RESULTS_DIR, find_duplicates, load_config, rel

PAIRS_SHEET = RESULTS_DIR / "near_dup_review.png"
CROSS_SHEET = RESULTS_DIR / "cross_label_review.png"
THUMB = 220  # size of each thumbnail in pixels
LABEL_H = 34  # space under each row for text


def paste_thumb(sheet, path, x, y):
    """Paste a small copy of the image at (x, y)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB - 4, THUMB - 4))
        sheet.paste(img, (x + 2, y + 2))


def removed_pairs_sheet(removed_rows, n_pairs):
    """Removed same-label pairs, highest distance first, two pairs per row."""
    # Only removed pairs (not corrupted files), highest distance first.
    pairs = [r for r in removed_rows if "pair" in r]
    pairs.sort(key=lambda r: (-r["distance"], -r["dhash_distance"], r["file"]))
    pairs = pairs[:n_pairs]

    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 2 * THUMB + (cols - 1) * 20, max(rows, 1) * (THUMB + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    print(f"\nTop {len(pairs)} removed pairs by distance:")
    for k, r in enumerate(pairs):
        row, col = divmod(k, cols)
        x0, y0 = col * (2 * THUMB + 20), row * (THUMB + LABEL_H)
        for m, path in enumerate((r["pair"], r["pair_with"])):
            paste_thumb(sheet, path, x0 + m * THUMB, y0)
        reason = r["reason"].replace("_duplicate", "")  # "near" or "exact"
        draw.text((x0 + 4, y0 + THUMB + 2),
                  f"#{k + 1} {reason}  pHash {r['distance']}  dHash {r['dhash_distance']}", fill="black")
        draw.text((x0 + 4, y0 + THUMB + 16),
                  f"{r['pair'].parent.name}/{r['pair'].name}  vs  {r['pair_with'].parent.name}/{r['pair_with'].name}",
                  fill="gray")
        print(f"  #{k + 1:<3} pHash {r['distance']}  dHash {r['dhash_distance']}  {reason:<6} "
              f"{rel(r['pair'])}  vs  {rel(r['pair_with'])}")
    sheet.save(PAIRS_SHEET)
    return PAIRS_SHEET


def cross_label_sheet(review_rows):
    """Every label-conflict group on its own row: image, label, and distance to its match."""
    groups = defaultdict(list)
    for r in review_rows:
        groups[r["group"]].append(r)
    widest = max((len(g) for g in groups.values()), default=1)
    sheet = Image.new("RGB", (widest * THUMB, max(len(groups), 1) * (THUMB + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)

    print(f"\n{len(groups)} label-conflict groups, {len(review_rows)} images:")
    for row, (group_id, members) in enumerate(groups.items()):
        y0 = row * (THUMB + LABEL_H)
        print(f"  {group_id}")
        for col, r in enumerate(members):
            x0 = col * THUMB
            paste_thumb(sheet, REPO_ROOT / r["file"], x0, y0)
            name = r["file"].split("/")[-1]
            draw.text((x0 + 4, y0 + THUMB + 2), f"{group_id} {r['label']}", fill="black")
            draw.text((x0 + 4, y0 + THUMB + 16), f"{name}  p{r['distance']} d{r['dhash_distance']}", fill="gray")
            print(f"    {r['label']:<10} pHash {r['distance']}  dHash {r['dhash_distance']}  {r['file']}  matches {r['matches']}")
    sheet.save(CROSS_SHEET)
    return CROSS_SHEET


def main():
    parser = argparse.ArgumentParser(description="Contact sheets for checking the near-duplicate rules by eye.")
    parser.add_argument("--pairs", type=int, default=20, help="how many removed pairs to show (default 20)")
    parser.add_argument("--cross-label", action="store_true", help="show label-conflict groups instead")
    parser.add_argument("--max-dhash", type=int, help="try this dHash cutoff instead of the one in label_map.yaml")
    args = parser.parse_args()

    config, active, _ = load_config()
    if args.max_dhash is not None:
        config["near_duplicates"]["max_dhash"] = args.max_dhash  # this review only; the yaml is not changed
    _, removed_rows, review_rows, _, _ = find_duplicates(config, active)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = cross_label_sheet(review_rows) if args.cross_label else removed_pairs_sheet(removed_rows, args.pairs)
    print(f"\nSaved {rel(out)}")


if __name__ == "__main__":
    main()
