"""
check_leakage.py - look for near copies of TRAIN images inside validation, test and wild_test.

Why: if (almost) the same photo is in train and in test, the model has
effectively seen the test answer during training, and its test score looks
better than it really is. build_splits.py already removes near copies with
strict cutoffs; this script uses LOOSER cutoffs to find anything that slipped
through, so a person can look at it.

READ-ONLY: it never moves, deletes or copies images, and never edits the
manifests. It only writes two report files:
    results/leakage_pairs.csv   every flagged pair
    results/leakage_review.png  side-by-side pictures of flagged pairs (git-ignored)

Run from the repo root, after build_splits.py:
    python src/data/check_leakage.py
Options for the picture only (the CSV always lists every flagged pair):
    --min-dhash 3 --max-dhash 5      only show pairs in this dHash range
    --sample 20                      show a random sample (seed 42) instead of the first 40
    --sheet results/some_name.png    save the picture under another name

A clean result: only the 15 look-alike pairs already judged in
docs/dataset_inventory.md, and 0 true copies. A new true copy goes into
known_duplicates in configs/label_map.yaml.
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Use the exact same fingerprint functions as build_splits.py, so the distances match.
from build_splits import REPO_ROOT, RESULTS_DIR, hamming_distances, image_hashes

MANIFEST_DIR = REPO_ROOT / "data" / "splits"
PAIRS_CSV = RESULTS_DIR / "leakage_pairs.csv"
SHEET_PNG = RESULTS_DIR / "leakage_review.png"
PAIR_COLUMNS = ["split", "file", "train_match", "file_label", "train_label", "phash_distance", "dhash_distance"]

# Looser than build_splits.py on purpose: we want to SEE borderline pairs.
FLAG_MAX_PHASH = 6
FLAG_MAX_DHASH = 8

THUMB = 200   # thumbnail size in pixels
LABEL_H = 34  # space for text under each pair


def read_manifest(split_name):
    """Return the rows of data/splits/<split>.csv as a list of {"path", "label"} dicts."""
    with open(MANIFEST_DIR / f"{split_name}.csv") as f:
        return list(csv.DictReader(f))


def fingerprints(rows):
    """pHash and dHash for every image in the rows, as two numpy arrays."""
    hashes = [image_hashes(REPO_ROOT / row["path"]) for row in rows]
    phashes = np.array([h[0] for h in hashes], dtype=np.uint64)
    dhashes = np.array([h[1] for h in hashes], dtype=np.uint64)
    return phashes, dhashes


def find_flagged_pairs(train_rows, other_rows, split_name):
    """For each validation/test image, find the closest train image that passes the flag rule.

    "Passes the flag rule" means pHash distance <= 6 AND dHash distance <= 8.
    Among the train images that pass, the closest is the one with the smallest
    pHash + dHash (ties broken by pHash, then dHash, then file order).
    Images with no train image passing the rule are not flagged.
    """
    train_p, train_d = fingerprints(train_rows)
    other_p, other_d = fingerprints(other_rows)
    flagged = []
    for k, row in enumerate(other_rows):
        p_dist = hamming_distances(other_p[k], train_p)
        d_dist = hamming_distances(other_d[k], train_d)
        candidates = np.nonzero((p_dist <= FLAG_MAX_PHASH) & (d_dist <= FLAG_MAX_DHASH))[0]
        if len(candidates) == 0:
            continue
        best = min(candidates, key=lambda j: (int(p_dist[j]) + int(d_dist[j]), int(p_dist[j]), int(d_dist[j]), int(j)))
        flagged.append({
            "split": split_name,
            "file": row["path"],
            "train_match": train_rows[best]["path"],
            "file_label": row["label"],
            "train_label": train_rows[best]["label"],
            "phash_distance": int(p_dist[best]),
            "dhash_distance": int(d_dist[best]),
        })
    return flagged


def print_count_table(pairs, sizes):
    """Flagged pairs per split, broken down by dHash distance 0 to 8. `sizes`: split -> images checked."""
    print(f"\nFlagged pairs (pHash <= {FLAG_MAX_PHASH} and dHash <= {FLAG_MAX_DHASH}) by dHash distance:")
    header = f"{'split':<12}" + "".join(f"{d:>5}" for d in range(FLAG_MAX_DHASH + 1)) + f"{'total':>8}{'of':>6}"
    print(header)
    print("-" * len(header))
    for split_name in sizes:
        by_d = Counter(p["dhash_distance"] for p in pairs if p["split"] == split_name)
        print(f"{split_name:<12}" + "".join(f"{by_d[d]:>5}" for d in range(FLAG_MAX_DHASH + 1))
              + f"{sum(by_d.values()):>8}{sizes[split_name]:>6}")
    all_d = Counter(p["dhash_distance"] for p in pairs)
    print(f"{'all':<12}" + "".join(f"{all_d[d]:>5}" for d in range(FLAG_MAX_DHASH + 1))
          + f"{len(pairs):>8}{sum(sizes.values()):>6}")
    print(f"\n{len(pairs)} flagged pairs.")


def draw_sheet(pairs, out_path):
    """Side-by-side picture: validation/test image on the left, its train match on the right."""
    cols = 2  # two pairs per row
    rows = max((len(pairs) + cols - 1) // cols, 1)
    sheet = Image.new("RGB", (cols * 2 * THUMB + (cols - 1) * 20, rows * (THUMB + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, p in enumerate(pairs):
        row, col = divmod(k, cols)
        x0, y0 = col * (2 * THUMB + 20), row * (THUMB + LABEL_H)
        for m, rel_path in enumerate((p["file"], p["train_match"])):
            with Image.open(REPO_ROOT / rel_path) as img:
                img = img.convert("RGB")
                img.thumbnail((THUMB - 4, THUMB - 4))
                sheet.paste(img, (x0 + m * THUMB + 2, y0 + 2))
        draw.text((x0 + 4, y0 + THUMB + 2),
                  f"#{k + 1} {p['split']}  pHash {p['phash_distance']}  dHash {p['dhash_distance']}", fill="black")
        draw.text((x0 + 4, y0 + THUMB + 16),
                  f"{Path(p['file']).name.split('__')[-1]} ({p['file_label']})  vs  "
                  f"{Path(p['train_match']).name.split('__')[-1]} ({p['train_label']})", fill="gray")
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Find near copies of train images in validation and test (read-only).")
    parser.add_argument("--min-dhash", type=int, default=0, help="picture only: smallest dHash distance to show")
    parser.add_argument("--max-dhash", type=int, default=FLAG_MAX_DHASH, help="picture only: largest dHash distance to show")
    parser.add_argument("--sample", type=int, help="picture only: random sample of this many pairs (seed 42)")
    parser.add_argument("--limit", type=int, default=40, help="picture only: most pairs to show (default 40)")
    parser.add_argument("--sheet", type=Path, default=SHEET_PNG, help="where to save the picture")
    args = parser.parse_args()

    train = read_manifest("train")
    names = ["validation", "test"] + (["wild_test"] if (MANIFEST_DIR / "wild_test.csv").exists() else [])
    others = {name: read_manifest(name) for name in names}
    print("Comparing " + ", ".join(f"{len(rows)} {name}" for name, rows in others.items())
          + f" images against {len(train)} train images...")

    pairs = []
    for split_name, rows in others.items():
        pairs += find_flagged_pairs(train, rows, split_name)
    # Sort by dHash, then pHash, then file name, so the order never changes between runs.
    pairs.sort(key=lambda p: (p["dhash_distance"], p["phash_distance"], p["split"], p["file"]))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(PAIRS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pairs)

    print_count_table(pairs, {name: len(rows) for name, rows in others.items()})
    print(f"Saved {PAIRS_CSV.relative_to(REPO_ROOT)}")

    # The picture: filter by dHash range, then a seeded sample or the first N.
    shown = [p for p in pairs if args.min_dhash <= p["dhash_distance"] <= args.max_dhash]
    if args.sample is not None:
        shown = random.Random(42).sample(shown, min(args.sample, len(shown)))
        shown.sort(key=lambda p: (p["dhash_distance"], p["phash_distance"], p["split"], p["file"]))
    shown = shown[: args.limit]
    out = args.sheet if args.sheet.is_absolute() else REPO_ROOT / args.sheet
    draw_sheet(shown, out)
    print(f"Saved {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out} ({len(shown)} pairs shown)")


if __name__ == "__main__":
    main()
