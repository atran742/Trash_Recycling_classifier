"""
freeze_wild_split.py - freeze both test sets and lock the split of every non-Kaggle image.

Run ONCE, right after a build the team wants to freeze (done at stamp def16b3,
manifest_sha256 08d058850b31, on 2026-10-26). It writes three files in configs/frozen/:

    test.csv               copy of data/splits/test.csv       (the Kaggle benchmark)
    wild_test.csv          copy of data/splits/wild_test.csv  (the wild test)
    wild_split_lock.csv    source,label,split for every non-Kaggle image in train,
                           validation and wild_test (source = its file under data/raw/)

and prints the two sha256 lines to put under `frozen:` in configs/label_map.yaml.
From then on build_splits.py:
  - keeps every locked image in its locked split,
  - puts every image NOT in the lock (new sources: amy_images, team_photos, any new
    dataset, or new files in an old one) only into train or validation,
  - stops if data/splits/test.csv or wild_test.csv differs from the frozen copy.

Re-running this script re-freezes both test sets: only do that on a team decision,
because every earlier score stops being comparable.

Run from the repo root, after build_splits.py:
    python src/data/freeze_wild_split.py
"""

import csv
import hashlib
import json
import shutil
from collections import Counter

from build_splits import (MANIFEST_DIR, RAW_DIR, REPO_ROOT, WEIGHTS_JSON, collect_images, copy_name,
                          load_config, manifest_fingerprint, rel)

FROZEN_DIR = REPO_ROOT / "configs" / "frozen"
LOCK_COLUMNS = ["source", "label", "split"]
LOCKED_SPLITS = ["train", "validation", "wild_test"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # The manifests on disk must be the build recorded in class_weights.json.
    with open(WEIGHTS_JSON) as f:
        recorded = json.load(f)["manifest_sha256"]
    if manifest_fingerprint() != recorded:
        raise SystemExit(f"data/splits/ (fingerprint {manifest_fingerprint()}) is not the build recorded in "
                         f"{rel(WEIGHTS_JSON)} ({recorded}). Run python src/data/build_splits.py first.")

    config, active, _ = load_config()
    base = set(config.get("base_datasets") or [])

    # Every non-Kaggle row in the manifests: (split, class, copy name).
    wanted = {}
    for split in LOCKED_SPLITS:
        with open(MANIFEST_DIR / f"{split}.csv", newline="") as f:
            for row in csv.DictReader(f):
                name = row["path"].split("/")[-1]
                if name.split("__")[0] not in base:
                    wanted[(row["label"], name)] = split
    print(f"{len(wanted)} non-Kaggle images in train / validation / wild_test.")

    # Match each one to its source file: copy names are built from the source path.
    items, _, _ = collect_images(config, active, removed_rows=[])
    rows, seen = [], set()
    for item in items:
        if item["path"].relative_to(RAW_DIR).parts[0] in base:
            continue
        key = (item["label"], copy_name(item))
        if key in wanted:
            if key in seen:
                raise SystemExit(f"Two source files give the copy name {key[1]}; cannot lock them safely.")
            seen.add(key)
            rows.append({"source": rel(item["path"]), "label": item["label"], "split": wanted[key]})
    missing = set(wanted) - seen
    if missing:
        raise SystemExit(f"{len(missing)} manifest rows have no matching source file (for example "
                         f"{sorted(missing)[0][1]}). Is data/raw/ complete?")

    FROZEN_DIR.mkdir(exist_ok=True)
    rows.sort(key=lambda r: r["source"])
    with open(FROZEN_DIR / "wild_split_lock.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOCK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for name in ["test.csv", "wild_test.csv"]:
        shutil.copyfile(MANIFEST_DIR / name, FROZEN_DIR / name)

    counts = Counter((r["source"].split("/")[2], r["split"]) for r in rows)
    print(f"Wrote {rel(FROZEN_DIR / 'wild_split_lock.csv')}: {len(rows)} locked images")
    for (source, split), n in sorted(counts.items()):
        print(f"  {source:<14}{split:<12}{n:>6}")
    print(f"Wrote {rel(FROZEN_DIR / 'test.csv')} and {rel(FROZEN_DIR / 'wild_test.csv')}.")
    print("\nPut these under `frozen:` in configs/label_map.yaml:")
    print(f"  test_csv_sha256: {sha256(FROZEN_DIR / 'test.csv')}")
    print(f"  wild_test_csv_sha256: {sha256(FROZEN_DIR / 'wild_test.csv')}")


if __name__ == "__main__":
    main()
