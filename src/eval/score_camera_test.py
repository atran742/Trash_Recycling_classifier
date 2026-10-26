"""
score_camera_test.py - score a model on the held-out demo-room photos in data/raw/camera_test/.

Those photos were taken with the webcam or the Pi camera in the demo room and
are NEVER split or trained on (build_splits.py refuses to read that folder), so
this is the closest number we have to how the demo will go.

What it does, in order:
  1. Collects every photo in data/raw/camera_test/<class>/ (folder name = label,
     matched to configs/label_map.yaml ignoring capital letters; sub-folders and
     iPhone HEIC files included).
  2. Near-copy check: compares each photo against every TRAIN image with the same
     pHash / dHash rules as build_splits.py and prints a WARNING for each near copy.
     A near copy means the photo (or one like it) was trained on, so its score is
     too kind. The run continues; move the photo and score again. --skip-leak-check
     skips this (it hashes all train images, which takes a minute or two).
  3. Runs the model over the photos, turned upright by their EXIF tag, with the same
     preprocessing as training (shared with export_predictions.py), and writes
     results/camera_test_predictions.csv  (path, true_label, predicted_label, confidence)
  4. Scores that file with src/eval/evaluate.py. If results/val_predictions.csv
     exists (from export_predictions.py) it is the --tune file, so the "I'm not sure"
     cutoff still comes from validation and not from these photos. Without it,
     evaluate.py runs on the one file and prints its same-file warning.

Run from the repo root:
    python src/eval/score_camera_test.py --model models/waste_classifier.pth --arch mobilenet_v3_small
Options:
    --tune results/val_predictions.csv   the validation predictions used to pick the cutoff (default)
    --target 0.85                        passed through to evaluate.py
    --skip-leak-check                    do not compare against the train images
"""

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# The pipeline's hashing rules and the export's model loading live in two other
# scripts. Import them instead of copying, so one change to the rules reaches here.
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
from build_splits import (CAMERA_TEST_DIR, CONFIG_PATH, CONVERT_TO_JPEG, HEIF_AVAILABLE,  # noqa: E402
                          IMAGE_EXTENSIONS, MANIFEST_DIR, RESULTS_DIR, hamming_distances, image_hashes,
                          is_near_match, is_valid_image, normalise_folder_name, rel)
from export_predictions import (ARCHITECTURES, class_order, load_model, match_model_classes,  # noqa: E402
                                predict_split, preprocessing_for)

OUTPUT_CSV = RESULTS_DIR / "camera_test_predictions.csv"
EVALUATE = REPO_ROOT / "src" / "eval" / "evaluate.py"


def collect_camera_photos(classes):
    """Rows {"path", "label"} for every photo in data/raw/camera_test/<class>/.

    Folder names are matched to the class list ignoring capital letters, dashes
    and underscores ("Recycling", "recycling" and "re-cycling" all work).
    A folder that matches no class stops the run (it would be a silent label error).
    """
    if not CAMERA_TEST_DIR.is_dir():
        raise SystemExit(f"{rel(CAMERA_TEST_DIR)}/ not found. Create it with one folder per class "
                         f"({', '.join(classes)}) and put the demo-room photos inside. See its README.md.")
    by_name = {normalise_folder_name(c): c for c in classes}
    rows, skipped, heic = [], Counter(), 0
    for folder in sorted(p for p in CAMERA_TEST_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = by_name.get(normalise_folder_name(folder.name))
        if label is None:
            raise SystemExit(f"{rel(folder)}/ matches no class in {rel(CONFIG_PATH)} ({', '.join(classes)}). "
                             f"Rename the folder.")
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            suffix = path.suffix.lower()
            if path.name.startswith(".") or suffix not in IMAGE_EXTENSIONS | CONVERT_TO_JPEG:
                skipped["not an image"] += 1
                continue
            if suffix in {".heic", ".heif"}:
                heic += 1
                if not HEIF_AVAILABLE:
                    raise SystemExit(f"{rel(path)} is an iPhone HEIC photo but pillow-heif is not installed. "
                                     f"Run: pip install -r requirements.txt")
            if not is_valid_image(path):
                skipped["corrupted"] += 1
                print(f"  WARNING: {rel(path)} cannot be opened, skipped.")
                continue
            rows.append({"path": rel(path), "label": label})
    if not rows:
        raise SystemExit(f"No photos in {rel(CAMERA_TEST_DIR)}/<class>/. Put the demo-room photos there first "
                         f"(see {rel(CAMERA_TEST_DIR)}/README.md).")
    counts = Counter(r["label"] for r in rows)
    print(f"Camera test photos: {len(rows)} in {rel(CAMERA_TEST_DIR)}/  "
          + ", ".join(f"{c} {counts.get(c, 0)}" for c in classes)
          + (f"  ({heic} HEIC)" if heic else "")
          + (f"  skipped: {dict(skipped)}" if skipped else ""))
    missing = [c for c in classes if counts.get(c, 0) == 0]
    if missing:
        print(f"  WARNING: no photos for {', '.join(missing)}. Those classes get no recall score.")
    return rows


def near_copy_check(rows, rules):
    """Warn about camera photos that are near copies of a TRAIN image.

    Same fingerprints and the same cutoffs (near_duplicates in label_map.yaml) as
    build_splits.py. Camera photos are turned upright first, like every other
    phone source; the train copies on disk are upright already.
    Returns the number of flagged photos.
    """
    train_csv = MANIFEST_DIR / "train.csv"
    if not train_csv.exists():
        print(f"  WARNING: {rel(train_csv)} not found, so the near-copy check against train was skipped. "
              f"Run python src/data/build_splits.py first.")
        return 0
    with open(train_csv, newline="") as f:
        train_rows = list(csv.DictReader(f))
    print(f"Near-copy check: {len(rows)} camera photos against {len(train_rows)} train images "
          f"(pHash <= {rules['max_phash']} and dHash <= {rules['max_dhash']})...")
    train_hashes = [image_hashes(REPO_ROOT / r["path"]) for r in train_rows]
    train_p = np.array([h[0] for h in train_hashes], dtype=np.uint64)
    train_d = np.array([h[1] for h in train_hashes], dtype=np.uint64)
    flagged = 0
    for r in rows:
        p, d = image_hashes(REPO_ROOT / r["path"], upright=True)
        p_dist, d_dist = hamming_distances(p, train_p), hamming_distances(d, train_d)
        hits = np.nonzero(is_near_match(p_dist, d_dist, rules))[0]
        if len(hits):
            flagged += 1
            best = min(hits, key=lambda j: (int(p_dist[j]) + int(d_dist[j]), int(j)))
            print(f"  WARNING: {r['path']} is a near copy of train image {train_rows[best]['path']} "
                  f"(pHash {int(p_dist[best])}, dHash {int(d_dist[best])}). Its score is not a fair test.")
    if flagged:
        print(f"  {flagged} camera photo(s) are near copies of train images. Move them out of "
              f"{rel(CAMERA_TEST_DIR)}/ (and out of any training source) for an honest number.")
    else:
        print("  No camera photo is a near copy of a train image.")
    return flagged


def main():
    parser = argparse.ArgumentParser(description="Score a model on the held-out demo-room photos.")
    parser.add_argument("--model", type=Path, required=True, help="model file: TorchScript, whole model, or state_dict")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), help="network to rebuild when --model is a state_dict")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tune", type=Path, default=RESULTS_DIR / "val_predictions.csv",
                        help="validation predictions CSV that picks the cutoff (default results/val_predictions.csv)")
    parser.add_argument("--target", type=float, help="passed to evaluate.py (accuracy on answered items)")
    parser.add_argument("--skip-leak-check", action="store_true", help="skip the near-copy check against train")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    classes = class_order()
    rows = collect_camera_photos(classes)

    flagged = 0 if args.skip_leak_check else near_copy_check(rows, config["near_duplicates"])

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, model_classes = load_model(args.model, args.arch, len(classes), device)
    if model_classes is not None:
        classes = match_model_classes(model_classes, classes)
    preprocess = preprocessing_for(args.arch)
    print(f"\nScoring {len(rows)} photos on {device}, classes {classes}, photos turned upright by EXIF...")
    predictions = predict_split(model, rows, classes, preprocess, device, args.batch_size, upright=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    correct = sum(p["true_label"] == p["predicted_label"] for p in predictions)
    print(f"Wrote {rel(OUTPUT_CSV)}  ({correct} of {len(predictions)} right, plain accuracy "
          f"{correct / len(predictions):.3f})")

    # Hand over to evaluate.py: cutoff from validation when we have it.
    cmd = [sys.executable, str(EVALUATE)]
    if args.tune.exists():
        cmd += ["--tune", str(args.tune), "--report", str(OUTPUT_CSV)]
    else:
        print(f"\n{args.tune} not found (run src/eval/export_predictions.py to make it), so evaluate.py "
              f"runs on the camera file alone and prints its same-file warning.")
        cmd += [str(OUTPUT_CSV)]
    if args.target is not None:
        cmd += ["--target", str(args.target)]
    print("\n" + " ".join(["python"] + cmd[1:]) + "\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if flagged:
        print(f"\nREMINDER: {flagged} photo(s) above are near copies of train images; the numbers are too kind.")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
