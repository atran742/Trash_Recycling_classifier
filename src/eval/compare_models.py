"""
compare_models.py - score two models on all three test sets and apply the swap rule.

Give it the CURRENT model (the one in use) and a NEW model (the one that wants to
replace it). Both are scored the same way, on:
    benchmark    data/splits/test.csv        (1,270 frozen Kaggle images)
    wild test    data/splits/wild_test.csv   (298 frozen TACO / RealWaste images)
    camera test  data/raw/camera_test/<class>/  (demo-room photos, never trained on)
with the preprocessing and model loading of export_predictions.py, so the numbers
match the two-command evaluation in docs/evaluation.md.

It prints one table (macro F1 and per-class recall, for each model on each test set)
and then the swap rule from docs/evaluation.md with a verdict. The camera test is
reported 2 ways:
    per frame  every photo counts once
    per item   the frames of one recording (capture_photos.py video mode, files named like
               trash_item03_f2.jpg) are one item; every other photo is an item of its own.
               An item counts as correct when MORE than half of its frames are right; its
               predicted label is the label more than half of its frames got (with no such
               label, "no majority", which is wrong for every class).
Rule 3 compares the per-frame camera test macro F1, as before (so a 5-frame recording
weighs 5 times as much as one single photo); the item count decides whether there are
enough camera_test photos to judge it.
    SWAP     all three rules pass: the new model may replace the current one
    KEEP     at least one rule fails: keep the current model (also when rule 1 or 2
             fails before the camera test can be judged: the new model cannot win then)
    BLOCKED  rules 1 and 2 pass, but rule 3 cannot be judged yet: camera_test has fewer
             than 4 items in some class, or some camera_test photos are near copies
             of train images (the near-copy check of score_camera_test.py)
Scores use every item with no "I'm not sure" cutoff (the "all items" numbers of evaluate.py).
Macro F1 averages only the classes present in a test set (the wild test has no hazardous).

It also saves results/camera_test_mistakes.png: every camera_test photo (frame) that either
model got wrong, with both predictions under it (git-ignored, share it in the PR).

Run from the repo root:
    python src/eval/compare_models.py models/waste_classifier.pth models/candidate.pth
Options:
    --arch mobilenet_v3_small   network to rebuild for state_dict files (the default)
    --skip-leak-check           do not compare camera_test photos with the train images
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
# Reuse the existing scripts instead of copying them, so a fix there reaches here.
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
from build_splits import CAMERA_TEST_DIR, CONVERT_TO_JPEG, IMAGE_EXTENSIONS, RESULTS_DIR, load_config, rel  # noqa: E402
from evaluate import macro_f1  # noqa: E402
from export_predictions import (ARCHITECTURES, class_order, load_model, match_model_classes,  # noqa: E402
                                predict_split, preprocessing_for, read_manifest)
from score_camera_test import collect_camera_photos, near_copy_check  # noqa: E402
from recording_names import ITEM_NAME  # noqa: E402  (the recording file names, e.g. trash_item03_f2.jpg)

MIN_CAMERA_ITEMS_PER_CLASS = 4  # fewer camera_test items than this in any class = BLOCKED
NO_MAJORITY = "no majority"     # an item's prediction when no label got more than half of its frames
BENCHMARK_TOLERANCE = 0.01  # rule 1: the new benchmark macro F1 may drop by at most this much
ROUNDING = 1e-9             # computers store 0.785 - 0.01 a hair off; this keeps an exact tie a pass
PHOTO_SUFFIXES = IMAGE_EXTENSIONS | CONVERT_TO_JPEG  # "|" joins two sets: .jpg, .png, ... plus iPhone .heic
SHEET_PATH = RESULTS_DIR / "camera_test_mistakes.png"
THUMB, CAPTION_H, COLS = 200, 50, 6
WIDTH = 77                  # width of the printed table


def camera_photo_count():
    """How many image files sit in data/raw/camera_test/<class>/ (0 if the folder is empty)."""
    if not CAMERA_TEST_DIR.is_dir():
        return 0
    # Only photos inside a class folder count; a photo loose in camera_test/ itself has no label.
    return sum(1 for p in CAMERA_TEST_DIR.rglob("*")
               if p.is_file() and p.parent != CAMERA_TEST_DIR and not p.name.startswith(".")
               and p.suffix.lower() in PHOTO_SUFFIXES)


def item_key(row):
    """Which item a camera_test photo belongs to.

    Frames of one recording (trash_item03_f1.jpg ... _f5.jpg in one class folder) share the
    key "trash item03"; any other photo is an item of its own, keyed by its path.
    """
    found = ITEM_NAME.fullmatch(Path(row["path"]).name)
    if found:
        return f"{row['label']} item{int(found['item']):02d}"
    return row["path"]


def item_labels(rows, predictions):
    """(true labels, predicted labels) with one entry per item instead of one per frame.

    The item's prediction is the label MORE than half of its frames got, so an item is
    right exactly when most of its frames are right (2 of 4 right is not most).
    """
    frames = defaultdict(list)  # item -> [(true label, predicted label) of each frame]
    for r, p in zip(rows, predictions):
        frames[item_key(r)].append((r["label"], p["predicted_label"]))
    true, pred = [], []
    for pairs in frames.values():
        top_label, votes = Counter(predicted for _, predicted in pairs).most_common(1)[0]
        true.append(pairs[0][0])
        pred.append(top_label if votes > len(pairs) / 2 else NO_MAJORITY)
    return true, pred


def items_per_class(rows):
    """How many different items each class has in camera_test (a Counter: class -> items)."""
    one_row_per_item = {item_key(r): r for r in rows}  # a dict keeps one row per key: the 5 frames become 1
    return Counter(r["label"] for r in one_row_per_item.values())


def score(rows, predictions):
    """Macro F1 and recall per class of one test set, one entry per photo."""
    return score_labels([r["label"] for r in rows], [p["predicted_label"] for p in predictions])


def score_labels(true, pred):
    """Macro F1 (over the classes present) and recall per class, from the true and predicted labels."""
    present = sorted(set(true))
    recall = {c: sum(t == c and p == c for t, p in zip(true, pred)) / true.count(c) for c in present}
    return {"macro_f1": macro_f1(true, pred, present), "recall": recall, "n": len(true)}


def run_model(path, arch, device, test_sets):
    """Load one model and predict every test set. Returns {test set name: list of predictions}."""
    classes = class_order()
    model, model_classes = load_model(path, arch, len(classes), device)
    if model_classes is not None:
        classes = match_model_classes(model_classes, classes)  # the checkpoint's own class order
    preprocess = preprocessing_for(arch)
    out = {}
    for name, (rows, upright) in test_sets.items():
        if rows:
            print(f"  {name}: {len(rows)} images")
            out[name] = predict_split(model, rows, classes, preprocess, device, 32, upright=upright)
    return out


def print_table(sections, scores, classes):
    """One table: each test set, macro F1 then recall per class, current vs new.

    sections = [(title, key in scores, how many images or items)], in print order.
    """
    print(f"\n{'test set':<30}{'score':<20}{'current':>9}{'new':>9}{'change':>9}")
    print("-" * WIDTH)
    for name, key, n in sections:
        title = f"{name} ({n})"
        if not n:
            print(f"{title:<30}{'(no photos)':<20}{'-':>9}{'-':>9}{'-':>9}")
            continue
        cur, new = scores["current"][key], scores["new"][key]
        lines = [("macro F1", cur["macro_f1"], new["macro_f1"])]
        for c in classes:
            if c in cur["recall"]:
                lines.append((f"recall {c}", cur["recall"][c], new["recall"][c]))
            else:
                lines.append((f"recall {c}", None, None))
        for i, (label, a, b) in enumerate(lines):
            first = title if i == 0 else ""
            if a is None:
                class_name = label.removeprefix("recall ")
                print(f"{first:<30}{label:<20}{'n/a':>9}{'n/a':>9}{'':>9}   (no {class_name} items)")
            else:
                print(f"{first:<30}{label:<20}{a:>9.3f}{b:>9.3f}{b - a:>+9.3f}")
        print("-" * WIDTH)


def swap_verdict(scores, camera_items, classes, leaked=0, leak_checked=True):
    """Apply the three rules of docs/evaluation.md. Returns (verdict, list of lines to print).

    camera_items = how many camera_test ITEMS each class has (a recording is one item,
    however many frames it saved). leaked = how many camera_test photos are near copies
    of train images.
    """
    cur, new = scores["current"], scores["new"]
    lines = []
    b_cur, b_new = cur["benchmark"]["macro_f1"], new["benchmark"]["macro_f1"]
    rule1 = b_new >= b_cur - BENCHMARK_TOLERANCE - ROUNDING
    lines.append(f"1. Benchmark macro F1 drops by no more than {BENCHMARK_TOLERANCE}: "
                 f"{b_new:.4f} {'>=' if rule1 else '<'} {b_cur - BENCHMARK_TOLERANCE:.4f}  -> {'PASS' if rule1 else 'FAIL'}")
    w_cur, w_new = cur["wild test"]["macro_f1"], new["wild test"]["macro_f1"]
    rule2 = w_new > w_cur
    lines.append(f"2. Wild test macro F1 goes up: {w_new:.4f} {'>' if rule2 else '<='} {w_cur:.4f}  "
                 f"-> {'PASS' if rule2 else 'FAIL'}")

    # Classes with too few camera_test items (.get(c, 0) gives 0 for a class with no photos).
    short = {c: camera_items.get(c, 0) for c in classes if camera_items.get(c, 0) < MIN_CAMERA_ITEMS_PER_CLASS}
    if short or leaked:
        why = []
        if short:
            why.append(f"fewer than {MIN_CAMERA_ITEMS_PER_CLASS} items in: "
                       + ", ".join(f"{c} ({n})" for c, n in short.items())
                       + ". Add items with: python src/data/capture_photos.py --dest camera_test "
                         "(Shift+number records one item)")
        if leaked:
            why.append(f"{leaked} camera_test photo(s) are near copies of train images (see the warnings above)")
        lines.append("3. Camera test macro F1 does not go down: NOT JUDGED, " + "; ".join(why))
        # A model that already fails rule 1 or 2 can never win, so the answer is KEEP, not BLOCKED.
        return ("BLOCKED" if rule1 and rule2 else "KEEP"), lines
    c_cur, c_new = cur["camera test"]["macro_f1"], new["camera test"]["macro_f1"]
    rule3 = c_new >= c_cur
    lines.append(f"3. Camera test macro F1 (per frame) does not go down: {c_new:.4f} {'>=' if rule3 else '<'} {c_cur:.4f}  "
                 f"-> {'PASS' if rule3 else 'FAIL'}")
    if not leak_checked:
        lines.append("   Note: the near-copy check against train did NOT run, so rule 3 assumes "
                     "no camera_test photo was trained on.")
    return ("SWAP" if rule1 and rule2 and rule3 else "KEEP"), lines


def remove_old_sheet():
    """Delete an old results/camera_test_mistakes.png: it would show mistakes from another run."""
    if SHEET_PATH.exists():
        SHEET_PATH.unlink()


def save_mistake_sheet(rows, predictions):
    """Contact sheet of every camera_test photo that either model got wrong. Returns how many."""
    wrong = [(r, pc, pn) for r, pc, pn in zip(rows, predictions["current"], predictions["new"])
             if pc["predicted_label"] != r["label"] or pn["predicted_label"] != r["label"]]
    if not wrong:
        remove_old_sheet()
        return 0
    n_rows = (len(wrong) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * THUMB, n_rows * (THUMB + CAPTION_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (r, pc, pn) in enumerate(wrong):
        x, y = (k % COLS) * THUMB, (k // COLS) * (THUMB + CAPTION_H)
        with Image.open(REPO_ROOT / r["path"]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")  # upright, as the models saw it
            img.thumbnail((THUMB - 6, THUMB - 6))
            sheet.paste(img, (x + 3, y + 3))
        found = ITEM_NAME.fullmatch(Path(r["path"]).name)
        tag = f"item{int(found['item']):02d} f{found['frame']}" if found else f"#{k + 1}"
        draw.text((x + 4, y + THUMB + 1), f"true: {r['label']}  {tag}", fill="black")
        for i, (who, p) in enumerate([("current", pc), ("new", pn)]):
            color = "green" if p["predicted_label"] == r["label"] else "red"
            draw.text((x + 4, y + THUMB + 16 + 15 * i), f"{who}: {p['predicted_label']} {float(p['confidence']):.2f}",
                      fill=color)
    RESULTS_DIR.mkdir(exist_ok=True)
    sheet.save(SHEET_PATH)
    return len(wrong)


def main():
    parser = argparse.ArgumentParser(description="Score two models on all three test sets and apply the swap rule.")
    parser.add_argument("current", type=Path, help="the model in use now, e.g. models/waste_classifier.pth")
    parser.add_argument("new", type=Path, help="the model that wants to replace it, e.g. models/candidate.pth")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=sorted(ARCHITECTURES),
                        help="network to rebuild for state_dict files (default mobilenet_v3_small)")
    parser.add_argument("--skip-leak-check", action="store_true",
                        help="do not compare camera_test photos with the train images")
    args = parser.parse_args()
    for path in (args.current, args.new):
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    # An NVIDIA GPU if there is one, else the Apple GPU (mps), else the CPU.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    classes = class_order()

    # The three test sets: (rows, turn upright by EXIF?). Split copies are upright already;
    # camera photos come straight from a camera, so they are turned like score_camera_test.py does.
    camera_rows = []
    if camera_photo_count():
        try:
            camera_rows = collect_camera_photos(classes)
        except SystemExit as err:  # e.g. every photo is broken: report it, then judge as "no photos"
            print(f"Camera test: {err}")
    leaked, leak_checked = 0, False
    if not camera_rows:
        print(f"Camera test: no photos in {rel(CAMERA_TEST_DIR)}/<class>/ yet.")
    elif args.skip_leak_check:
        print("Near-copy check against train skipped (--skip-leak-check).")
    else:
        config, _, _ = load_config()
        leaked = near_copy_check(camera_rows, config["near_duplicates"])  # how many photos look like a train image
        leak_checked = (REPO_ROOT / "data" / "splits" / "train.csv").exists()  # the check skips itself without it
    test_sets = {
        "benchmark": (read_manifest("test"), False),
        "wild test": (read_manifest("wild_test"), False),
        "camera test": (camera_rows, True),
    }

    predictions, scores = {}, {}
    camera_items = items_per_class(camera_rows)  # items per class (a recording counts once)
    if camera_rows:
        print(f"Camera test items: {sum(camera_items.values())} from {len(camera_rows)} photos  "
              + ", ".join(f"{c} {camera_items.get(c, 0)}" for c in classes))
    for who, path in [("current", args.current), ("new", args.new)]:
        print(f"\nScoring the {who} model {path} on {device}:")
        predictions[who] = run_model(path, args.arch, device, test_sets)
        scores[who] = {name: score(test_sets[name][0], preds) for name, preds in predictions[who].items()}
        if camera_rows:  # the same camera predictions again, one entry per item
            item_true, item_pred = item_labels(camera_rows, predictions[who]["camera test"])
            scores[who]["camera test per item"] = score_labels(item_true, item_pred)

    print_table([("benchmark", "benchmark", len(test_sets["benchmark"][0])),
                 ("wild test", "wild test", len(test_sets["wild test"][0])),
                 ("camera test, per frame", "camera test", len(camera_rows)),
                 ("camera test, per item", "camera test per item", sum(camera_items.values()))],
                scores, classes)

    verdict, lines = swap_verdict(scores, camera_items, classes, leaked, leak_checked)
    print("\nSwap rule (docs/evaluation.md):")
    for line in lines:
        print("  " + line)
    explain = {"SWAP": f"{args.new} may replace {args.current}.",
               "KEEP": f"keep {args.current}; {args.new} fails at least one rule.",
               "BLOCKED": "rules 1 and 2 pass, but the camera test cannot be judged yet, so no swap."}
    print(f"\nVerdict: {verdict} - {explain[verdict]}")

    if camera_rows:
        n = save_mistake_sheet(camera_rows, {w: predictions[w]["camera test"] for w in predictions})
        print(f"\nCamera test mistakes: {n} photo(s) that either model got wrong"
              + (f" -> {rel(SHEET_PATH)} (do not commit it)" if n else "; no contact sheet needed."))
    else:
        remove_old_sheet()
        print("\nCamera test mistakes: no camera_test photos, so no contact sheet.")


if __name__ == "__main__":
    main()
