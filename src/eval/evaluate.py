"""
evaluate.py - score a model's predictions and pick the "I'm not sure" cutoff the right way.

Two files, two jobs, never mixed:
    --tune   validation predictions: the confidence sweep runs here and picks the cutoff.
    --report test predictions:       the chosen cutoff is applied here ONCE and the final
                                     numbers come from here.
Picking the cutoff on the same file you report on inflates the score, because the
cutoff was chosen to look good on exactly those items. If only one file is given,
the script still runs but prints a warning saying so.

Each CSV has one row per image and these columns:
    path, true_label, predicted_label, confidence
confidence is the model's score for its predicted label, between 0 and 1.
docs/sample_validation_predictions.csv and docs/sample_test_predictions.csv are
fake examples (no real images) to try it on.

What it prints:
  1. Tune file: accuracy, macro F1, and the confidence sweep (cutoffs 0.50 to 0.95):
     how many items get an answer (confidence >= cutoff), how accurate those
     answers are, and how many wrong answers become "I'm not sure".
  2. The lowest cutoff whose accuracy on answered items reaches --target (default 0.90).
  3. Report file: accuracy, macro F1 and precision / recall / F1 per class on all
     items, the "always say recycling" baseline for comparison, and then the
     numbers WITH the cutoff applied: answered share, accuracy and macro F1 on the
     answered items, wrong answers left, wrong answers avoided.

What it saves (git-ignored, never committed):
    results/threshold_sweep.csv / .png   the sweep on the tune file
    results/confusion_matrix.png         the report file, all items

Run from the repo root:
    python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv
    python src/eval/evaluate.py --tune docs/sample_validation_predictions.csv --report docs/sample_test_predictions.csv
    python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv --target 0.85
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_LABEL = "recycling"  # the biggest class; see docs/dataset_inventory.md
CUTOFFS = [round(0.50 + 0.05 * k, 2) for k in range(10)]  # 0.50, 0.55, ... 0.95


def read_predictions(csv_path):
    """Read the CSV into three lists: true labels, predicted labels, confidences."""
    needed = ["path", "true_label", "predicted_label", "confidence"]
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} is missing column(s): {', '.join(missing)}. Needed: {', '.join(needed)}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{csv_path} has no rows.")
    true = [r["true_label"].strip() for r in rows]
    pred = [r["predicted_label"].strip() for r in rows]
    conf = []
    for r in rows:
        try:
            c = float(r["confidence"])
        except ValueError:
            raise SystemExit(f"Bad confidence {r['confidence']!r} for {r['path']}")
        if not 0 <= c <= 1:
            raise SystemExit(f"Confidence {c} for {r['path']} is not between 0 and 1")
        conf.append(c)
    return true, pred, np.array(conf)


def macro_f1(true, pred, classes):
    """Average of the per-class F1 scores, every class counting the same.

    zero_division=0: a class that is never predicted gets precision 0 instead of an error.
    """
    return f1_score(true, pred, labels=classes, average="macro", zero_division=0)


def print_scores(title, true, pred, classes, per_class=True):
    """Print accuracy, macro F1 and (optionally) a per-class table."""
    acc = accuracy_score(true, pred)
    print(f"\n{title}")
    print(f"  accuracy {acc:.4f}   macro F1 {macro_f1(true, pred, classes):.4f}   ({len(true)} items)")
    if per_class:
        prec, rec, f1, support = precision_recall_fscore_support(true, pred, labels=classes, zero_division=0)
        print(f"  {'class':<12}{'precision':>10}{'recall':>8}{'F1':>8}{'items':>7}")
        for c, p, r, s, n in zip(classes, prec, rec, f1, support):
            print(f"  {c:<12}{p:>10.3f}{r:>8.3f}{s:>8.3f}{n:>7}")


def save_confusion_matrix(true, pred, classes, out_path):
    """Rows = true class, columns = predicted class, with the count in every cell."""
    cm = confusion_matrix(true, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (report file, all items)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            # White text on dark cells, black on light ones.
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def at_cutoff(true, pred, conf, cutoff, classes):
    """What happens if the system only answers when confidence >= cutoff.

    Returns a dict:
      answered           how many items get an answer
      share_answered     answered / all items
      accuracy_answered  accuracy on the answered items (None if none answered)
      macro_f1_answered  macro F1 on the answered items (None if none answered)
      wrong_avoided      wrong predictions that became "I'm not sure" instead
      wrong_remaining    wrong predictions still given as answers
    """
    answered = conf >= cutoff
    correct = np.array([t == p for t, p in zip(true, pred)])
    n = int(answered.sum())
    true_a = [t for t, a in zip(true, answered) if a]
    pred_a = [p for p, a in zip(pred, answered) if a]
    return {
        "answered": n,
        "share_answered": n / len(conf),
        "accuracy_answered": float(correct[answered].mean()) if n else None,
        "macro_f1_answered": macro_f1(true_a, pred_a, classes) if n else None,
        "wrong_avoided": int((~correct & ~answered).sum()),
        "wrong_remaining": int((~correct & answered).sum()),
    }


def confidence_sweep(true, pred, conf, classes):
    """at_cutoff() for every cutoff in CUTOFFS, in rising order."""
    rows = []
    for cutoff in CUTOFFS:
        r = at_cutoff(true, pred, conf, cutoff, classes)
        r["cutoff"] = cutoff
        rows.append(r)
    return rows


def save_sweep(rows, csv_path, png_path):
    """Write the sweep table and draw share-answered and accuracy-on-answered against the cutoff."""
    columns = ["cutoff", "answered", "share_answered", "accuracy_answered", "wrong_avoided", "wrong_remaining"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r["cutoff"], r["answered"], round(r["share_answered"], 4),
                             "" if r["accuracy_answered"] is None else round(r["accuracy_answered"], 4),
                             r["wrong_avoided"], r["wrong_remaining"]])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cutoffs = [r["cutoff"] for r in rows]
    ax.plot(cutoffs, [r["share_answered"] for r in rows], marker="o", label="share of items answered")
    ax.plot(cutoffs, [np.nan if r["accuracy_answered"] is None else r["accuracy_answered"] for r in rows],
            marker="s", label="accuracy on answered items")
    ax.set_xlabel("Confidence cutoff (below it the system says \"I'm not sure\")")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Confidence sweep (tune file)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def print_sweep(rows, n_items):
    """The sweep as a small table."""
    print(f"\nConfidence sweep on the tune file ({n_items} items; a wrong prediction below the cutoff is 'avoided'):")
    print(f"  {'cutoff':>6}{'answered':>10}{'share':>8}{'accuracy':>10}{'wrong avoided':>15}{'wrong left':>12}")
    for r in rows:
        acc = "  -  " if r["accuracy_answered"] is None else f"{r['accuracy_answered']:.3f}"
        print(f"  {r['cutoff']:>6.2f}{r['answered']:>10}{r['share_answered']:>8.2f}{acc:>10}{r['wrong_avoided']:>15}{r['wrong_remaining']:>12}")


def main():
    parser = argparse.ArgumentParser(description="Pick the confidence cutoff on one file, report on another.")
    parser.add_argument("--tune", type=Path, help="validation predictions CSV: the cutoff is chosen here")
    parser.add_argument("--report", type=Path, help="test predictions CSV: the cutoff is applied here once")
    parser.add_argument("csv", nargs="?", type=Path, help="(old form) one CSV used for both; prints a warning")
    parser.add_argument("--target", type=float, default=0.90,
                        help="accuracy on answered items the cutoff must reach (default 0.90)")
    args = parser.parse_args()
    if not 0 < args.target <= 1:
        parser.error("--target must be between 0 and 1, for example 0.90")
    if args.csv and (args.tune or args.report):
        parser.error("give either one CSV, or --tune and --report, not both")

    # Work out which file does which job. One file means it does both, with a warning.
    tune_file = args.tune or args.csv or args.report
    report_file = args.report or args.csv or args.tune
    if tune_file is None:
        parser.error("give --tune validation.csv and --report test.csv")
    same_file = tune_file.resolve() == report_file.resolve()
    if same_file:
        print("WARNING: the cutoff is tuned and reported on the SAME file, so on the same items.\n"
              "         That inflates the score: the cutoff was chosen to look good on exactly these items.\n"
              "         For honest numbers run:  --tune validation_predictions.csv --report test_predictions.csv\n")

    # ---- 1. Tune file: sweep and pick the cutoff. The report file is not read yet. ----
    t_true, t_pred, t_conf = read_predictions(tune_file)
    classes = sorted(set(t_true) | set(t_pred))  # alphabetical, like the split folders
    print(f"TUNE file:   {tune_file} ({len(t_true)} items, classes {classes})")
    print_scores("Tune file, all items", t_true, t_pred, classes, per_class=False)
    rows = confidence_sweep(t_true, t_pred, t_conf, classes)
    print_sweep(rows, len(t_true))
    RESULTS_DIR.mkdir(exist_ok=True)
    sweep_csv, sweep_png = RESULTS_DIR / "threshold_sweep.csv", RESULTS_DIR / "threshold_sweep.png"
    save_sweep(rows, sweep_csv, sweep_png)

    good = [r for r in rows if r["accuracy_answered"] is not None and r["accuracy_answered"] >= args.target]
    cutoff = None  # decided here, from the tune file only; never changed after this point
    if good:
        chosen = good[0]  # rows are in rising cutoff order, so this is the lowest cutoff that works
        cutoff = chosen["cutoff"]
        print(f"\nChosen cutoff: {cutoff:.2f} (lowest cutoff reaching {args.target:.0%} accuracy on answered tune items: "
              f"answers {chosen['share_answered']:.0%} at {chosen['accuracy_answered']:.1%}).")
    else:
        print(f"\nNo cutoff between {CUTOFFS[0]:.2f} and {CUTOFFS[-1]:.2f} reaches {args.target:.0%} accuracy on "
              f"answered items of the tune file. The model needs to improve, or lower --target.")

    # ---- 2. Report file: read it only now, apply the cutoff once. ----
    r_true, r_pred, r_conf = read_predictions(report_file)
    classes = sorted(set(classes) | set(r_true) | set(r_pred))
    print(f"\nREPORT file: {report_file} ({len(r_true)} items)")
    print_scores("Report file, all items (no cutoff)", r_true, r_pred, classes)
    print_scores(f"Baseline on the report file: always answer '{BASELINE_LABEL}'",
                 r_true, [BASELINE_LABEL] * len(r_true), classes, per_class=False)
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(r_true, r_pred, classes, cm_path)

    if cutoff is None:
        print("\nNo cutoff was chosen, so the report file is shown without one.")
        print(f"\nSaved {sweep_csv.relative_to(REPO_ROOT)}, {sweep_png.relative_to(REPO_ROOT)}, {cm_path.relative_to(REPO_ROOT)}")
        return
    res = at_cutoff(r_true, r_pred, r_conf, cutoff, classes)
    print(f"\nReport file with cutoff {cutoff:.2f} applied (below it the system says \"I'm not sure\"):")
    print(f"  answered           {res['answered']} of {len(r_true)} items ({res['share_answered']:.1%})")
    if res["answered"]:
        print(f"  accuracy answered  {res['accuracy_answered']:.4f}")
        print(f"  macro F1 answered  {res['macro_f1_answered']:.4f}")
    else:
        print("  nothing answered at this cutoff")
    print(f"  wrong answers left {res['wrong_remaining']}")
    print(f"  wrong avoided      {res['wrong_avoided']}  (now \"I'm not sure\")")
    if same_file:
        print("\n  (tuned and reported on the same file: treat these numbers as an upper bound)")

    print(f"\nSaved {sweep_csv.relative_to(REPO_ROOT)}, {sweep_png.relative_to(REPO_ROOT)}, {cm_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
