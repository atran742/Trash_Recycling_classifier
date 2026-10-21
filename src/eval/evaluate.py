"""
evaluate.py - score a model's predictions and pick a confidence cutoff.

Input: a CSV with one row per image and these columns:
    path, true_label, predicted_label, confidence
confidence is the model's score for its predicted label, between 0 and 1.
docs/sample_predictions.csv is a fake example (no real images) to try it on.

What it prints:
  1. Accuracy and macro F1, plus precision / recall / F1 for every class.
  2. The same numbers for an "always say recycling" baseline, for comparison.
  3. A confidence sweep: for each cutoff from 0.50 to 0.95, how many items
     the system would answer (confidence >= cutoff), how accurate those
     answers are, and how many wrong answers turning into "I'm not sure" avoids.
  4. The lowest cutoff whose answered-items accuracy reaches the target (--target, default 0.90).

What it saves (git-ignored, never committed):
    results/confusion_matrix.png   which classes get mixed up
    results/threshold_sweep.csv    the sweep table
    results/threshold_sweep.png    the sweep as a chart

Run from the repo root:
    python src/eval/evaluate.py docs/sample_predictions.csv
    python src/eval/evaluate.py results/my_predictions.csv --target 0.85
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


def print_scores(title, true, pred, classes):
    """Print accuracy, macro F1 and a per-class table. Returns (accuracy, macro F1)."""
    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, labels=classes, average="macro", zero_division=0)
    # zero_division=0: a class that is never predicted gets precision 0 instead of an error.
    prec, rec, f1, support = precision_recall_fscore_support(true, pred, labels=classes, zero_division=0)
    print(f"\n{title}")
    print(f"  accuracy {acc:.4f}   macro F1 {macro_f1:.4f}   ({len(true)} items)")
    print(f"  {'class':<12}{'precision':>10}{'recall':>8}{'F1':>8}{'items':>7}")
    for c, p, r, s, n in zip(classes, prec, rec, f1, support):
        print(f"  {c:<12}{p:>10.3f}{r:>8.3f}{s:>8.3f}{n:>7}")
    return acc, macro_f1


def save_confusion_matrix(true, pred, classes, out_path):
    """Rows = true class, columns = predicted class, with the count in every cell."""
    cm = confusion_matrix(true, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            # White text on dark cells, black on light ones.
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confidence_sweep(true, pred, conf):
    """For each cutoff: answer only items with confidence >= cutoff, the rest become "I'm not sure".

    Returns a list of dicts, one per cutoff:
      answered           how many items get an answer
      share_answered     answered / all items
      accuracy_answered  accuracy on the answered items (blank if none answered)
      wrong_avoided      wrong predictions that became "I'm not sure" instead
      wrong_remaining    wrong predictions still given as answers
    """
    correct = np.array([t == p for t, p in zip(true, pred)])
    rows = []
    for cutoff in CUTOFFS:
        answered = conf >= cutoff
        n_answered = int(answered.sum())
        rows.append({
            "cutoff": cutoff,
            "answered": n_answered,
            "share_answered": round(n_answered / len(conf), 4),
            "accuracy_answered": round(float(correct[answered].mean()), 4) if n_answered else "",
            "wrong_avoided": int((~correct & ~answered).sum()),
            "wrong_remaining": int((~correct & answered).sum()),
        })
    return rows


def save_sweep(rows, csv_path, png_path):
    """Write the sweep table and draw share-answered and accuracy-on-answered against the cutoff."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cutoffs = [r["cutoff"] for r in rows]
    ax.plot(cutoffs, [r["share_answered"] for r in rows], marker="o", label="share of items answered")
    ax.plot(cutoffs, [r["accuracy_answered"] if r["accuracy_answered"] != "" else np.nan for r in rows],
            marker="s", label="accuracy on answered items")
    ax.set_xlabel("Confidence cutoff (below it the system says \"I'm not sure\")")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Confidence sweep")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Score predictions and pick a confidence cutoff.")
    parser.add_argument("csv", type=Path, help="predictions CSV: path, true_label, predicted_label, confidence")
    parser.add_argument("--target", type=float, default=0.90,
                        help="accuracy on answered items the cutoff must reach (default 0.90)")
    args = parser.parse_args()

    true, pred, conf = read_predictions(args.csv)
    classes = sorted(set(true) | set(pred))  # alphabetical, like the split folders
    print(f"Read {len(true)} predictions from {args.csv} with classes {classes}")

    print_scores("Model", true, pred, classes)
    print_scores(f"Baseline: always answer '{BASELINE_LABEL}'", true, [BASELINE_LABEL] * len(true), classes)

    RESULTS_DIR.mkdir(exist_ok=True)
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(true, pred, classes, cm_path)

    rows = confidence_sweep(true, pred, conf)
    print(f"\nConfidence sweep ({len(true)} items; a wrong prediction below the cutoff is 'avoided'):")
    print(f"  {'cutoff':>6}{'answered':>10}{'share':>8}{'accuracy':>10}{'wrong avoided':>15}{'wrong left':>12}")
    for r in rows:
        acc = f"{r['accuracy_answered']:.3f}" if r["accuracy_answered"] != "" else "  -  "
        print(f"  {r['cutoff']:>6.2f}{r['answered']:>10}{r['share_answered']:>8.2f}{acc:>10}{r['wrong_avoided']:>15}{r['wrong_remaining']:>12}")
    sweep_csv, sweep_png = RESULTS_DIR / "threshold_sweep.csv", RESULTS_DIR / "threshold_sweep.png"
    save_sweep(rows, sweep_csv, sweep_png)

    good = [r for r in rows if r["accuracy_answered"] != "" and r["accuracy_answered"] >= args.target]
    if good:
        best = good[0]  # rows are in rising cutoff order, so this is the lowest cutoff that works
        print(f"\nRecommended cutoff: {best['cutoff']:.2f}. It answers {best['share_answered']:.0%} of items "
              f"at {best['accuracy_answered']:.1%} accuracy (target {args.target:.0%}) and turns "
              f"{best['wrong_avoided']} wrong answers into \"I'm not sure\".")
    else:
        print(f"\nNo cutoff between {CUTOFFS[0]:.2f} and {CUTOFFS[-1]:.2f} reaches {args.target:.0%} accuracy on answered items. "
              f"The model needs to improve, or lower --target.")
    print(f"\nSaved {cm_path.relative_to(REPO_ROOT)}, {sweep_csv.relative_to(REPO_ROOT)}, {sweep_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
