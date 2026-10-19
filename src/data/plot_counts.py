"""
plot_counts.py - draw two bar charts of the class counts for sharing (e.g. in Discord).

Reads the split manifests that build_splits.py writes (data/splits/train.csv,
validation.csv, test.csv), so the charts always match the real splits.

Writes (git-ignored, like every .png):
    results/class_counts.png   total images per class
    results/split_counts.png   train / validation / test images per class

Run from the repo root, after build_splits.py:
    python src/data/plot_counts.py
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_NAMES = ["train", "validation", "test"]

# Big text so the charts stay readable on a phone screen.
plt.rcParams.update({"font.size": 16, "axes.titlesize": 22, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 15, "legend.fontsize": 15})

# One fixed color per class, and one per split.
CLASS_COLORS = {"compost": "#4c9a2a", "hazardous": "#d1495b", "recycling": "#2e86de", "trash": "#6c757d"}
SPLIT_COLORS = {"train": "#2e86de", "validation": "#f4a259", "test": "#8e7dbe"}


def read_counts():
    """Return {split: Counter(class -> number of images)} from the manifest CSVs."""
    counts = {}
    for split in SPLIT_NAMES:
        path = MANIFEST_DIR / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} not found. Run src/data/build_splits.py first.")
        with open(path) as f:
            counts[split] = Counter(row["label"] for row in csv.DictReader(f))
    return counts


def plot_class_counts(counts, classes):
    """Bar chart: total images per class, with the number printed on each bar."""
    totals = [sum(counts[s][c] for s in SPLIT_NAMES) for c in classes]
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(classes, totals, color=[CLASS_COLORS.get(c, "#999999") for c in classes])
    ax.bar_label(bars, labels=[f"{n:,}" for n in totals], padding=4, fontsize=17, fontweight="bold")
    ax.set_title(f"Images per class ({sum(totals):,} total)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(totals) * 1.12)  # headroom so the top label fits
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "class_counts.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out, totals


def plot_split_counts(counts, classes):
    """Grouped bars: train / validation / test images for each class."""
    fig, ax = plt.subplots(figsize=(11, 7))
    width = 0.27
    for k, split in enumerate(SPLIT_NAMES):
        values = [counts[split][c] for c in classes]
        # Shift each split's bars left/center/right of the class position.
        positions = [x + (k - 1) * width for x in range(len(classes))]
        bars = ax.bar(positions, values, width, label=split, color=SPLIT_COLORS[split])
        ax.bar_label(bars, labels=[f"{n:,}" for n in values], padding=3, fontsize=14)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_title("Train / validation / test images per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts["train"].values()) * 1.12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "split_counts.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    counts = read_counts()
    classes = sorted(set().union(*counts.values()))  # alphabetical, same as the docs tables
    RESULTS_DIR.mkdir(exist_ok=True)

    class_png, totals = plot_class_counts(counts, classes)
    split_png = plot_split_counts(counts, classes)

    # Print the numbers the charts show, so they can be checked against the docs.
    print(f"{'class':<12}{'train':>8}{'validation':>12}{'test':>8}{'total':>8}")
    for c, total in zip(classes, totals):
        print(f"{c:<12}{counts['train'][c]:>8}{counts['validation'][c]:>12}{counts['test'][c]:>8}{total:>8}")
    print(f"{'TOTAL':<12}{sum(counts['train'].values()):>8}{sum(counts['validation'].values()):>12}"
          f"{sum(counts['test'].values()):>8}{sum(totals):>8}")
    print(f"\nSaved {class_png.relative_to(REPO_ROOT)} and {split_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
