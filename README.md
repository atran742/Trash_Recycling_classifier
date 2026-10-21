# Trash / Recycling Classifier

A Raspberry Pi 5 with a camera looks at one item and says where it goes under San Diego rules: **recycling**, **compost**, **trash** or **hazardous** (batteries). When the model is not confident enough, it says **"I'm not sure"** instead of guessing. SDSU AI Club project, final demo due early December 2026.

## Team and roles

| Role | Who |
|---|---|
| Data pipeline and evaluation | Isiah |
| Model training | Amy |
| Raspberry Pi setup and camera | **open** |
| Demo app (screen output, "I'm not sure" rules) | **open** |

## Quickstart

Needs Python 3.10+ and a Kaggle token in `~/.kaggle/` (see [docs/data_pipeline.md](docs/data_pipeline.md)).

```bash
pip install -r requirements.txt
python src/data/download.py
python src/data/build_splits.py
python src/data/check_leakage.py
```

This downloads the Kaggle images, removes duplicates, builds `data/train`, `data/validation` and `data/test`, and checks that no near-copy of a training image sits in validation or test. Images never go into git. After training, pick the "I'm not sure" cutoff on validation and report on test with `python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv` (the two `docs/sample_*_predictions.csv` files are fake examples to try it on).

## The data right now

From [docs/dataset_inventory.md](docs/dataset_inventory.md), 4 classes, split 70 / 15 / 15 with seed 42:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| compost | 718 | 154 | 153 | 1,025 |
| hazardous | 681 | 146 | 145 | 972 |
| recycling | 4,059 | 865 | 864 | 5,788 |
| trash | 503 | 106 | 108 | 717 |
| **Total** | **5,961** | **1,271** | **1,270** | **8,502** |

Recycling is 68% of the training data. A "model" that always answers recycling gets 68% accuracy but only **0.20 macro F1** on validation, so we report macro F1, not accuracy. Class weights for training are in `results/class_weights.json`. Every build prints a stamp (`Split version` and `manifest_sha256`); two runs only compare fairly if the stamps match.

## Status by plan week

| Plan week | Work | Status |
|---|---|---|
| 1 to 3 | Rules, datasets, cleaning, splits | **Done** (data pipeline merged; cleanup in PR #2) |
| 3 to 5 | Baseline model, improvement, evaluation | **In progress** (Amy) |
| 6 | Confidence threshold and "I'm not sure" | Evaluation script ready (`src/eval/evaluate.py`) |
| 7 to 8 | Raspberry Pi deployment and camera | **Not started** (camera smoke test in `raspberry_pi/`) |
| 9 | Final demo and report | Not started |

## Where things are

- `src/data/` downloads, cleans and splits the data. `docs/data_pipeline.md` explains every step and every error message.
- `docs/dataset_inventory.md` lists each dataset, its license, the real counts and what was removed.
- `src/eval/evaluate.py` scores predictions, draws a confusion matrix and picks a confidence cutoff.
- `raspberry_pi/` has the camera test and a first-time Pi guide.
- `trash_recycling_classifier_project_plan.md` is the original plan.
