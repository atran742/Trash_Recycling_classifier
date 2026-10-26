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
python src/data/crop_taco.py
python src/data/build_splits.py
python src/data/check_leakage.py
```

This downloads the Kaggle images and the two wild datasets (TACO litter photos, RealWaste landfill photos), cuts the TACO objects into single images, removes duplicates, builds `data/train`, `data/validation`, `data/test` (the frozen benchmark) and `data/wild_test`, and checks that no near-copy of a training image sits in validation or either test set. Images never go into git. After training, pick the "I'm not sure" cutoff on validation and report on test with `python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv` (the two `docs/sample_*_predictions.csv` files are fake examples to try it on).

## The data right now

From [docs/dataset_inventory.md](docs/dataset_inventory.md), 4 classes, Kaggle plus TACO and RealWaste, split 70 / 15 / 15 with seed 42. Test is the frozen Kaggle benchmark; wild test holds only TACO and RealWaste images.

| Class | Train | Validation | Test (benchmark) | Wild test | Total |
|---|---:|---:|---:|---:|---:|
| compost | 1,009 | 214 | 153 | 60 | 1,436 |
| hazardous | 681 | 146 | 145 | 0 | 972 |
| recycling | 4,339 | 925 | 864 | 60 | 6,188 |
| trash | 1,341 | 289 | 108 | 178 | 1,916 |
| **Total** | **7,370** | **1,574** | **1,270** | **298** | **10,512** |

Recycling is 59% of the training data. A "model" that always answers recycling gets 59% validation accuracy but only **0.19 macro F1**, so we report macro F1, not accuracy. Class weights for training are in `results/class_weights.json`. Every build prints a stamp (`Split version` and `manifest_sha256`); two runs only compare fairly if the stamps match.

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
- `src/eval/evaluate.py` scores predictions, draws a confusion matrix and picks a confidence cutoff. `src/eval/score_camera_test.py` scores the held-out demo-room photos.
- `data/raw/team_photos/<class>/` is where our own phone photos go (they join the split). `data/raw/camera_test/<class>/` is where demo-room webcam or Pi photos go (never trained on). Each folder has a README with the rules.
- `src/webcam_smooth.py` is the demo loop: averages the last 8 predictions, classifies only a center box, says "I'm not sure" below a 0.55 cutoff and shows the San Diego rule for the class. `src/webcam_test.py` is Amy's plain version.
- `raspberry_pi/` has the camera test and a first-time Pi guide.
- `trash_recycling_classifier_project_plan.md` is the original plan.
