# Evaluation: from model file to confusion matrix and threshold chart

Two commands take a trained model to the final numbers. The first writes a
predictions CSV per split, the second scores them and picks the "I'm not sure"
cutoff the right way: tuned on validation, reported on test.

## Before you start

- The splits are built: `python src/data/build_splits.py` ends with
  `Split version: def16b3  manifest_sha256: 08d058850b31`. Both commands below
  read `data/splits/validation.csv`, `data/splits/test.csv` and
  `data/splits/wild_test.csv`, so they score exactly that split.
- `pip install -r requirements.txt` plus PyTorch and torchvision for your
  computer (see https://pytorch.org/get-started/locally/).
- A model file. Three forms work: a TorchScript file, a whole model saved with
  `torch.save(model, path)`, or a state_dict saved with
  `torch.save(model.state_dict(), path)`. A state_dict also needs `--arch`.

## The 2 commands

```bash
python src/eval/export_predictions.py --model models/waste_classifier.pth --arch mobilenet_v3_small
```

```bash
python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv --wild results/wild_test_predictions.csv
```

| Command | What it does | Writes |
|---|---|---|
| `export_predictions.py` | Runs the model over every validation, benchmark test and wild test image and records the true label, the predicted label and the model's confidence. | `results/val_predictions.csv`, `results/test_predictions.csv`, `results/wild_test_predictions.csv` |
| `evaluate.py` | Prints accuracy, macro F1 and per-class scores, sweeps cutoffs on the validation file, picks the lowest cutoff reaching 90% accuracy on answered items (`--target` to change), and applies it once to the benchmark test file and, with `--wild`, to the wild test file. `--balanced` adds scores with equal counts per class (see below). Ends with one summary line: benchmark macro F1 and wild test macro F1. | `results/confusion_matrix.png`, `results/confusion_matrix_wild.png`, `results/threshold_sweep.csv`, `results/threshold_sweep.png` |

## Two test sets: the benchmark and the wild test

| Test set | File | What is in it | Changes? |
|---|---|---|---|
| **Benchmark** | `data/splits/test.csv` | The 1,270 Kaggle test images of Oct 24, 2026 | **Never.** `build_splits.py` stops if a single byte changes (`benchmark:` in `configs/label_map.yaml`). |
| **Wild test** | `data/splits/wild_test.csv` | 15% of the new real-world sources (TACO crops, RealWaste, and later team or Amy photos), split on their own | Grows when new photos are added; its share of each source never touches the benchmark. |

The benchmark keeps every score comparable with every earlier one. The wild
test asks the harder question: how does the model do on messy photos that look
more like the demo? It has no hazardous items (hazardous was left alone), so its
macro F1 averages recycling, compost and trash. A wrong "hazardous" guess still
counts against the item's true class.

**Compost in the wild set comes only from RealWaste, and trash and recycling only from TACO, so judge any retrain on camera_test too.** A model can score well on the wild test by learning each source's look (RealWaste's concrete floor, TACO's ground litter) instead of the item, and neither looks like the demo room. `src/eval/score_camera_test.py` on the demo-room photos in `data/raw/camera_test/` is the check that cannot be passed that way.

## The swap rule: when a new model replaces the current one

Score both models with the commands above on the same split (same stamp line).
The new model replaces `models/waste_classifier.pth` only if **both** hold:

1. **Benchmark macro F1 drops by no more than 0.01.** New benchmark macro F1 >= current benchmark macro F1 - 0.01.
2. **Wild test macro F1 goes up.** New wild test macro F1 > current wild test macro F1.

Baseline to beat (Amy's first model, `models/waste_classifier.pth`, on the build stamped
`manifest_sha256: 08d058850b31`): **benchmark macro F1 0.785, wild test macro F1 0.289**
(recall: compost 0.00, recycling 0.23, trash 0.58; it calls every RealWaste food photo trash).
The Oct 24 build had a smaller wild test (209 images, macro F1 0.270); the two numbers are not comparable.

Example: a new model with 0.778 on the benchmark and 0.45 on the wild test replaces it
(0.778 >= 0.775, and 0.45 > 0.289). A new model with 0.770 and 0.60 does not
(0.770 < 0.775), however much better it is on the wild test. The rule protects what
already works while the wild data is added.

Note: validation now also holds wild images, so the "I'm not sure" cutoff is tuned on
both kinds. For Amy's first model, validation accuracy is 0.746 and no cutoff reaches the
default 90% target; a model trained with the wild images in train should do better.

The CSVs and charts are git-ignored. Share them in the PR or Discord.

## Settings, matched to `train.py`

| Setting | Value | Where |
|---|---|---|
| Framework | PyTorch + torchvision | |
| Network | `mobilenet_v3_small` | `--arch` (needed for a state_dict file such as `models/waste_classifier.pth`) |
| Preprocessing | `MobileNet_V3_Small_Weights.DEFAULT.transforms()`: resize the short side to 256, center-crop 224 x 224, ImageNet mean and std. Exactly what `train.py` evaluates with. | `preprocessing_for()` in `export_predictions.py` |
| Class order | the `classes` list stored in the checkpoint (`train.py` saves one), matched to the label map ignoring capital letters, so `Compost` = `compost`; without a stored list, the label-map classes alphabetical | a name that matches no label-map class stops the run and prints both lists |

The preprocessing matters: a plain resize to 224 x 224 (no crop) scored Amy's
model at 0.757 test macro F1; the matching resize + center crop scores it at
0.785. Do not change it without retraining.

## `--balanced`: model quality apart from the class mix

The test set is 68% recycling and 8.5% trash, so a few percent of recycling
images guessed as trash outnumber the whole trash class and pull trash
precision, and so macro F1, down. `--balanced` scores the test file as if the
classes were equal: it draws the smallest class's count (108) from every class,
scores that set, and repeats 1000 times with seed 42, printing the mean and the
middle 95% of the draws. On Amy's first model this gives 0.846 macro F1
(interval 0.827 to 0.864) against 0.785 on the real mix. The balanced number
says how good the model is per class; the real-mix number is what the test set
scores. Report both.

## The camera test set

`data/raw/camera_test/<class>/` holds demo-room photos from the webcam or Pi
camera that are never split or trained on. After the two commands above:

```bash
python src/eval/score_camera_test.py --model models/waste_classifier.pth --arch mobilenet_v3_small
```

It warns about any photo that is a near copy of a train image (same pHash /
dHash rules as the pipeline), writes `results/camera_test_predictions.csv` and
runs `evaluate.py` with `results/val_predictions.csv` as the tune file, so the
cutoff still comes from validation. This is the number closest to the demo.

## Reading the result

- **Macro F1 on the benchmark test file** is the headline number. Accuracy is
  inflated by the recycling class (59% of train, 68% of the benchmark). **Wild test macro F1** is the
  second number; together they decide the swap rule above.
- **The chosen cutoff** comes from validation only. The test numbers "with
  cutoff applied" are what the Pi demo would show: how many items get an
  answer, how accurate those answers are, and how many wrong answers became
  "I'm not sure".
- **Passing one file** to `evaluate.py` still works but prints a warning: a
  cutoff tuned and reported on the same items looks better than it is.
