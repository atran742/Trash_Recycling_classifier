# Evaluation: from model file to confusion matrix and threshold chart

Two commands take a trained model to the final numbers. The first writes a
predictions CSV per split, the second scores them and picks the "I'm not sure"
cutoff the right way: tuned on validation, reported on test.

## Before you start

- The splits are built: `python src/data/build_splits.py` ends with
  `Split version: 7ee11b0  manifest_sha256: 4d578382b730`. Both commands below
  read `data/splits/validation.csv` and `data/splits/test.csv`, so they score
  exactly that split.
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
python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv
```

| Command | What it does | Writes |
|---|---|---|
| `export_predictions.py` | Runs the model over every validation and test image and records the true label, the predicted label and the model's confidence. | `results/val_predictions.csv`, `results/test_predictions.csv` |
| `evaluate.py` | Prints accuracy, macro F1 and per-class scores, sweeps cutoffs on the validation file, picks the lowest cutoff reaching 90% accuracy on answered items (`--target` to change), and applies it once to the test file. `--balanced` adds the test score with equal counts per class (see below). | `results/confusion_matrix.png`, `results/threshold_sweep.csv`, `results/threshold_sweep.png` |

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

## Reading the result

- **Macro F1 on the test file** is the headline number. Accuracy is inflated
  by the recycling class (68% of train).
- **The chosen cutoff** comes from validation only. The test numbers "with
  cutoff applied" are what the Pi demo would show: how many items get an
  answer, how accurate those answers are, and how many wrong answers became
  "I'm not sure".
- **Passing one file** to `evaluate.py` still works but prints a warning: a
  cutoff tuned and reported on the same items looks better than it is.
