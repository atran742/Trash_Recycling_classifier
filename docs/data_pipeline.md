# Data Pipeline: zero to train / validation / test

This takes a teammate from a fresh clone to `data/train`, `data/validation` and
`data/test` folders that are ready for training. None of these images go into git.

## One-time setup

1. **Python 3.10 or newer** and this repo cloned.
2. **A virtual environment**, so packages stay inside the project:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. **A Kaggle token.** Sign in at kaggle.com, open Settings, then API, and click
   "Create New Token". Save it as `~/.kaggle/kaggle.json` (file download) or
   `~/.kaggle/access_token` (token string), then run `chmod 600` on that file.

## The 3 commands

Run these from the repo root with the virtual environment active.

```bash
pip install -r requirements.txt
python src/data/download.py
python src/data/build_splits.py
```

| Command | What it does | Time |
|---|---|---|
| `pip install -r requirements.txt` | Installs kaggle, pyyaml, pillow, imagehash and numpy. | under 1 minute |
| `python src/data/download.py` | Downloads `garbage12_enhanced` (about 557 MB unzipped) into `data/raw/`. Skips it if it is already there. Also fetches `garbage_v2` only if there are fewer than 500 battery images. | a few minutes |
| `python src/data/build_splits.py` | Applies `configs/label_map.yaml`, drops broken files, exact duplicates and near duplicates, copies a 70/15/15 split into `data/train`, `data/validation` and `data/test`, and writes the CSV manifests and class weights. | about 20 seconds |

## What you should see

With `num_classes: 4`, the last command prints:

```text
Removed 2 exact duplicates, 125 near duplicates, and 48 images in label conflicts (10 near-duplicate groups). Skipped 0 corrupted files. 8403 images remain.

class          train  validation    test   total
------------------------------------------------
compost          715         153     154    1022
hazardous        674         144     146     964
recycling       4009         859     860    5728
trash            482         103     104     689
------------------------------------------------
TOTAL           5880        1259    1264    8403

Train split share and class weights (results/class_weights.json):
  compost        715 images   12.16% of train   weight 2.0559
  hazardous      674 images   11.46% of train   weight 2.1810
  recycling     4009 images   68.18% of train   weight 0.3667
  trash          482 images    8.20% of train   weight 3.0498

Overlap check passed: no exact or near duplicate appears in more than one split.
```

The result is the same on every machine because the seed is fixed at 42.

## What else the split step writes

| File | In git? | What it is |
|---|---|---|
| `data/train/`, `data/validation/`, `data/test/` | No | The image folders, one sub-folder per class. Works with `torchvision.datasets.ImageFolder`. |
| `data/splits/train.csv`, `validation.csv`, `test.csv` | Yes | Manifests with columns `path,label`. Paths are relative to the repo root and point at the copied images. They are small and hold no images, so everyone trains on the exact same split. |
| `results/class_weights.json` | Yes | Class order, train counts, percent of train per class, and inverse-frequency weights computed from the train split only. |
| `results/removed_duplicates.csv` | Yes | Every dropped file with `file,kept_copy,reason,distance`. Reasons are `exact_duplicate`, `near_duplicate`, `label_conflict (...)` or `corrupted`. |

### Using the class weights in PyTorch

`class_order` is alphabetical, the same order `ImageFolder` uses for its labels:

```python
import json, torch
w = json.load(open("results/class_weights.json"))
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w["weights_in_class_order"]))
```

## Near-duplicate cutoff

`near_duplicate_max_distance` in `configs/label_map.yaml` sets how close two
perceptual hashes must be to count as the same picture (5 by default). At 4 or
5 it also catches different objects with a similar shape on a white background.
See the duplicate section in `docs/dataset_inventory.md` before changing it.

## Switching between 3 and 4 classes

Edit one line at the top of `configs/label_map.yaml`, then rerun `build_splits.py`:

```yaml
num_classes: 3   # or 4
```

With 3 classes, every folder mapped to `hazardous` is left out. For a quick test
without editing the file, run `python src/data/build_splits.py --num-classes 3`.

## Adding personal battery photos

Put photos straight into `data/raw/personal_battery/` and follow the naming rules
in that folder's `README.md`. Rerun `build_splits.py`, and they join the
`hazardous` class automatically.

## Troubleshooting

- **"No Kaggle token found"**: redo setup step 3. On Windows the folder is `C:\Users\<you>\.kaggle\`.
- **`CERTIFICATE_VERIFY_FAILED` on macOS**: the python.org installer ships without
  root certificates. Run `/Applications/Python\ 3.*/Install\ Certificates.command` once.
- **"label_map.yaml key ... has no folder"**: a folder name in the yaml does not
  match `data/raw/`. Run `download.py` again to print the real folder names.
