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

Optional: `python src/data/plot_counts.py` draws `results/class_counts.png` and
`results/split_counts.png` (bar charts of the counts, sized for a phone screen;
not committed).

| Command | What it does | Time |
|---|---|---|
| `pip install -r requirements.txt` | Installs kaggle, pyyaml, pillow, imagehash, numpy and matplotlib. | under 1 minute |
| `python src/data/download.py` | Downloads `garbage12_enhanced` (about 557 MB unzipped) into `data/raw/`. Skips it if it is already there. Also fetches `garbage_v2` only if there are fewer than 500 battery images. | a few minutes |
| `python src/data/build_splits.py` | Applies `configs/label_map.yaml`, drops broken files, exact duplicates and near duplicates, checks the counts add up, copies a 70/15/15 split into `data/train`, `data/validation` and `data/test`, and writes the CSV manifests and class weights. | about 20 seconds |

## What you should see

With `num_classes: 4`, the last command prints:

```text
Removed 2 exact duplicates and 57 near duplicates. Held 0 images in 0 label-conflict groups for review (results/conflicts_to_review.csv). Dropped 0 corrupted files. 8519 images go into the splits.

class          train  validation    test   total
------------------------------------------------
compost          718         154     155    1027
hazardous        681         146     147     974
recycling       4059         869     871    5799
trash            503         107     109     719
------------------------------------------------
TOTAL           5961        1276    1282    8519

Train split share and class weights (results/class_weights.json):
  class         train   % train   inverse  sqrt inv
  compost         718    12.04%    2.0756    1.0983
  hazardous       681    11.42%    2.1883    1.1277
  recycling      4059    68.09%    0.3671    0.4619
  trash           503     8.44%    2.9627    1.3121

Baseline 'always guess recycling' on validation (1276 images): accuracy 68.10%, macro F1 0.2026

Overlap check passed: no exact or near duplicate appears in more than one split.
Count check: 8578 found - 0 corrupted - 2 exact duplicates - 57 near duplicates - 0 held for review = 8519; 8519 in the splits; 59 rows in results/removed_duplicates.csv; 0 rows in results/conflicts_to_review.csv.
Count check passed.
Split version: 08b8dde  manifest_sha256: 003abb987adb  (both saved in results/class_weights.json)
```

The result is the same on every machine because the seed is fixed at 42.

**Check `split_version` and `manifest_sha256` before comparing models.** Both are
printed at the end of `build_splits.py` and saved in `results/class_weights.json`.
`split_version` is the last commit that changed `src/data/build_splits.py` or
`configs/`, so docs, results and helper-script commits do not change it. `manifest_sha256` fingerprints
the split lists themselves, so it also changes when images are added to
`data/raw/`. Two models only compare fairly if both match. Splits from before the
data-cleanup branch (8,403 images, older duplicate rule) are stale: rebuild them
before training.

**After any rebuild, run `python src/data/check_leakage.py`.** It is read-only. A
clean result is a count table with 0 true copies when you look at
`results/leakage_review.png`. Today it flags 32 pairs, of which 17 are true
copies (0.66% of validation plus test, under the 1% line, so no change), and
those 17 are listed in `docs/dataset_inventory.md`. If a rebuild flags more,
compare with that list.

## What else the split step writes

| File | In git? | What it is |
|---|---|---|
| `data/train/`, `data/validation/`, `data/test/` | No | The image folders, one sub-folder per class. Works with `torchvision.datasets.ImageFolder`. |
| `data/splits/train.csv`, `validation.csv`, `test.csv` | Yes | Manifests with columns `path,label`. Paths are relative to the repo root and point at the copied images. They are small and hold no images, so everyone trains on the exact same split. |
| `results/class_weights.json` | Yes | `split_version` (the last commit that changed `build_splits.py` or `configs/`), `manifest_sha256` (fingerprint of the split CSVs), class order, train counts, percent of train per class, and two weight sets computed from the train split only: inverse frequency and square-root inverse frequency. |
| `results/removed_duplicates.csv` | Yes | Every dropped file with `file,kept_copy,reason,distance,dhash_distance`. `distance` is the pHash distance. Reasons are `exact_duplicate`, `near_duplicate` or `corrupted`. |
| `results/conflicts_to_review.csv` | Yes | Label conflicts: look-alike images with different labels, as `group,file,label,matches,distance,dhash_distance`. These are **not deleted** but kept out of every split until a person checks them. Header only when there are none. |
| `results/class_counts.png`, `results/split_counts.png` | No | Written by `python src/data/plot_counts.py` from the manifests. |
| `results/near_dup_review.png`, `results/cross_label_review.png` | No | Written by `python src/data/review_near_dups.py` (add `--cross-label` for the second), not by this step. Show the most borderline removed pairs, or every label-conflict group, side by side for checking the cutoffs by eye. |

**After reviewing a conflict group:** if it is a real copy, delete the image with the wrong label from `data/raw/` and rerun, so the right one goes back into the splits. If it is a false match (two different objects), note the group and file paths for the team. There is no switch yet to let both images back in. If false matches show up, lower `max_phash` or `max_dhash`, or add an "allowed pairs" list to the script. Today there are no conflicts to review.

### Using the class weights in PyTorch

`class_order` is alphabetical, the same order `ImageFolder` uses for its labels. Pick one of the two weight sets:

```python
import json, torch
w = json.load(open("results/class_weights.json"))["weights"]
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w["inverse_frequency"]["in_class_order"]))  # or w["sqrt_inverse_frequency"]["in_class_order"]
```

Train once with each and keep the one with the higher **validation macro F1**.

## Baseline to beat: always guess recycling

A "model" that answers recycling for every photo gets these scores on the
validation split (printed by `build_splits.py` on every run):

| Classes | Validation images | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 4 | 1,276 | 68.10% | 0.2026 |
| 3 | 1,130 | 76.90% | 0.2898 |

Accuracy looks good only because recycling is most of the data. Macro F1
averages the F1 score of every class equally, so ignoring compost, trash and
hazardous drags it down. **Report macro F1, not accuracy.** A useful model must
beat 0.2026 by a wide margin.

## Near-duplicate cutoffs

Both cutoffs live under `near_duplicates:` in `configs/label_map.yaml`:

```yaml
near_duplicates:
  max_phash: 3     # pHash distance must be <= 3 ...
  max_dhash: 2     # ... AND dHash distance must be <= 2 for a pair to count as the same picture
```

Same label: the copy is removed. Different label: the group goes to
`results/conflicts_to_review.csv` and into no split. Lower numbers are stricter
and treat fewer images as copies. After changing them, run
`python src/data/review_near_dups.py` and look at `results/near_dup_review.png`
to check the borderline pairs are real copies. The spot-check results are in
`docs/dataset_inventory.md`.

## Adding Amy's images (or any extra source) in 2 steps

Every dataset listed under `datasets:` in `configs/label_map.yaml` goes through
the same corrupted-file check, duplicate removal and split. An example entry for
Amy's photos is already there, switched off with `#`.

1. **Copy the photos** into one folder per class:
   `data/raw/amy_images/trash/`, `recycling/`, `compost/` and `hazardous/`.
   JPG, PNG, BMP and WEBP all work, and upper-case extensions are fine. Convert
   iPhone HEIC photos to JPG first.
2. **Switch the entry on**: in `configs/label_map.yaml`, delete the `# ` at the
   start of the 7 lines from `# amy_images:` down to `#     hazardous: hazardous`.
   Then rerun `python src/data/build_splits.py`.

Her photos are checked against every other image, so a copy of a Kaggle photo is
removed, and a look-alike with a different label shows up in
`results/conflicts_to_review.csv`. The images never go into git. Share them through
Drive or a zip.

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
