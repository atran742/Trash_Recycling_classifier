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
Files skipped: 1 not an image, 0 outside a class folder, 0 unsupported format, 0 corrupted
  not an image: data/raw/personal_battery/README.md
Checking duplicates: pHash <= 3 and dHash <= 2...

Removed 2 exact duplicates and 57 near duplicates. Held 0 images in 0 label-conflict groups for review (results/conflicts_to_review.csv). Dropped 0 corrupted and 0 unsupported files. 8519 images are split (8519 base, 0 new).
Dropped 17 known duplicates listed in configs/label_map.yaml (the copy outside train is dropped; see results/removed_duplicates.csv).

class          train  validation    test   total
------------------------------------------------
compost          718         154     153    1025
hazardous        681         146     145     972
recycling       4059         865     864    5788
trash            503         106     108     717
------------------------------------------------
TOTAL           5961        1271    1270    8502

Train split share and class weights (results/class_weights.json):
  class         train   % train   inverse  sqrt inv
  compost         718    12.04%    2.0756    1.0983
  hazardous       681    11.42%    2.1883    1.1277
  recycling      4059    68.09%    0.3671    0.4619
  trash           503     8.44%    2.9627    1.3121

Baseline 'always guess recycling' on validation (1271 images): accuracy 68.06%, macro F1 0.2025

Overlap check passed: no exact or near duplicate appears in more than one split.
Count check: 8578 found - 0 corrupted - 0 unsupported format - 2 exact duplicates - 57 near duplicates - 17 known duplicates - 0 held for review = 8502; 8502 in the splits; 76 rows in results/removed_duplicates.csv; 0 rows in results/conflicts_to_review.csv.
Count check passed.
Split version: 07a0572  manifest_sha256: 4d578382b730  (both saved in results/class_weights.json)
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
clean result is the 15 flagged pairs already judged as look-alikes in
`docs/dataset_inventory.md`, with 0 true copies. The 17 confirmed copies are
dropped through `known_duplicates` in `configs/label_map.yaml`. If a new pair
shows up, look at it in `results/leakage_review.png`. If it is a true copy, add it to
`known_duplicates`.

## What else the split step writes

| File | In git? | What it is |
|---|---|---|
| `data/train/`, `data/validation/`, `data/test/` | No | The image folders, one sub-folder per class. Works with `torchvision.datasets.ImageFolder`. |
| `data/splits/train.csv`, `validation.csv`, `test.csv` | Yes | Manifests with columns `path,label`. Paths are relative to the repo root and point at the copied images. They are small and hold no images, so everyone trains on the exact same split. |
| `results/class_weights.json` | Yes | `split_version` (the last commit that changed `build_splits.py` or `configs/`), `manifest_sha256` (fingerprint of the split CSVs), class order, train counts, percent of train per class, and two weight sets computed from the train split only: inverse frequency and square-root inverse frequency. |
| `results/removed_duplicates.csv` | Yes | Every dropped file with `file,kept_copy,reason,distance,dhash_distance`. `distance` is the pHash distance. Reasons are `exact_duplicate`, `near_duplicate`, `known_duplicate` (listed in `known_duplicates` in `configs/label_map.yaml`) or `corrupted`. |
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
| 4 | 1,271 | 68.06% | 0.2025 |
| 3 | 1,125 | 76.89% | 0.2898 |

Accuracy looks good only because recycling is most of the data. Macro F1
averages the F1 score of every class equally, so ignoring compost, trash and
hazardous drags it down. **Report macro F1, not accuracy.** A useful model must
beat 0.2025 by a wide margin.

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

## Adding Amy's images in 3 steps

Her photos never go into git: everything under `data/` is ignored, except the
split CSVs and the battery README. Share the photos through Drive or a zip.

**Before step 3, once:** a Kaggle token in `~/.kaggle/` (setup step 3 above),
because `download.py` fetches the Kaggle images the first time. Also run
`pip install -r requirements.txt`, which installs `pillow-heif` for iPhone
HEIC photos. If a package is missing, `download.py` stops right away and
prints that command, and `build_splits.py` stops rather than drop HEIC photos.

**Run `pip install -r requirements.txt` before step 3.**

1. **Put her folders in `data/raw/amy_images/`,** named `trash`, `recycling`,
   `compost` and `hazardous`. Upper or lower case both work, and so do common
   variants such as `Recycle`, `recyclables`, `garbage`, `hazard` and
   `batteries` (the full list is under `folder_aliases:` in
   `configs/label_map.yaml`). Sub-folders inside a class folder are fine. JPG,
   PNG, WEBP, TIFF and iPhone HEIC/HEIF all work, and HEIC is saved as JPG in the
   splits.
2. **Switch the entry on:** in `configs/label_map.yaml`, delete the two
   characters `# ` at the start of the 7 lines from `# amy_images:` down to
   `#     hazardous: hazardous`.
3. **Run:**
   ```bash
   python src/data/download.py
   ```
   ```bash
   python src/data/build_splits.py
   ```

### What a good run prints

These lines come from a test run in a git clone with a fake `amy_images` folder (every tricky case included). Her numbers will differ.

```text
  Ignored folders in data/raw/amy_images/ (system or zip metadata): __MACOSX
  data/raw/amy_images/Trash/ -> trash (trash)
  data/raw/amy_images/Recycle/ -> recycling (recycling)
  data/raw/amy_images/recyclables/ -> recycling (recycling)
  data/raw/amy_images/compost/ -> compost (compost)
  data/raw/amy_images/Batteries/ -> hazardous (hazardous)
Files skipped: 5 not an image, 1 outside a class folder, 1 unsupported format, 1 corrupted
  not an image: data/raw/personal_battery/README.md, data/raw/amy_images/.DS_Store, data/raw/amy_images/Trash/IMG_0300.MOV (+2 more)
  outside a class folder: data/raw/amy_images/IMG_9999.jpg
  unsupported format: data/raw/amy_images/Batteries/IMG_0400.DNG
  corrupted: data/raw/amy_images/Trash/broken.jpg
Removed 7 exact duplicates and 60 near duplicates. Held 6 images in 3 label-conflict groups for review (results/conflicts_to_review.csv). Dropped 1 corrupted and 1 unsupported files. 8653 images are split (8516 base, 137 new).
Placed 137 new images: 0 look-alike groups kept together, 1 groups followed a base look-alike.
Overlap check passed: no exact or near duplicate appears in more than one split.
Count check passed.
Split version: 07a0572-dirty  manifest_sha256: d45367601158  (both saved in results/class_weights.json)
```

- **The `-> trash`, `-> recycling`, ... lines** show how each of her folders was read, one line per folder. `__MACOSX` and hidden folders are ignored with a note.
- **`Files skipped`** lists every file that did not become a training image, and why. Check the examples if a number looks high.
- **`(N base, M new)`** shows how many Kaggle and how many of her photos made it into the splits.
- **`Count check passed`** means every file is accounted for.
- **`-dirty` after the split version** is expected, because step 2 edited `label_map.yaml`. `manifest_sha256` is what identifies her split.
- **A `__2/__3 suffix` line** can appear if two of her files would get the same copied name. Both are kept.

After the build, run `python src/data/check_leakage.py` and open
`results/conflicts_to_review.csv`.

### How her photos are treated

- **iPhone rotation is fixed in the pipeline.** Phones often store a photo sideways plus an EXIF "rotate me" tag. Her photos are turned upright before the duplicate checks and before copying, and the copies are saved without the tag, so the training loader needs no extra step. A sideways copy and an upright copy of one photo count as duplicates. Kaggle images are copied byte for byte, exactly as before.
- **A copy of a Kaggle photo with the same label** is removed, and the Kaggle photo is kept.
- **A copy with a different label** holds both images out of every split and lists them in `results/conflicts_to_review.csv` for a person to check.
- **Adding her photos never moves a Kaggle image to another split.** `base_datasets:` in `configs/label_map.yaml` is split first, on its own, exactly as before.
- **Look-alike photos stay in one split.** Her photos that look alike (pHash ≤ 6 and dHash ≤ 8, the leakage check's rule) go to the same split, and so do near-copies of a Kaggle train photo. Burst shots of one object taken with a bigger camera shift than that can still land in different splits. A photo that looks like Kaggle validation or test images joins the split holding most of them.

### What each error means

| Message starts with | What happened | What to do |
|---|---|---|
| `data/raw/amy_images/ not found` | The entry is switched on, but the folder is not there. | Create the folder with the 4 class folders, or put `# ` back on the 7 lines. |
| `data/raw/amy_images/ has no folders inside` | The folder is empty. | Put the 4 class folders inside it. |
| `data/raw/amy_images/ has no images` | The class folders hold no photos (only things like `.DS_Store`). | Copy the photos in. |
| `... has folder(s) that match no class: X` | A folder name is not a class name or a known variant. The message lists the folders found and the names accepted. | Rename the folder, or add the name under `folder_aliases:`. |
| `label_map.yaml folder(s) ... have no folder` | One class has no folder at all. | Add that class folder. |
| `configs/label_map.yaml could not be read` | Step 2 changed the spacing, for example by deleting only `#`. | Delete exactly `# ` (hash and one space) on each of the 7 lines. |
| `Found N iPhone HEIC/HEIF photos ... pillow-heif ... is not installed` | Her photos are HEIC but the package that opens them is missing. Nothing was built. | Run `pip install -r requirements.txt` and rerun. |
| `ERROR: missing Python packages` (from `download.py`) | A package from `requirements.txt` is not installed. | Run `pip install -r requirements.txt` and rerun. |

## Switching between 3 and 4 classes

Edit one line at the top of `configs/label_map.yaml`, then rerun `build_splits.py`:

```yaml
num_classes: 3   # or 4
```

With 3 classes, every folder mapped to `hazardous` is left out. For a quick test
without editing the file, run `python src/data/build_splits.py --num-classes 3`.

## Adding personal battery photos

Put photos into `data/raw/personal_battery/` and follow the naming rules
in that folder's `README.md`. Rerun `build_splits.py`, and they join the
`hazardous` class automatically. Like Amy's photos, they are placed around the
Kaggle split without moving it, and look-alike shots stay in one split.

## Troubleshooting

- **"No Kaggle token found"**: redo setup step 3. On Windows the folder is `C:\Users\<you>\.kaggle\`.
- **`CERTIFICATE_VERIFY_FAILED` on macOS**: the python.org installer ships without
  root certificates. Run `/Applications/Python\ 3.*/Install\ Certificates.command` once.
- **"label_map.yaml key ... has no folder"**: a folder name in the yaml does not
  match `data/raw/`. Run `download.py` again to print the real folder names.
