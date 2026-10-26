# Data Pipeline: zero to train / validation / test / wild_test

This takes a teammate from a fresh clone to `data/train`, `data/validation`,
`data/test` (the frozen benchmark) and `data/wild_test` folders that are ready
for training and scoring. None of these images go into git.

## One-time setup

1. **Python 3.10 or newer** and this repo cloned.
2. **A virtual environment**, so packages stay inside the project:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. **A Kaggle token.** Sign in at kaggle.com, open Settings, then API, and click
   "Create New Token". Save it as `~/.kaggle/kaggle.json` (file download) or
   `~/.kaggle/access_token` (token string), then run `chmod 600` on that file.

## The 4 commands

Run these from the repo root with the virtual environment active.

```bash
pip install -r requirements.txt
python src/data/download.py
python src/data/crop_taco.py
python src/data/build_splits.py
```

Optional: `python src/data/plot_counts.py` draws `results/class_counts.png` and
`results/split_counts.png` (bar charts of the counts, sized for a phone screen;
not committed).

| Command | What it does | Time |
|---|---|---|
| `pip install -r requirements.txt` | Installs kaggle, pyyaml, pillow, pillow-heif, imagehash, numpy, matplotlib, scikit-learn and opencv-python. | under 1 minute |
| `python src/data/download.py` | Downloads `garbage12_enhanced` (about 557 MB unzipped) into `data/raw/`, then the two wild datasets: TACO (the official annotations file, pinned to one commit, and its 1,500 photos, about 2.5 GB) and RealWaste (657 MB zip from UCI). Skips what is already there. Fetches `garbage_v2` only if there are fewer than 500 battery images. | 10 to 30 minutes, mostly TACO |
| `python src/data/crop_taco.py` | Cuts one image per TACO object whose category maps to a class in `configs/label_map.yaml` (box + 10% padding, at least 64 px a side) into `data/raw/taco_crops/<class>/<category>/`, and prints counts per category and class. | about 2 minutes |
| `python src/data/build_splits.py` | Applies `configs/label_map.yaml`, drops broken files, exact duplicates and near duplicates, splits the Kaggle set 70/15/15 into `data/train`, `data/validation` and `data/test` (the frozen benchmark, checked byte for byte), keeps every locked TACO and RealWaste image in its frozen split (train, validation or `data/wild_test`, also checked byte for byte), puts every other image 70/15 into train and validation only, checks the counts add up, and writes the CSV manifests and class weights. | about 10 minutes |

## What you should see

With `num_classes: 4`, the last command prints (abridged; the first lines map each
folder, and the recycling cap lists every TACO category with 0 of its unlocked images kept,
because the 400 locked recycling images fill the cap):

```text
Files skipped: 6 not an image, 0 outside a class folder, 0 unsupported format, 0 corrupted
  not an image: data/raw/personal_battery/README.md, data/raw/team_photos/README.md, data/raw/team_photos/trash/.gitkeep (+3 more)
Checking duplicates: pHash <= 3 and dHash <= 2...

Removed 2 exact duplicates and 57 near duplicates, and 0 new images that copy a base or locked image. Held 0 images in 0 label-conflict groups for review (results/conflicts_to_review.csv). Dropped 0 corrupted and 0 unsupported files. 11204 images go on to the split (8519 base, 2685 new).
Dropped 17 known duplicates listed in configs/label_map.yaml (the copy outside train is dropped; see results/removed_duplicates.csv).
Recycling cap: 1075 new recycling images from realwaste, taco_crops, kept 400 (cap 400, seed 42; 400 locked images kept first, 0 of 675 unlocked):

Wild split (2010 new images: 2010 locked, 0 unlocked; 933 groups, 324 with more than one image; seed 42): 0 dropped as look-alikes of benchmark test images, 0 as look-alikes of wild test images, 0 wild_test images dropped as look-alikes of train/benchmark images.
  source        class                train  validation  wild_test  total
  realwaste     compost    locked      291          60         60    411
  taco_crops    recycling  locked      280          60         60    400
  taco_crops    trash      locked      838         183        178   1199

class          train  validation    test  wild_test   total
-----------------------------------------------------------
compost         1009         214     153         60    1436
hazardous        681         146     145          0     972
recycling       4339         925     864         60    6188
trash           1341         289     108        178    1916
-----------------------------------------------------------
TOTAL           7370        1574    1270        298   10512
(test = the frozen Kaggle benchmark; wild_test = the frozen TACO / RealWaste test, never new photos)

Train split share and class weights (results/class_weights.json):
  class         train   % train   inverse  sqrt inv
  compost        1009    13.69%    1.8261    1.1214
  hazardous       681     9.24%    2.7056    1.3650
  recycling      4339    58.87%    0.4246    0.5408
  trash          1341    18.20%    1.3740    0.9728

Baseline 'always guess recycling' on validation (1574 images): accuracy 58.77%, macro F1 0.1851

Frozen check passed: data/splits/test.csv is byte-identical to configs/frozen/test.csv (sha256 44805ae865a0).
Frozen check passed: data/splits/wild_test.csv is byte-identical to configs/frozen/wild_test.csv (sha256 ee224c6731f3).
Camera test check passed: no data/raw/camera_test/ photo is in any split.
Overlap check passed: no exact or near duplicate appears in more than one split.

Count check: 11263 found - 0 corrupted - 0 unsupported format - 2 exact duplicates - 57 near duplicates - 0 copies of base or locked images - 17 known duplicates - 675 recycling cap - 0 look-alikes of benchmark - 0 look-alikes of wild test - 0 wild_test look-alikes - 0 held for review = 10512; 10512 in the splits; 751 rows in results/removed_duplicates.csv; 0 rows in results/conflicts_to_review.csv.
Count check passed.
Split version: b40b08a  manifest_sha256: 08d058850b31  (both saved in results/class_weights.json)
```

The result is the same on every machine because the seed is fixed at 42.

**Check `split_version` and `manifest_sha256` before comparing models.** Both are
printed at the end of `build_splits.py` and saved in `results/class_weights.json`.
`split_version` is the last commit that changed `src/data/build_splits.py` or
`configs/`, so docs, results and helper-script commits do not change it. `manifest_sha256` fingerprints
the split lists themselves, so it also changes when images are added to
`data/raw/`. Train and validation compare fairly only if both match. Test scores always compare, because both test sets are frozen. Splits from before the
data-cleanup branch (8,403 images, older duplicate rule) are stale: rebuild them
before training.

**After any rebuild, run `python src/data/check_leakage.py`.** It is read-only. A
clean result is the 15 flagged pairs already judged as look-alikes in
`docs/dataset_inventory.md` (validation and benchmark test), 0 for the wild test, and 0 true copies. The 17 confirmed copies are
dropped through `known_duplicates` in `configs/label_map.yaml`. If a new pair
shows up, look at it in `results/leakage_review.png`. If it is a true copy, add it to
`known_duplicates`.

## What else the split step writes

| File | In git? | What it is |
|---|---|---|
| `data/train/`, `data/validation/`, `data/test/`, `data/wild_test/` | No | The image folders, one sub-folder per class. Works with `torchvision.datasets.ImageFolder`. `data/test/` is the frozen benchmark; `data/wild_test/` holds only non-Kaggle images. |
| `data/splits/train.csv`, `validation.csv`, `test.csv`, `wild_test.csv` | Yes | Manifests with columns `path,label`. Paths are relative to the repo root and point at the copied images. They are small and hold no images, so everyone trains on the exact same split. `test.csv` and `wild_test.csv` must stay byte-identical to their frozen copies in `configs/frozen/` (the build stops otherwise). |
| `results/class_weights.json` | Yes | `split_version` (the last commit that changed `build_splits.py` or `configs/`), `manifest_sha256` (fingerprint of the split CSVs), class order, train counts, percent of train per class, and two weight sets computed from the train split only: inverse frequency and square-root inverse frequency. |
| `results/removed_duplicates.csv` | Yes | Every dropped file with `file,kept_copy,reason,distance,dhash_distance`. `distance` is the pHash distance. Reasons are `exact_duplicate`, `near_duplicate`, `known_duplicate` (listed in `known_duplicates` in `configs/label_map.yaml`), `corrupted`, `unsupported_format`, `copy_of_base` and `copy_of_locked` (a new image copying a Kaggle or locked image), `recycling_cap`, `lookalike_of_benchmark` and `lookalike_of_wild_test` (a new image looking like a frozen test image), and `wild_test_lookalike`. |
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
| 4 (with the wild datasets, Oct 25) | 1,574 | 58.77% | 0.1851 |
| 4 (with the wild datasets, Oct 24) | 1,478 | 62.58% | 0.1925 |
| 4 (Kaggle only, before Oct 24) | 1,271 | 68.06% | 0.2025 |
| 3 (Kaggle only, before Oct 24) | 1,125 | 76.89% | 0.2898 |

Accuracy looks good only because recycling is most of the data. Macro F1
averages the F1 score of every class equally, so ignoring compost, trash and
hazardous drags it down. **Report macro F1, not accuracy.** A useful model must
beat 0.1851 by a wide margin.

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
2. **The entry is already switched on** in `configs/label_map.yaml` (Amy did
   this on Oct 23) and marked `optional: true`. On a machine without the
   folder, `build_splits.py` prints a warning and builds from the other
   datasets. Any dataset without that flag still stops when its folder is missing.
3. **Run:**
   ```bash
   python src/data/download.py
   ```
   ```bash
   python src/data/build_splits.py
   ```

### What a good run prints

These lines come from a test run in a git clone with a fake `amy_images` folder (every tricky case included), made before the frozen benchmark existed. Her numbers will differ, and the duplicate summary now has the format shown under "What you should see" (`... go on to the split (N base, M new)`, then a `Wild split (...)` line and a table of new images per source, class and split).

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
Overlap check passed: no exact or near duplicate appears in more than one split.
Count check passed.
Split version: 07a0572-dirty  manifest_sha256: d45367601158  (both saved in results/class_weights.json)
```

- **The `-> trash`, `-> recycling`, ... lines** show how each of her folders was read, one line per folder. `__MACOSX` and hidden folders are ignored with a note.
- **`Files skipped`** lists every file that did not become a training image, and why. Check the examples if a number looks high.
- **`(N base, M new)`** and the `Wild split` table show how many Kaggle and how many of her photos made it into the splits, and where her photos went.
- **`Count check passed`** means every file is accounted for.
- **`-dirty` after the split version** means `build_splits.py` or `configs/` has uncommitted edits on that machine (the sample above was made that way). With nothing edited there is no `-dirty`. `manifest_sha256` is what identifies her split.
- **A `__2/__3 suffix` line** can appear if two of her files would get the same copied name. Both are kept.

After the build, run `python src/data/check_leakage.py` and open
`results/conflicts_to_review.csv`.

### How her photos are treated

- **iPhone rotation is fixed in the pipeline.** Phones often store a photo sideways plus an EXIF "rotate me" tag. Her photos are turned upright before the duplicate checks and before copying, and the copies are saved without the tag, so the training loader needs no extra step. A sideways copy and an upright copy of one photo count as duplicates. Kaggle images are copied byte for byte, exactly as before.
- **A copy of a Kaggle photo with the same label** is removed, and the Kaggle photo is kept.
- **A copy with a different label** holds her photo out of every split and lists it in `results/conflicts_to_review.csv` for a person to check. The Kaggle photo stays where it is.
- **Adding her photos never moves, removes or joins a Kaggle image.** `base_datasets:` in `configs/label_map.yaml` is split first, on its own, exactly as before, and its test split is the frozen benchmark: none of her photos ever goes into `data/splits/test.csv`.
- **Her photos go only into train and validation** (70/15, seed 42). Since Oct 26 both test sets are frozen, so no new photo ever enters `test.csv` or `wild_test.csv`; to score on photos like hers, use `data/raw/camera_test/`.
- **Look-alike photos stay in one split.** Her photos that look alike (pHash ≤ 6 and dHash ≤ 8, the leakage check's rule) go to the same split. A group that looks like a Kaggle train photo goes to train, one that only looks like Kaggle validation photos goes to validation, and one that looks like a benchmark test photo or a frozen wild test photo is dropped (`lookalike_of_benchmark` or `lookalike_of_wild_test` in `results/removed_duplicates.csv`). A copy of a frozen TACO or RealWaste image is dropped too (`copy_of_locked`). Burst shots of one object taken with a bigger camera shift than that can still land in different splits.

### What each error means

| Message starts with | What happened | What to do |
|---|---|---|
| `WARNING: data/raw/amy_images/ not found ... marked optional` | The folder is not on this machine. The build goes on without her photos. | Nothing, unless you expected her photos: then copy the folder in and rerun. |
| `data/raw/<dataset>/ not found, but "<dataset>" is switched on` | A dataset without `optional: true` has no folder. | Create the folder, or comment the entry out. |
| `data/raw/amy_images/ has no folders inside` | The folder is empty. | Put the 4 class folders inside it. |
| `data/raw/amy_images/ has no images` | The class folders hold no photos (only things like `.DS_Store`). | Copy the photos in. |
| `... has folder(s) that match no class: X` | A folder name is not a class name or a known variant. The message lists the folders found and the names accepted. | Rename the folder, or add the name under `folder_aliases:`. |
| `label_map.yaml folder(s) ... have no folder` | One class has no folder at all. | Add that class folder. |
| `configs/label_map.yaml could not be read` | The file was edited by hand and its spacing broke (YAML needs the same indentation as the lines around it). | Undo the edit with `git checkout configs/label_map.yaml`, or fix the indentation. |
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

## Team photos: `data/raw/team_photos/`

Phone photos the team takes go in `data/raw/team_photos/<class>/` (`trash`,
`recycling`, `compost`, `hazardous`), one item per photo on a plain background;
the naming and photo rules are in that folder's `README.md`. Rerun
`build_splits.py` and they are treated like Amy's photos: duplicates removed,
turned upright by their EXIF tag, HEIC converted, placed around the fixed
Kaggle split with look-alikes kept in one split. The dataset is `optional: true`,
so a machine whose folders are still empty prints
`WARNING: data/raw/team_photos/ has no images yet` and builds without them;
the manifests are then byte-identical to a build that never had the folder.

## Camera test set: `data/raw/camera_test/` (held out)

Photos taken with the webcam or Pi camera in the demo room go in
`data/raw/camera_test/<class>/` and **nowhere else**. `build_splits.py` never
reads that folder: it is not in `configs/label_map.yaml`, a dataset entry that
points at it stops the build, and after the manifests are written the build
checks them and stops if any path mentions `camera_test`. Score the photos with

```bash
python src/eval/export_predictions.py --model models/waste_classifier.pth --arch mobilenet_v3_small
python src/eval/score_camera_test.py --model models/waste_classifier.pth --arch mobilenet_v3_small
```

The first command gives the validation predictions that pick the "I'm not sure"
cutoff; the second compares every camera photo against the train images with
the pipeline's pHash / dHash rules (a near copy prints a WARNING), writes
`results/camera_test_predictions.csv` and scores it with `evaluate.py`.

## Wild datasets: TACO and RealWaste

Two open datasets add messy real-world photos (litter on the ground, items from a
landfill) next to the clean Kaggle images. How each category was mapped with the
City of San Diego guide, the quality check and the licenses are in
`docs/dataset_inventory.md`. How they are used:

- **They never touch the benchmark.** `data/splits/test.csv` stays the 1,270 Kaggle
  test images of Oct 24, byte for byte; the build stops if it changes.
- **Their split is frozen.** They were split once, 70/15/15 with seed 42, into train,
  validation and `data/splits/wild_test.csv` (crops of one TACO photo, RealWaste shots of
  one item and look-alikes in one split). On Oct 26 (stamp def16b3) every one of them was
  locked to its split in `configs/frozen/wild_split_lock.csv`, and `wild_test.csv` was
  frozen in `configs/frozen/wild_test.csv`; the build stops if it differs by one byte.
- **A wild test image may not look like any train or benchmark image** (pHash <= 6
  and dHash <= 8, the leakage-check rule).
- **Recycling is capped**: at most 400 new recycling images, spread evenly over the
  categories, because recycling is already most of train.
- Both are `optional: true`, but the frozen wild test needs them: a machine without them
  warns, and the build then stops at the frozen check. Run `download.py` and `crop_taco.py`.

**New sources go only into train and validation.** `amy_images`, `team_photos`,
`personal_battery`, any new dataset, and any new file in an old one are not in the lock,
so they are split 70/15 (about 82% / 18%) into train and validation and never into either
test set. A new photo that copies or looks like a frozen test image is dropped. To score on
demo-room photos, use `data/raw/camera_test/` instead.

**Re-freezing** (only on a team decision, because every earlier score stops being
comparable): build, run `python src/data/freeze_wild_split.py`, and put the three sha256 lines
it prints under `frozen:` in `configs/label_map.yaml`.

After a build, `python src/data/review_new_images.py` saves
`results/new_images_<class>.png`, 30 random new images per class for a label check by eye.

## Troubleshooting

- **"No Kaggle token found"**: redo setup step 3. On Windows the folder is `C:\Users\<you>\.kaggle\`.
- **`CERTIFICATE_VERIFY_FAILED` on macOS**: the python.org installer ships without
  root certificates. Run `/Applications/Python\ 3.*/Install\ Certificates.command` once.
- **"label_map.yaml key ... has no folder"**: a folder name in the yaml does not
  match `data/raw/`. Run `download.py` again to print the real folder names.
