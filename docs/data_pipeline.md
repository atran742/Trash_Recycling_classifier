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
| `pip install -r requirements.txt` | Installs kaggle, pyyaml and pillow. | under 1 minute |
| `python src/data/download.py` | Downloads `garbage12_enhanced` (about 557 MB unzipped) into `data/raw/`. Skips it if it is already there. Also fetches `garbage_v2` only if there are fewer than 500 battery images. | a few minutes |
| `python src/data/build_splits.py` | Applies `configs/label_map.yaml`, drops duplicates and broken files, and copies a 70/15/15 split into `data/train`, `data/validation` and `data/test`. | about 1 minute |

## What you should see

With `num_classes: 4`, the last command prints this table near the end:

```text
class          train  validation    test   total
------------------------------------------------
compost          724         155     156    1035
hazardous        685         146     148     979
recycling       4085         875     876    5836
trash            508         108     110     726
------------------------------------------------
TOTAL           6002        1284    1290    8576
Overlap check passed: no image appears in more than one split.
```

The result is the same on every machine because the seed is fixed at 42.

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
