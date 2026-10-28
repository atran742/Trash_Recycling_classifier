# Camera test set (`data/raw/camera_test/`) - held out, never trained on

Photos taken **with the webcam or the Raspberry Pi camera, in the demo room**,
of the kinds of items we will show at the demo. This is the closest thing we
have to the real demo, so it is kept apart from everything else:

- `build_splits.py` never reads this folder. It is not listed in
  `configs/label_map.yaml`, and the build stops if a `camera_test` path ever
  shows up in `data/splits/*.csv`.
- No photo from here may also be in `data/raw/team_photos/` or any other
  source. Before scoring, `score_camera_test.py` compares every photo here
  against the training images with the pipeline's pHash and dHash rules and
  warns on any near copy.
- **Images in this folder are never committed to git.** Only this README and
  the four empty class folders are tracked.

## Where to drop photos

```
data/raw/camera_test/
├── README.md
├── trash/
├── recycling/
├── compost/
└── hazardous/
```

The folder name is the label. Name photos `<item>_<nnn>.jpg`, for example `sodacan_001.jpg`.
`python src/data/capture_photos.py --dest camera_test` saves webcam photos here for you,
named `<class>_<date>_<time>.jpg` (for example `trash_20261027_163005.jpg`). Shift+1/2/3/4 records
one item for 10 seconds and saves a frame every 2 seconds, named `<class>_item<NN>_f<frame>.jpg`
(for example `trash_item03_f2.jpg`); `compare_models.py` scores those frames as one item. `raspberry_pi/camera_test.py`
saves a photo from the Pi camera that you can move here.

## How to take them

1. Use the same camera, distance and lighting the demo will use. Put the item
   where the demo box will be (the center of the frame).
2. One item per photo, plain background, a few angles per item. For several angles of one
   object, record it (Shift+number) rather than taking single photos: every single photo
   counts as an item of its own, and one recording is one item. Item numbers count only the
   photos on one computer: take the camera_test recordings on one computer, or rename items
   before two people's folders are put together.
3. `compare_models.py` needs at least 4 items per class (one recording or one single photo is one
   item). Aim for 10 or more items per class. Twenty per class gives a steadier number.

## Scoring (2 commands from the repo root)

```bash
python src/eval/export_predictions.py --model models/waste_classifier.pth --arch mobilenet_v3_small
python src/eval/score_camera_test.py --model models/waste_classifier.pth --arch mobilenet_v3_small
```

The first writes the validation predictions that pick the "I'm not sure" cutoff
(skip it if `results/val_predictions.csv` is already there). The second warns
about near copies of train images, writes `results/camera_test_predictions.csv`
(git-ignored) and scores it with `src/eval/evaluate.py`, so the cutoff still
comes from validation and not from these photos.
