# Team photos (`data/raw/team_photos/`)

Photos **we take ourselves** of everyday items, so the model learns from real
phone shots and not only clean web images. They go through the same pipeline
as every other source: duplicate removal, leakage check, EXIF rotation, HEIC
conversion, and the 70 / 15 / 15 split (`python src/data/build_splits.py`).

**Images in this folder are never committed to git.** Only this README and the
four empty class folders are tracked. Share photos through Drive or a zip and
drop them in here locally. A machine without any photos here still builds:
the dataset is marked `optional: true` in `configs/label_map.yaml`, so the
build prints a warning and continues.

Demo-room photos taken with the webcam or Pi camera do **not** go here. They go
in `data/raw/camera_test/` (see its README) and are never used for training.

## Where to drop photos

```
data/raw/team_photos/
├── README.md          <- this file (tracked)
├── trash/             <- masks, wrappers, chip bags, broken ceramics, diapers, ...
├── recycling/         <- bottles, cans, cardboard, paper, clean plastic tubs, ...
├── compost/           <- food scraps, peels, coffee grounds, yard trimmings, ...
└── hazardous/         <- batteries, bulbs, paint, electronics, ...
```

The folder name is the label. Put each photo straight into its class folder
(no sub-folders needed; sub-folders are read too). Not sure which class an
item is under San Diego rules? Leave it out rather than guess.

## File naming

```
<item>_<nnn>.jpg        for example  chipbag_001.jpg, sodacan_012.jpg, banana_003.HEIC
                        (python src/data/capture_photos.py --dest team_photos names them
                        <class>_<date>_<time>.jpg, which works too)
```

Lowercase item name, underscore, three-digit counter. `.jpg`, `.png` or
iPhone `.HEIC` all work (HEIC is converted when the split is built). No
spaces. The pipeline renames copies to `team_photos__<class>__<file>` so two
people can both have a `chipbag_001.jpg` without a clash.

## Photo rules

1. **One item per photo.** The model classifies one thing at a time.
2. **Plain background.** A table, floor or wall with nothing else in frame.
3. **Phone camera is fine.** Normal photo mode, no filters, no portrait blur.
4. **Vary the lighting:** daylight, warm indoor light, dim light, some shadows.
5. **Vary the angle:** top-down, side, tilted, a few close-ups. The item should
   fill roughly 20 to 80 percent of the frame.
6. **Vary the item:** crushed and whole, clean and dirty, label facing away.
   Ten different chip bags beat ten shots of the same chip bag.
7. **Privacy:** no faces, no addresses, no personal documents in the frame.

## Before sharing

- Delete blurry shots and accidental repeats. Near copies are removed at build
  time anyway, but it saves everyone time.
- Note the count per class and the phone model in the team doc.
