# Dataset Inventory

Owner: Isiah (data side). Last updated: 2026-10-17.

Images are **never committed**. They download into `data/raw/` (git-ignored) with
`python src/data/download.py`. Licenses are copied from each Kaggle page on
2026-10-17. Check the page again before publishing anything.

## Summary

| Name | Kaggle URL | License (as listed on Kaggle) | Images | Classes | Clean or real-world | Battery images | Status |
|---|---|---|---:|---|---|---:|---|
| `garbage12_enhanced` | [huberthamelin/garbage-classification-labels-corrections](https://www.kaggle.com/datasets/huberthamelin/garbage-classification-labels-corrections) | Database: Open Database, Contents: Database Contents | 15,870 (counted locally) | 12: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass | Clean. Mostly web-scraped, centered objects | **979** (counted locally) | **Primary. Downloaded and mapped.** |
| `garbage12` | [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | Database: Open Database, Contents: © Original Authors | 15,150 per page text; its folders add up to 15,515 | Same 12 classes | Clean. Mostly web-scraped | 945 (Kaggle page) | Listed only. Not downloaded. |
| `garbage_v2` | [sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | MIT | 13,348 (Kaggle page) | 10: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash | Mixed. Phone-app photos plus curated web images | 756 (Kaggle page) | Listed only. Not downloaded. |
| `personal_battery` | none, our own photos | Ours. Team decides before publishing | 0 today | 1: hazardous | **Real-world** | 0 today, target 50 | Folder and photo rules ready. |

## Overlap notes

- **`garbage12_enhanced` contains two copies.** `garbage_classification_corrected/` has 15,467 files and `garbage_classification_enhanced/` has 15,870. Hashing both found 15,459 images in both folders, 409 only in enhanced, and 5 only in corrected. The corrected folder also holds 3 exact duplicate files of its own. We train from the enhanced folder only.
- **Inside the enhanced folder,** 2 files are exact duplicates across class folders: `metal/metal459.jpg` equals `white-glass/white-glass412.jpg`, and `white-glass/white-glass761.jpg` equals `plastic/plastic556.jpg`. Both pairs map to recycling, so no label conflict. `build_splits.py` keeps the copy from the folder listed first in `label_map.yaml`, so it keeps `metal459` and `plastic556`.
- **`garbage12` is the original version of `garbage12_enhanced`.** The enhanced set removed duplicates, fixed 70 labels (listed in `labels_corrections.csv`), and added web images. Downloading both would mostly add duplicates, so `garbage12` is not downloaded.
- **`garbage_v2`** is only needed if `garbage12_enhanced` has fewer than 500 battery images. It has 979, so `download.py` skips it. Its overlap with `garbage12_enhanced` was not measured because it was not downloaded.

## `garbage12_enhanced`: real per-class counts

Counted by `download.py` from `data/raw/garbage12_enhanced/garbage_classification_enhanced/`.

| Folder | Maps to (4 classes) | Images |
|---|---|---:|
| battery | hazardous | 979 |
| biological | compost | 1,035 |
| brown-glass | recycling | 619 |
| cardboard | recycling | 936 |
| clothes | VERIFY (skipped) | 5,286 |
| green-glass | recycling | 641 |
| metal | recycling | 837 |
| paper | recycling | 1,108 |
| plastic | recycling | 901 |
| shoes | VERIFY (skipped) | 2,006 |
| trash | trash | 726 |
| white-glass | recycling | 796 |
| **Total** | | **15,870** |

## Train / validation / test splits (seed 42, 70/15/15)

Output of `python src/data/build_splits.py` on 2026-10-17. 8,578 images were read,
2 exact duplicates were dropped, and 0 were corrupted.

**`num_classes: 4`**

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| compost | 724 | 155 | 156 | 1,035 |
| hazardous | 685 | 146 | 148 | 979 |
| recycling | 4,085 | 875 | 876 | 5,836 |
| trash | 508 | 108 | 110 | 726 |
| **Total** | **6,002** | **1,284** | **1,290** | **8,576** |

**`num_classes: 3`** (hazardous excluded)

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| compost | 724 | 155 | 156 | 1,035 |
| recycling | 4,085 | 875 | 876 | 5,836 |
| trash | 508 | 108 | 110 | 726 |
| **Total** | **5,317** | **1,138** | **1,142** | **7,597** |

Both runs passed the overlap check: no image appears in more than one split.

## Class imbalance to watch

- `recycling` has 5,836 images, about 8 times `trash` (726). Amy may want class weights or oversampling.
- `clothes` and `shoes` are 7,292 images, nearly half the download. They are skipped until the team decides.
- `hazardous` comes from web photos only today. The 50 personal battery photos add real-world examples.

## License notes

- `garbage12_enhanced` is ODbL/DbCL as a database, but its original author says the web-scraped images belong to their photographers. That is fine for a class project and for training. Do not redistribute the images, which matches our "no images in git" rule.
- `garbage_v2` is MIT, the most permissive of the three.

## Other candidates looked at, not used

| Kaggle URL | License | Why not |
|---|---|---|
| [akshat103/e-waste-image-dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset) | Apache 2.0 | Electronics, not batteries. Revisit if hazardous grows to cover e-waste. |
| [karansolanki01/garbage-classification](https://www.kaggle.com/datasets/karansolanki01/garbage-classification) | Apache 2.0 | 775 battery images that likely repeat `garbage12`. |
| [harshadsgore/e-waste-image-classification-dataset-18-classes](https://www.kaggle.com/datasets/harshadsgore/e-waste-image-classification-dataset-18-classes) | Other (specified in description) | Page says no blanket license for the images. Avoid. |

## Open questions for the team

1. **3 vs 4 classes.** Batteries fit none of recycling, compost or trash, and San Diego says they never go in curbside bins. Proposal: add `hazardous`. Flip `num_classes` in `configs/label_map.yaml`.
2. **Clothes and shoes.** San Diego curbside recycling does not accept them. Map to `trash`, drop them, or keep them as VERIFY?
