# Dataset Inventory

Owner: Isiah (data side). Last updated: 2026-10-18.

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
- **Near duplicates** are covered in their own section below.
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

## Duplicate removal (`build_splits.py`, 2026-10-18)

The script reads 8,578 images in 4-class mode. It removes exact copies by SHA-256,
then near copies by perceptual hash (pHash) at distance 5 or less, comparing
across all classes. Near copies are grouped in chains: if A matches B and B
matches C, all three are one group. When a group carries more than one label,
the whole group is dropped as a label conflict. Every removed file is listed in
`results/removed_duplicates.csv`.

| Removed (4 classes) | Images |
|---|---:|
| Exact duplicates | 2 |
| Near duplicates (same label) | 125 |
| Label conflicts (10 groups) | 48 |
| Corrupted | 0 |
| **Remaining** | **8,403** |

In 3-class mode the counts are 2 exact, 114 near, and 40 conflict images in 7 groups, leaving 7,443.

In the CSV, a `near_duplicate` row's distance is measured to the kept copy:
0 (32 rows), 2 (50), 4 (40), 6 (2) or 8 (1). The rows above 5 belong to
chains, where the image matched the kept copy through another image. A
`label_conflict` row's distance is to its closest image in the group, and it is
always 4 or less.

### What the near-duplicate check really catches

I looked at the removed images by eye.

- **Near duplicates:** in a random sample of 18 pairs, all 6 at distance 0 and 5 of 6 at distance 2 were the same photo, such as a watermarked copy. At distance 4, only 3 of 6 were. The others were different objects with the same shape, like two different toothbrushes or two laptop batteries.
- **Label conflicts:** almost all 10 groups are false matches, not mislabeled copies. They are thin upright objects on a white background: brown and green bottles, AA batteries and toothbrushes. The largest group holds 15 images. pHash only keeps an 8x8 light/dark pattern, so these look alike to it. The groups mix recycling and trash (26 images), hazardous, recycling and trash (18), or hazardous and trash (4). They drop 23 trash, 21 recycling and 4 battery images.
- **Recommendation:** lowering `near_duplicate_max_distance` in `configs/label_map.yaml` from 5 to 2 or 3 would keep most of these legitimate images. The current value 5 is what the team asked for.

## Train / validation / test splits (seed 42, 70/15/15)

Output of `python src/data/build_splits.py` on 2026-10-18, after duplicate removal.

**`num_classes: 4`**

| Class | Train | Validation | Test | Total | % of train | Class weight |
|---|---:|---:|---:|---:|---:|---:|
| compost | 715 | 153 | 154 | 1,022 | 12.16% | 2.0559 |
| hazardous | 674 | 144 | 146 | 964 | 11.46% | 2.1810 |
| recycling | 4,009 | 859 | 860 | 5,728 | **68.18%** | 0.3667 |
| trash | 482 | 103 | 104 | 689 | 8.20% | 3.0498 |
| **Total** | **5,880** | **1,259** | **1,264** | **8,403** | 100% | |

**`num_classes: 3`** (hazardous excluded)

| Class | Train | Validation | Test | Total | % of train | Class weight |
|---|---:|---:|---:|---:|---:|---:|
| compost | 715 | 153 | 154 | 1,022 | 13.73% | 2.4284 |
| recycling | 4,010 | 859 | 860 | 5,729 | **76.98%** | 0.4330 |
| trash | 484 | 103 | 105 | 692 | 9.29% | 3.5875 |
| **Total** | **5,209** | **1,115** | **1,119** | **7,443** | 100% | |

Class weights use the train split only: `total_train / (num_classes * class_train_count)`.
They are saved in `results/class_weights.json`. All runs passed the overlap check:
no exact or near duplicate appears in two splits. Running 4, then 3, then 4 again
gave byte-identical 4-class outputs.

## Class imbalance to watch

- `recycling` is 68.18% of train with 4 classes and 76.98% with 3. A model that always says "recycling" would already score 68% accuracy, so watch per-class recall, not just accuracy.
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
