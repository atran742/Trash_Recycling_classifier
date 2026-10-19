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
- **Near duplicates** are covered in the duplicate removal section below.
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
| clothes | EXCLUDE (dropped Oct 18) | 5,286 |
| green-glass | recycling | 641 |
| metal | recycling | 837 |
| paper | recycling | 1,108 |
| plastic | recycling | 901 |
| shoes | VERIFY (skipped, undecided) | 2,006 |
| trash | trash | 726 |
| white-glass | recycling | 796 |
| **Total** | | **15,870** |

## Duplicate removal (`build_splits.py`, 2026-10-18)

The script finds 8,578 images in 4-class mode. It removes exact copies by
SHA-256, then near copies using two image fingerprints, pHash and dHash. Both
must agree before a pair counts as the same picture: pHash distance ≤ 3
(`max_phash`, team decision Oct 18) and dHash distance ≤ 2 (`max_dhash`), both
under `near_duplicates:` in `configs/label_map.yaml`.

| Pair | Result |
|---|---|
| Same label | near duplicate: keep the first, drop the copy (`results/removed_duplicates.csv`) |
| Different label | label conflict: **nothing is deleted**. The whole group goes to `results/conflicts_to_review.csv` (group, file, label, matches, distance, dhash_distance) and stays out of every split until a person checks it |

There is no chaining. An image joins a group only if it matches the group's
kept copy directly. A final pass also holds any kept image that directly matches
an already removed or held image with a different label, so file order cannot
hide a conflict.

| 4 classes, from 8,578 found | pHash ≤ 5, chained, conflicts deleted | Now |
|---|---:|---:|
| Exact duplicates removed | 2 | 2 |
| Near duplicates removed | 125 | 57 |
| Label-conflict images | 48 deleted (10 groups) | 0 held (0 groups) |
| Corrupted | 0 | 0 |
| **In the splits** | **8,403** | **8,519** |

The new rules keep **116 more images** than the old pHash ≤ 5 setting.
`results/conflicts_to_review.csv` is header-only today because no cross-class
pair passes both checks. Checking pHash ≤ 3 alone, without dHash, would flag
6 conflict groups with 16 images. All of them are look-alikes, not copies (see
the cross-label review below), and requiring dHash agreement clears them all.

In 3-class mode the script finds 7,599 images, removes 2 exact and 52 near duplicates with no conflicts, and keeps 7,545.
Every run ends with a count check: images found minus removed minus held for
review must equal the images in the splits, and each CSV must have one row per image.
Count gap fixed: the "175 removed but only 173 fewer" mismatch came from comparing 8,403 against 8,576, the split total from `f238069`, which had already dropped the same 2 exact duplicates. The true start was 8,578, and the count check added in `1421f23` now stops any run whose numbers do not add up.

The 57 near duplicates are all within one folder: green-glass 13, metal 8,
biological 8, trash 7, paper 7, cardboard 6, battery 5, white-glass 2, plastic 1.
Their distances are pHash 0 or 2 and dHash 0 to 2. Every pHash from `imagehash`
has exactly 32 one-bits, so the distance between two pHashes is always even.
That means `max_phash: 3` acts like 2; set it to 4 if the team wants it looser.

### Spot check of the borderline pairs

`python src/data/review_near_dups.py` draws the 20 removed pairs with the
highest distances side by side in `results/near_dup_review.png` (not committed).
I counted a pair as a true copy only if it is the same photograph, possibly
resized, re-saved, watermarked or with its background swapped.

| dHash cutoff | Near duplicates removed | True copies in top 20 |
|---|---:|---:|
| 4 | 71 | 15 |
| 3 | 66 | 15 |
| 2 (final) | 57 | 14 |

The target was 18 of 20, with at most two tightenings, so the cutoff stopped at 2.
At that cutoff, the 6 pairs that are not true copies are all same-class:

- **Toothbrushes in `trash`:** 4 pairs are different toothbrushes on white backgrounds, including a gold and a pink version of the same model.
- **Cardboard:** 1 pair is two shots of the same cardboard sheet.
- **Wine glasses:** 1 pair is two similar wine glasses.

No cross-class pair passes both checks, so no image is held as a label conflict.
Toothbrushes are hard for both fingerprints because almost the whole picture is
white background. A third fingerprint or a smaller pHash cutoff would
be the next step if the team wants to keep those few images.

### Cross-label review and the final `max_dhash`

`python src/data/review_near_dups.py --cross-label --max-dhash 64` turns the
dHash check off and draws every label-conflict group in
`results/cross_label_review.png` (not committed). With pHash ≤ 3 alone there are
6 groups with 16 images:

| Group | Labels | What it really is | dHash distance across labels |
|---|---|---|---:|
| G001 | hazardous vs trash | Duracell battery vs blue electric toothbrush | 15 |
| G002 | hazardous vs trash | Duracell AA on beige vs black toothbrush | 20 |
| G003 | recycling vs trash | two different brown beer bottles vs a toothbrush head and a white toothbrush | 12-14 |
| G004 | recycling vs trash | cardboard edge vs bamboo toothbrush | 17 |
| G005 | recycling vs trash | green olive-oil bottle vs two electric toothbrushes | 14 |
| G006 | recycling vs trash | brown pump bottle vs two toothbrushes | 8-18 |

**0 of the 16 images are true cross-label copies.** Every group is a thin,
upright object on a white background matched to a toothbrush. Since none are
real copies, there was nothing to loosen for. The smallest cross-label dHash
distance is 8, far above the cutoff.

**Final `max_dhash: 2`, and why:** the borderline same-label spot check above
tightened it from 4 to 2 to cut look-alike removals. This cross-label check
found no true copy that dHash ≤ 2 would miss, and every look-alike sits at
dHash 8 or more, so 2 keeps all of them out of the review list.

## Train / validation / test splits (seed 42, 70/15/15)

Output of `python src/data/build_splits.py` on 2026-10-18, after duplicate removal.
Weights come from the train split only. Inverse frequency is
`total_train / (num_classes * class_count)`. Square-root inverse frequency takes
the square root of that and divides by the mean, so the weights average 1.

**`num_classes: 4`**

| Class | Train | Validation | Test | Total | % of train | Inverse weight | Sqrt-inverse weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compost | 718 | 154 | 155 | 1,027 | 12.04% | 2.0756 | 1.0983 |
| hazardous | 681 | 146 | 147 | 974 | 11.42% | 2.1883 | 1.1277 |
| recycling | 4,059 | 869 | 871 | 5,799 | **68.09%** | 0.3671 | 0.4619 |
| trash | 503 | 107 | 109 | 719 | 8.44% | 2.9627 | 1.3121 |
| **Total** | **5,961** | **1,276** | **1,282** | **8,519** | 100% | | |

**`num_classes: 3`** (hazardous excluded)

| Class | Train | Validation | Test | Total | % of train | Inverse weight | Sqrt-inverse weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compost | 718 | 154 | 155 | 1,027 | 13.60% | 2.4513 | 1.1471 |
| recycling | 4,059 | 869 | 871 | 5,799 | **76.88%** | 0.4336 | 0.4824 |
| trash | 503 | 107 | 109 | 719 | 9.53% | 3.4990 | 1.3705 |
| **Total** | **5,280** | **1,130** | **1,135** | **7,545** | 100% | | |

All runs passed the overlap check: no exact or near duplicate appears in two
splits. Running 4, then 3, then 4 again gave byte-identical 4-class outputs.

## Class imbalance to watch

- `recycling` is 68.09% of train with 4 classes and 76.88% with 3. Always guessing recycling scores 68.10% validation accuracy but only 0.2026 macro F1 (see `docs/data_pipeline.md`), so report macro F1.
- `clothes` (5,286 images) is dropped by team decision. `shoes` (2,006) is skipped until the team decides.
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

1. **Shoes.** San Diego curbside recycling does not accept them. Map the 2,006 shoe images to `trash`, or drop them like clothes? They stay VERIFY until decided.

Decided on Oct 18: 4 classes including `hazardous`, and clothes dropped (`EXCLUDE`).
