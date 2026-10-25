# Dataset Inventory

Owner: Isiah (data side). Last updated: 2026-10-24 (wild datasets TACO and RealWaste added).

Images are **never committed**. They download into `data/raw/` (git-ignored) with
`python src/data/download.py`. Kaggle licenses are copied from each Kaggle page on
2026-10-17; TACO and RealWaste licenses are in "License notes" below (checked 2026-10-24).
Check the source again before publishing anything.

## Summary

| Name | Source | License (as listed at the source) | Images | Classes | Clean or real-world | Battery images | Status |
|---|---|---|---:|---|---|---:|---|
| `garbage12_enhanced` | [huberthamelin/garbage-classification-labels-corrections](https://www.kaggle.com/datasets/huberthamelin/garbage-classification-labels-corrections) | Database: Open Database, Contents: Database Contents | 15,870 (counted locally) | 12: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass | Clean. Mostly web-scraped, centered objects | **979** (counted locally) | **Primary. Downloaded and mapped.** |
| `garbage12` | [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | Database: Open Database, Contents: © Original Authors | 15,150 per page text; its folders add up to 15,515 | Same 12 classes | Clean. Mostly web-scraped | 945 (Kaggle page) | Listed only. Not downloaded. |
| `garbage_v2` | [sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) | MIT | 13,348 (Kaggle page) | 10: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash | Mixed. Phone-app photos plus curated web images | 756 (Kaggle page) | Listed only. Not downloaded. |
| `personal_battery` | none, our own photos | Ours. Team decides before publishing | 0 today | 1: hazardous | **Real-world** | 0 today, target 50 | Folder and photo rules ready. |
| `taco` / `taco_crops` | [github.com/pedropro/TACO](https://github.com/pedropro/TACO) (not Kaggle) | CC BY 4.0 per dataset listings; see License notes | 1,500 photos, 4,784 objects, 60 categories; 1,665 crops cut, 990 used after the recycling cap | 14 categories to recycling, 8 to trash (after the quality check) | **Real-world** litter in context | 0 | **Wild dataset. Downloaded, cropped, mapped.** |
| `realwaste` | [UCI dataset 908](https://archive.ics.uci.edu/dataset/908/realwaste) (not Kaggle) | CC BY 4.0 | 4,752 | 9 folders; Food Organics to compost (after the quality check) | **Real-world** landfill items | 0 | **Wild dataset. Downloaded and mapped.** |

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
| Known duplicates (leakage check, `known_duplicates`) | 0 | 17 |
| **In the splits** | **8,403** | **8,502** |

The new rules keep **99 more images** than the old pHash ≤ 5 setting: 116 more
from the stricter duplicate rule, minus the 17 leaked copies dropped later.
`results/conflicts_to_review.csv` is header-only today because no cross-class
pair passes both checks. Checking pHash ≤ 3 alone, without dHash, would flag
6 conflict groups with 16 images. All of them are look-alikes, not copies (see
the cross-label review below), and requiring dHash agreement clears them all.

In 3-class mode the script finds 7,599 images, removes 2 exact, 52 near and 15 known duplicates with no conflicts, and keeps 7,530. Two of the 17 known duplicates are batteries, which 3-class mode leaves out anyway.
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

**0 of the 16 images are true cross-label copies.** Every group pairs a
toothbrush with a different object of a similar outline on a plain background:
batteries, bottles, and in one case a cardboard edge. Since none are
real copies, there was nothing to loosen for. The smallest cross-label dHash
distance is 8, far above the cutoff.

**Final `max_dhash: 2`, and why:** the borderline same-label spot check above
tightened it from 4 to 2 to cut look-alike removals. This cross-label check
found no true copy that dHash ≤ 2 would miss, and every look-alike sits at
dHash 8 or more, so 2 keeps all of them out of the review list.

## Leakage check (`check_leakage.py`, 2026-10-18)

**Why it matters:** if nearly the same photo is in train and in test, the model
has already seen that test answer, so its test score looks better than it really is.

`python src/data/check_leakage.py` is read-only. For every validation and test
image it finds the closest train image, using the same pHash and dHash functions
as `build_splits.py`. It flags the pair if **pHash ≤ 6 and dHash ≤ 8**, which is
looser than the removal rule, so borderline copies that slipped through show
up. Every flagged pair is in `results/leakage_pairs.csv`, and
`results/leakage_review.png` shows them side by side (not committed).

Results on the splits **before** the 17 copies were removed:

| Split | Images | Flagged | True copies | Look-alikes |
|---|---:|---:|---:|---:|
| validation | 1,276 | 10 | 5 | 5 |
| test | 1,282 | 22 | 12 | 10 |
| **both** | **2,558** | **32** | **17** | **15** |

All 32 pairs were checked by eye, which covers every pair at dHash 3 to 5. A true
copy is the same photograph, possibly resized, re-saved, watermarked or slightly
cropped. A look-alike is a different photo: another toothbrush or laptop battery,
a recolored can or yogurt cup, a different bottle, or the same crushed can shot again.

| dHash distance | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flagged | 1 | 3 | 4 | 7 | 4 | 3 | 4 | 5 | 1 | 32 |
| True copies | 1 | 2 | 1 | 5 | 2 | 3 | 1 | 2 | 0 | 17 |
| Look-alikes | 0 | 1 | 3 | 2 | 2 | 0 | 3 | 3 | 1 | 15 |

The 17 true copies (numbers are the row order in `results/leakage_pairs.csv`):

| # | Split | Image | Train copy | pHash | dHash |
|---:|---|---|---|---:|---:|
| 1 | test | paper__paper886.jpg | paper__paper755.jpg | 4 | 0 |
| 2 | test | paper__paper551.jpg | paper__paper679.jpg | 4 | 1 |
| 4 | validation | metal__metal439.jpg | metal__metal376.jpg | 6 | 1 |
| 5 | test | metal__metal408.jpg | metal__metal40.jpg | 4 | 2 |
| 9 | test | biological__biological369.jpg | biological__biological823.jpg | 2 | 3 |
| 10 | test | battery__battery128.jpg | battery__battery777.jpg | 2 | 3 |
| 13 | test | trash__trash8.jpg | trash__trash15.jpg | 2 | 3 |
| 14 | validation | green-glass__green-glass210.jpg | green-glass__green-glass115.jpg | 2 | 3 |
| 15 | validation | white-glass__white-glass343.jpg | white-glass__white-glass456.jpg | 4 | 3 |
| 16 | test | paper__paper812.jpg | paper__paper517.jpg | 2 | 4 |
| 17 | test | battery__battery104.jpg | battery__battery168.jpg | 4 | 4 |
| 20 | test | green-glass__green-glass581.jpg | green-glass__green-glass595.jpg | 2 | 5 |
| 21 | test | paper__paper982.jpg | paper__paper716.jpg | 2 | 5 |
| 22 | validation | trash__trash20.jpg | trash__trash12.jpg | 2 | 5 |
| 26 | validation | metal__metal164.jpg | metal__metal27.jpg | 6 | 6 |
| 27 | test | biological__biological981.jpg | biological__biological723.jpg | 4 | 7 |
| 28 | test | paper__paper1011.jpg | paper__paper892.jpg | 4 | 7 |

The true copies that slipped through sit at pHash 2, 4 or 6. The 4 copies at
dHash 0 to 2 have pHash 4 or 6, above `max_phash: 3`. The other 13 have dHash 3
to 7, above `max_dhash: 2`. Catching all 17 would need pHash ≤ 6 and dHash ≤ 7.
That breaks the pHash cap of 5, and at those cutoffs 14 of the look-alikes match too.

**Decision: no rule change.** 17 true copies are 0.66% of validation plus test,
under the 1% line (26 images), so the split rules stay as they are:
**`max_phash: 3`, `max_dhash: 2`**. The cross-label conflicts file stays empty.

**The 17 confirmed copies are now removed through `known_duplicates` in `configs/label_map.yaml`.**
Each validation/test copy is dropped and its train copy kept. The drop happens
right after the split is assigned, so no other image changes split. Train is
unchanged (same 5,961 images, same class weights), and validation and test lose
5 and 12 images. A rerun of `check_leakage.py` flags 15 pairs, all of them the
look-alikes judged above, and 0 true copies.

## Train / validation / test splits (seed 42, 70/15/15)

Output of `python src/data/build_splits.py` on 2026-10-18, after duplicate removal and the 17 known duplicates.
The Kaggle set is split on its own (`base_datasets:` in `configs/label_map.yaml`), so adding Amy's or personal photos never moves a Kaggle image to another split; new photos are placed around it, look-alikes together (see `docs/data_pipeline.md`).
Weights come from the train split only. Inverse frequency is
`total_train / (num_classes * class_count)`. Square-root inverse frequency takes
the square root of that and divides by the mean, so the weights average 1.

**`num_classes: 4`**

| Class | Train | Validation | Test | Total | % of train | Inverse weight | Sqrt-inverse weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compost | 718 | 154 | 153 | 1,025 | 12.04% | 2.0756 | 1.0983 |
| hazardous | 681 | 146 | 145 | 972 | 11.42% | 2.1883 | 1.1277 |
| recycling | 4,059 | 865 | 864 | 5,788 | **68.09%** | 0.3671 | 0.4619 |
| trash | 503 | 106 | 108 | 717 | 8.44% | 2.9627 | 1.3121 |
| **Total** | **5,961** | **1,271** | **1,270** | **8,502** | 100% | | |

**`num_classes: 3`** (hazardous excluded)

| Class | Train | Validation | Test | Total | % of train | Inverse weight | Sqrt-inverse weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| compost | 718 | 154 | 153 | 1,025 | 13.60% | 2.4513 | 1.1471 |
| recycling | 4,059 | 865 | 864 | 5,788 | **76.88%** | 0.4336 | 0.4824 |
| trash | 503 | 106 | 108 | 717 | 9.53% | 3.4990 | 1.3705 |
| **Total** | **5,280** | **1,125** | **1,125** | **7,530** | 100% | | |

All runs passed the overlap check: no exact or near duplicate appears in two
splits. Running 4, then 3, then 4 again gave byte-identical 4-class outputs.

## Wild datasets: TACO and RealWaste (2026-10-24)

Two open datasets add messy, real-world photos next to the clean Kaggle images:
TACO (litter photographed where it lies) and RealWaste (items picked from a
landfill, photographed on concrete). They never enter the benchmark test set; they
are split on their own into train, validation and `data/splits/wild_test.csv`.

### San Diego source used for the mapping

**City of San Diego Environmental Services, "What Goes Where" guide** (English, April 2026 edition):
https://www.sandiego.gov/sites/default/files/what_goes_where_guide_english.pdf
(linked from the curbside page https://www.sandiego.gov/environmental-services/recycling/residential/curbside).

| Bin | What the guide lists |
|---|---|
| Blue (recycling) | Aluminum and steel cans; glass bottles and jars; cardboard (empty, flattened); paper cartons; plastic bottles, containers, lids, hard plastics, toys and Styrofoam packaging; mixed paper. "All recyclables empty, dry and loose." The Spanish text also lists plastic cups and tubs, and aluminum foil and trays, with recycling. |
| Green (organics) | Food scraps; food-soiled paper (paper towels, napkins, paper bags, parchment paper, coffee filters); plant and yard trimmings. NO plastic bags, glass, food packaging, plant pots, dirt or rocks. |
| Black (trash) | Hygiene products, diapers, pet waste; dishes and glassware; paper plates, cups and takeout boxes; tissues and disposable wipes; plastic bags, wrappers and film; plastic straws and utensils; anything labeled compostable or biodegradable. "When in doubt, throw it out." |
| None (hazardous drop-off) | Batteries, toxic products, electronics, paint, bulbs, motor oil, needles, propane and helium tanks, chemical cleaners. |

Rules used, in order:

1. The guide puts the item in one bin: that class.
2. The guide leaves it unclear (two bins fit, or it is not listed and could be recyclable): EXCLUDE.
   The team list is EXCLUDE regardless: **pizza boxes, paper cups, aerosol cans, plastic bags and tissues.**
   Note: this guide edition does name a bin for three of them (paper cups, plastic bags and film,
   and tissues all go in black trash), so they could move to trash on a team decision.
3. Not on the blue or green list, not hazardous, and plainly neither recyclable nor organic:
   trash, the guide's own default ("When in doubt, throw it out"). Used for cigarettes, gloves, rope and six-pack rings.

Hazardous is left alone (it already scores 0.99 recall), so nothing maps to it.

### TACO: all 60 categories from the official annotations file

Category names come from `data/annotations.json` in the TACO repository, pinned to commit
`3844bcd` (2026-10-20) and checked by sha256 in `download.py`. `src/data/crop_taco.py` cuts each
object of a mapped category out of its photo (turned upright by EXIF first, because TACO's boxes
are drawn on the upright photo), with 10% padding on each side, and skips crops under 64 px on
either side (most cigarettes: 543 of 667). The mapping lives in `configs/label_map.yaml`
(`taco_crops: categories:`).

| TACO category | Objects | Class | Why (City of San Diego guide, April 2026) |
|---|---:|---|---|
| Clear plastic bottle | 285 | **recycling** | blue: "Plastic Bottles" |
| Corrugated carton | 64 | **recycling** | blue: "Cardboard" |
| Disposable food container | 38 | **recycling** | blue: "Plastic Bottles, Containers" (Spanish: "envases", "charolas" de plastico) |
| Disposable plastic cup | 104 | **recycling** | blue: Spanish text lists plastic cups ("tazas ... de plastico") with bottles and containers |
| Drink can | 229 | **recycling** | blue: "Aluminum and Steel Cans" |
| Drink carton | 45 | **recycling** | blue: "Paper Cartons" |
| Food Can | 34 | **recycling** | blue: "Aluminum and Steel Cans" |
| Glass bottle | 104 | **recycling** | blue: "Glass Bottles and Jars" |
| Glass jar | 6 | **recycling** | blue: "Glass Bottles and Jars" |
| Magazine paper | 12 | **recycling** | blue: "Mixed Paper" |
| Other plastic bottle | 50 | **recycling** | blue: "Plastic Bottles" |
| Other plastic cup | 2 | **recycling** | same as Disposable plastic cup |
| Plastic bottle cap | 209 | **recycling** | blue: "Plastic Bottles, Containers, Lids" |
| Toilet tube | 5 | **recycling** | blue: "Cardboard" / "Mixed Paper" |
| Cigarette | 667 | **trash** | not on the blue or green list, not hazardous; rule 3 |
| Crisp packet | 39 | **trash** | black: wrappers |
| Other plastic wrapper | 260 | **trash** | black: "Plastic Bags, Wrappers and Film" (wrappers) |
| Plastic glooves | 4 | **trash** | not on the blue or green list, not hazardous; rule 3 |
| Plastic straw | 157 | **trash** | black: "Plastic Straws and Utensils" |
| Plastic utensils | 37 | **trash** | black: "Plastic Straws and Utensils" |
| Rope & strings | 29 | **trash** | not on the blue or green list, not hazardous; rule 3 (the curbside web page also lists rope as a "tangler" that must not go in blue) |
| Six pack rings | 5 | **trash** | soft plastic, not a bottle, container or hard plastic; black: "Wrappers and Film" is the closest line; rule 3 |
| Aerosol | 10 | EXCLUDE | team list; not in the guide, and a full can is hazardous waste |
| Aluminium blister pack | 6 | EXCLUDE | not in the guide; foil and plastic mixed |
| Aluminium foil | 62 | EXCLUDE | guide unclear: English layout lists foil and trays with trash, Spanish text lists them with recycling |
| Battery | 2 | EXCLUDE | hazardous; hazardous is left alone this round |
| Broken glass | 138 | EXCLUDE | unclear: bottles and jars are blue, dishes and glassware are trash |
| Carded blister pack | 1 | EXCLUDE | not in the guide; card and plastic mixed |
| Egg carton | 11 | EXCLUDE | not in the guide; made of paper, foam or plastic |
| Foam cup | 13 | EXCLUDE | unclear: "Styrofoam Packaging" is blue, cups and takeout are trash |
| Foam food container | 15 | EXCLUDE | unclear: polystyrene containers are blue, takeout boxes are trash |
| Food waste | 8 | EXCLUDE | was compost; dropped at the quality check: 2 of 5 showed food inside a can or a plastic bag |
| Garbage bag | 31 | EXCLUDE | team list (plastic bags) |
| Glass cup | 6 | EXCLUDE | was trash; dropped at the quality check: 2 of 6 look like plastic cups (blue) |
| Meal carton | 30 | EXCLUDE | was trash; dropped at the quality check: 4 of 30 were a paper cup, a drink carton, a plastic piece and a corrugated sleeve |
| Metal bottle cap | 80 | EXCLUDE | not in the guide (only cans are listed for metal) |
| Metal lid | 10 | EXCLUDE | not in the guide (only cans are listed for metal) |
| Normal paper | 82 | EXCLUDE | was recycling; dropped at the quality check: 4 of 30 look like tissues or napkins (green or excluded) |
| Other carton | 93 | EXCLUDE | was recycling; dropped at the quality check: 6 of 15 were cigarette packs (foil-lined, not clearly blue) |
| Other plastic | 273 | EXCLUDE | unidentified plastic: could be hard plastic (blue) or film (trash) |
| Other plastic container | 6 | EXCLUDE | was recycling; dropped at the quality check: 2 of 5 were film or foil |
| Paper bag | 27 | EXCLUDE | unclear: the guide lists paper bags under green food-soiled paper; clean paper is blue |
| Paper cup | 67 | EXCLUDE | team list; the guide puts paper cups in black trash |
| Paper straw | 4 | EXCLUDE | not in the guide; "compostable" products are trash, plain paper is unclear |
| Pizza box | 3 | EXCLUDE | team list; not in the guide (city web page: soiled part trash, clean part blue) |
| Plastic film | 451 | EXCLUDE | team list (plastic bags): mostly bag pieces; the guide puts film in black trash |
| Plastic lid | 77 | EXCLUDE | was recycling; dropped at the quality check: 7 of 45 show the lid on a paper coffee cup (paper cups are excluded) |
| Plastified paper bag | 0 | EXCLUDE | not in the guide; paper and plastic mixed |
| Polypropylene bag | 3 | EXCLUDE | team list (plastic bags) |
| Pop tab | 99 | EXCLUDE | not in the guide (only cans are listed for metal) |
| Scrap metal | 20 | EXCLUDE | not in the guide (only cans are listed for metal) |
| Shoe | 7 | EXCLUDE | shoes are still VERIFY for the team (see garbage12_enhanced) |
| Single-use carrier bag | 61 | EXCLUDE | team list (plastic bags) |
| Spread tub | 9 | EXCLUDE | was recycling; dropped at the quality check: 2 of 9 were a foil tub or unidentifiable |
| Squeezable tube | 7 | EXCLUDE | not in the guide; tubes are mixed plastic and metal |
| Styrofoam piece | 112 | EXCLUDE | unclear: foam packaging is blue, but loose broken pieces are not packaging |
| Tissues | 42 | EXCLUDE | team list; the guide puts tissues in black trash |
| Tupperware | 4 | EXCLUDE | was recycling; dropped at the quality check: 2 of 4 were plastic film or bags |
| Unlabeled litter | 517 | EXCLUDE | no category: could be anything |
| Wrapping paper | 12 | EXCLUDE | unclear: mixed paper is blue, but gift wrap is often foil or plastic coated |

### RealWaste: the starting mapping, checked against the guide

| RealWaste folder | Images | Started as | Final | Why |
|---|---:|---|---|---|
| Cardboard | 461 | recycling | EXCLUDE | was recycling; dropped at the quality check: 4 of 15 wrong (pizza box, packing straw, foil-lined coffee bag, plastic pouch) |
| Glass | 420 | recycling | EXCLUDE | was recycling; dropped at the quality check: 8 of 15 were broken glass or a glass dish (not bottles or jars) |
| Metal | 790 | recycling | EXCLUDE | was recycling; dropped at the quality check: 4 of 15 were aerosol cans (team list) and 1 crumpled foil |
| Food Organics | 411 | compost | **compost** | green: "Food Scraps" |
| Vegetation | 436 | compost | EXCLUDE | was compost; dropped at the quality check: 8 of 45 were piles of dirt (the guide: no dirt in the green bin) |
| Miscellaneous Trash | 495 | trash | EXCLUDE | was trash; dropped at the quality check: 5 of 45 belong elsewhere (notebook, hard plastic tubs, plant matter, string lights) |
| Paper | 500 | EXCLUDE | EXCLUDE | landfill paper is often soiled; soiled paper is green, clean is blue |
| Plastic | 921 | EXCLUDE | EXCLUDE | mixes rigid plastic (blue) with bags and film (trash) |
| Textile Trash | 318 | EXCLUDE | EXCLUDE | clothes are not blue; textiles left out, like the Kaggle clothes folder |

RealWaste photographed each item several times in a row, moved between shots, so the image
fingerprints do not see them as copies (files 373 and 374 of Food Organics: pHash 20, dHash 32).
The files are numbered in shooting order, so `build_splits.py` keeps blocks of 20 consecutive
numbers per folder in one split (`group_block: 20`). A burst that crosses a multiple of 20 can
still be split; this is the remaining leakage risk for the wild test.

### Quality check

Every mapped category was checked by eye on 15 random images; borderline ones got 30 to 45. A
category with more than 10% of images that belong in another bin was dropped (EXCLUDE, with the
count in its `label_map.yaml` comment):

| Dropped | Source | Wrong | What was wrong |
|---|---|---:|---|
| Cardboard | RealWaste | 4 of 15 | pizza box, packing straw, foil-lined coffee bag, plastic pouch |
| Glass | RealWaste | 8 of 15 | broken glass and a glass dish, not bottles or jars |
| Metal | RealWaste | 5 of 15 | aerosol cans (team list) and crumpled foil |
| Vegetation | RealWaste | 8 of 45 | piles of dirt (no dirt in the green bin) |
| Miscellaneous Trash | RealWaste | 5 of 45 | notebook, hard plastic tubs, plant matter, string lights |
| Other carton | TACO | 6 of 15 | cigarette packs (foil-lined) |
| Plastic lid | TACO | 7 of 45 | lid shown on a paper coffee cup |
| Normal paper | TACO | 4 of 30 | tissues or napkins |
| Meal carton | TACO | 4 of 30 | paper cup, drink carton, plastic piece, corrugated sleeve |
| Food waste | TACO | 2 of 5 | food inside a can or a plastic bag |
| Glass cup | TACO | 2 of 6 | plastic cups |
| Tupperware | TACO | 2 of 4 | plastic film or bags |
| Other plastic container | TACO | 2 of 5 | film or foil |
| Spread tub | TACO | 2 of 9 | a foil tub, unidentifiable pieces |

Kept after a larger sample: Plastic straw (3 of 45, 7%, show a cup or lid). Every other mapped
category had at most 1 wrong in 15.

Final check on what was built, 30 random new images per class (`python src/data/review_new_images.py`,
sheets in `results/new_images_<class>.png`, not committed):

| Class | Fit the class | Look wrong | Too blurry or cropped to use |
|---|---:|---:|---:|
| recycling | 28 | 0 | 2 |
| compost | 30 | 0 | 0 |
| trash | 25 | 1 (a straw crop dominated by a plastic cup lid) | 4 |

### Recycling cap

Recycling was 68% of train, so new recycling images are capped at **400** in total, shared evenly
by the categories (a small category keeps all its images) and drawn at random with seed 42
(`wild_split: recycling_cap` in `label_map.yaml`). 1,075 TACO recycling crops, 400 kept:

| TACO recycling category | Crops | Kept |
|---|---:|---:|
| clear-plastic-bottle | 269 | 39 |
| corrugated-carton | 64 | 38 |
| disposable-food-container | 37 | 37 |
| disposable-plastic-cup | 102 | 38 |
| drink-can | 214 | 38 |
| drink-carton | 43 | 38 |
| food-can | 34 | 34 |
| glass-bottle | 100 | 38 |
| glass-jar | 6 | 6 |
| magazine-paper | 12 | 12 |
| other-plastic-bottle | 49 | 38 |
| other-plastic-cup | 2 | 2 |
| plastic-bottle-cap | 139 | 38 |
| toilet-tube | 4 | 4 |


No cap on compost or trash. Hazardous is untouched.

### What the wild datasets added (build of 2026-10-24)

| Source | Class | Train | Validation | Wild test | Total |
|---|---|---:|---:|---:|---:|
| realwaste | compost | 292 | 59 | 60 | 411 |
| taco_crops | recycling | 280 | 60 | 60 | 400 |
| taco_crops | trash | 413 | 88 | 89 | 590 |
| **Total** | | **985** | **207** | **209** | **1,401** |

Duplicates: 0 new images were copies of a Kaggle image, 0 were copies of each other, 0 groups
looked like a benchmark image, and 0 wild test images looked like a train or benchmark image
(pHash <= 6 and dHash <= 8). `check_leakage.py` on this build (6,946 train against 1,478 validation,
1,270 test and 209 wild test images) flags 0 wild test pairs and the same 15 judged Kaggle look-alikes as before.

Train class shares, before and after:

| Class | Train before | Share before | Train after | Share after |
|---|---:|---:|---:|---:|
| compost | 718 | 12.04% | 1,010 | 14.54% |
| hazardous | 681 | 11.42% | 681 | 9.80% |
| recycling | 4,059 | 68.09% | 4,339 | 62.47% |
| trash | 503 | 8.44% | 916 | 13.19% |
| **Total** | **5,961** | | **6,946** | |

### Frozen benchmark

`data/splits/test.csv` is frozen as the benchmark: the 1,270 Kaggle test images of 2026-10-24,
sha256 `44805ae865a07e6315bf44e9b240c406a320d9fa0ad063a2282de354aad9dd42`
(`benchmark:` in `label_map.yaml`). `build_splits.py` stops if it changes by one byte. Every Kaggle
image keeps its split; new sources can never remove, move or join a Kaggle image (a new copy of a
Kaggle image is dropped, or held for review if its label differs, and the Kaggle image stays).
The swap rule for models is in `docs/evaluation.md`.

## Class imbalance to watch

- `recycling` was 68.09% of train with 4 classes (76.88% with 3) before the wild datasets, and is 62.47% after them (recycling cap of 400). Always guessing recycling scores 62.58% validation accuracy but only 0.1925 macro F1 on this build's 1,478 validation images (68.06% and 0.2025 before the wild datasets; see `docs/data_pipeline.md`), so report macro F1.
- `clothes` (5,286 images) is dropped by team decision. `shoes` (2,006) is skipped until the team decides.
- `hazardous` comes from web photos only today. The 50 personal battery photos add real-world examples.

## License notes

- `garbage12_enhanced` is ODbL/DbCL as a database, but its original author says the web-scraped images belong to their photographers. That is fine for a class project and for training. Do not redistribute the images, which matches our "no images in git" rule.
- `garbage_v2` is MIT, the most permissive of the three.
- **RealWaste: CC BY 4.0**, stated on the UCI page (https://archive.ics.uci.edu/dataset/908/realwaste).
  The IEEE DataPort copy lists CC BY-NC-SA 4.0; we use the UCI copy. Attribution required. Cite:
  - Dataset: Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G
  - Paper: Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information*, 14(12), 633. https://doi.org/10.3390/info14120633
- **TACO: CC BY 4.0 as listed by the dataset indexes** (Dataset Ninja, Roboflow Universe). The
  official sources are less explicit: the repository's LICENSE file is MIT and covers the code, the
  annotations file has an empty license list, and the paper says "all images are under free
  copyright licences". We only train on it and never redistribute images. Attribution as the
  authors ask (README of https://github.com/pedropro/TACO):
  ```bibtex
  @article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proen{\c{c}}a and Pedro Sim{\~o}es},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
  }
  ```

## Other candidates looked at, not used

| Kaggle URL | License | Why not |
|---|---|---|
| [akshat103/e-waste-image-dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset) | Apache 2.0 | Electronics, not batteries. Revisit if hazardous grows to cover e-waste. |
| [karansolanki01/garbage-classification](https://www.kaggle.com/datasets/karansolanki01/garbage-classification) | Apache 2.0 | 775 battery images that likely repeat `garbage12`. |
| [harshadsgore/e-waste-image-classification-dataset-18-classes](https://www.kaggle.com/datasets/harshadsgore/e-waste-image-classification-dataset-18-classes) | Other (specified in description) | Page says no blanket license for the images. Avoid. |

## Open questions for the team

1. **Shoes.** San Diego curbside recycling does not accept them. Map the 2,006 shoe images to `trash`, or drop them like clothes? They stay VERIFY until decided.

Decided on Oct 18: 4 classes including `hazardous`, and clothes dropped (`EXCLUDE`).
