# Personal battery photos (`data/raw/personal_battery/`)

These are photos **we take ourselves** of household batteries, so the model
sees real-world shots instead of only clean web images. Public datasets have
only web-style battery photos (979 in garbage12_enhanced), and batteries are the
main reason we are proposing a 4th `hazardous` class. (San Diego: batteries never go in any curbside bin.)

**Images in this folder are never committed to git.** Only this README is
tracked. Share the photos with the team through Drive or a zip, then drop them
in here locally before running `build_splits.py`.

## Folder layout

Put every photo directly in this folder (no sub-folders). `build_splits.py`
treats the whole folder as one class and maps it with the
`personal_battery` entry in `configs/label_map.yaml`.

```
data/raw/personal_battery/
├── README.md          <- this file (tracked)
├── aa_kitchen_top_01.jpg
├── aa_kitchen_top_02.jpg
├── 9v_desk_side_01.jpg
└── ...
```

## File naming

```
<type>_<background>_<angle>_<nn>.jpg     (or .png)
```

| Part         | Allowed values                                                     |
|--------------|--------------------------------------------------------------------|
| `type`       | `aa`, `aaa`, `9v`, `coin` (coin / button cell), `phone` (phone or laptop lithium pack), `d`, `c`, `other` |
| `background` | `kitchen`, `desk`, `floor`, `outdoor`, `hand`, `bin`, `cardboard`, `other` |
| `angle`      | `top` (looking straight down), `side`, `tilt` (about 45 degrees), `close` (macro) |
| `nn`         | two-digit counter: `01`, `02`, ...                                 |

Example: `coin_hand_close_03.jpg` is the third close-up of a coin cell held in a hand.

Lowercase, underscores only, `.jpg` or `.png`. No spaces.

## Photo rules

**Target: 50 photos total**, about 10 per battery type (AA, AAA, 9V, coin cell,
phone battery). Spread each type across the backgrounds, lighting and angles
below so no single look dominates.

1. **One battery (or one small pile of the same type) per photo.** The model
   classifies one item at a time.
2. **Battery types to cover:** AA, AAA, 9V, coin cell, phone battery. Add C, D
   or rechargeable packs if you have them.
3. **Backgrounds:** at least 4 different ones (counter, desk, floor, outdoors,
   hand, inside a bin, on cardboard). Avoid the same white table every time.
4. **Lighting:** mix daylight, warm indoor light, dim light and some harsh
   shadows. Do not use flash for every shot.
5. **Angles:** top-down, side, tilted, and a few close-ups. Include a few where
   the battery is partly covered or lying with other non-battery items nearby.
6. **Condition:** include used, scuffed, corroded or taped batteries, not only
   new ones. Include some with the label facing away.
7. **Camera:** phone camera is fine. Try to also take some shots with the
   Raspberry Pi camera once it is set up, since that is what the demo uses.
8. **Framing:** the battery should fill roughly 20 to 80 percent of the frame.
   No extreme zoom-outs where it is a dot.
9. **Privacy:** no faces, no addresses, no personal documents in the frame.

## Before sharing

- Quick scan for blurry or duplicate shots and delete them.
- Count per type so we can report class balance (`ls *.jpg *.png 2>/dev/null | cut -d_ -f1 | sort | uniq -c`).
- Note the date and phone model in the team doc.
