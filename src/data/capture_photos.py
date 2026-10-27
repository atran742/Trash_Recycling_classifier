"""
capture_photos.py - take labeled photos with the webcam, straight into the right folder.

It opens the webcam and shows the live feed with the same center box as the demo
(src/webcam_smooth.py), so you hold each item where the demo will look for it.
Press a number key to save the current frame:

    1 = compost    2 = hazardous    3 = recycling    4 = trash    q = quit

--dest picks the folder, and there is no default, so you always choose on purpose:

    camera_test  -> data/raw/camera_test/<class>/   demo-room photos, NEVER trained on
    team_photos  -> data/raw/team_photos/<class>/   our own photos, they join train/validation

The FULL frame is saved as a JPG named <class>_<date>_<time>.jpg, for example
trash_20261027_163005.jpg (with _2, _3 ... if you save twice in one second). The box
is only drawn on the screen, never on the saved photo. After each save the class name
and how many photos of each class you saved this run show on screen for 1 second.

One photo must never land in both folders: a camera_test photo that is also trained
on would make the camera test score too kind. --dest is fixed for a whole run, so the
check looks at the OTHER folder on disk, which holds every earlier session's photos:
  - with --dest team_photos, a frame that looks the same as any photo in
    data/raw/camera_test/ is NOT saved (the case the camera test must be protected from);
  - with --dest camera_test, a frame that looks the same as any photo in
    data/raw/team_photos/ is NOT saved either (same leak, the other way round).
"Looks the same" uses the pipeline's LOOSER look-alike rule (keep_together_max_phash and
keep_together_max_dhash under near_duplicates in configs/label_map.yaml): the rule
build_splits.py uses to keep near-identical shots out of both train and a test set, so a
re-shot of the same item a second later, a little moved or brighter, is caught too. In the
same room a different item can now and then be refused as well: move it or change the angle.
The screen and the terminal say why a frame was refused.

Run from the repo root:
    python src/data/capture_photos.py --dest camera_test
    python src/data/capture_photos.py --dest team_photos --camera 1   # another camera
For tests only:
    python src/data/capture_photos.py --dest camera_test --press 4 --max-frames 60
    (acts as if 4 was pressed once after 30 frames, and quits after 60 frames)
"""

import argparse
import io
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# Reuse instead of copy: the box comes from the demo script, the look-alike check from
# the pipeline, so a change in either place reaches this script too.
# (Importing webcam_smooth also loads PyTorch, so the window takes a few seconds to open.)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
from build_splits import (CAMERA_TEST_DIR, CONFIG_PATH, CONVERT_TO_JPEG, IMAGE_EXTENSIONS,  # noqa: E402
                          RAW_DIR, hamming_distances, image_hashes, is_near_match, rel)
from webcam_smooth import center_box  # noqa: E402

# Number key -> class. Keys are the characters the keyboard sends.
KEYS = {"1": "compost", "2": "hazardous", "3": "recycling", "4": "trash"}
DESTINATIONS = {"camera_test": CAMERA_TEST_DIR, "team_photos": RAW_DIR / "team_photos"}
OTHER = {"camera_test": "team_photos", "team_photos": "camera_test"}  # the folder a photo must not also be in
PHOTO_SUFFIXES = IMAGE_EXTENSIONS | CONVERT_TO_JPEG  # "|" joins two sets: .jpg, .png, ... plus iPhone .heic
FLASH_SECONDS = 1.0       # how long the "saved" message stays on screen
WARNING_SECONDS = 2.5     # a "not saved" warning stays a bit longer
JPEG_QUALITY = 95
PRESS_AFTER_FRAMES = 30   # --press waits this long so the camera has adjusted its exposure

GREEN, RED, GREY, WHITE, DARK = (80, 255, 120), (60, 60, 255), (170, 170, 170), (255, 255, 255), (25, 25, 25)


def open_camera(index):
    """The camera, or a clear message and exit code 1."""
    camera = cv2.VideoCapture(index)
    if camera.isOpened():
        return camera
    print(f"Could not open camera {index}.\n"
          f"  Another camera:   add --camera {index + 1}\n"
          f"  macOS:            allow camera access for your terminal app in "
          f"System Settings > Privacy & Security > Camera, then run again.")
    sys.exit(1)


def look_alike_rules():
    """The looser pHash / dHash limits the pipeline uses to keep look-alike shots in one split.

    Returned under the names is_near_match() expects (max_phash, max_dhash).
    """
    with open(CONFIG_PATH) as f:
        rules = yaml.safe_load(f)["near_duplicates"]
    return {"max_phash": rules["keep_together_max_phash"], "max_dhash": rules["keep_together_max_dhash"]}


def folder_fingerprints(folder):
    """(pHash, dHash, path) of every photo in a photo folder and its class sub-folders."""
    out = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES and not path.name.startswith("."):
            try:
                p, d = image_hashes(path, upright=True)  # upright, like the pipeline does for phone photos
            except Exception as err:  # a broken file, or a HEIC photo without pillow-heif installed
                print(f"WARNING: could not read {rel(path)} for the look-alike check ({err}).")
                continue
            out.append((p, d, rel(path)))
    return out


def find_match(jpeg_bytes, known, rules):
    """The photo in `known` that this frame looks the same as, or None.

    jpeg_bytes are the exact bytes that would be saved, so the check sees the same
    picture the pipeline would later fingerprint.
    """
    if not known:
        return None
    p, d = image_hashes(io.BytesIO(jpeg_bytes))  # BytesIO lets PIL read bytes as if they were a file
    p_dist = hamming_distances(p, np.array([k[0] for k in known], dtype=np.uint64))
    d_dist = hamming_distances(d, np.array([k[1] for k in known], dtype=np.uint64))
    hits = np.nonzero(is_near_match(p_dist, d_dist, rules))[0]  # positions of every look-alike
    if len(hits) == 0:
        return None
    # Of all look-alikes, report the closest one (smallest total distance).
    best = min(hits, key=lambda j: int(p_dist[j]) + int(d_dist[j]))
    return f"{known[best][2]} (pHash {int(p_dist[best])}, dHash {int(d_dist[best])})"


def free_name(folder, label):
    """A file name like trash_20261027_163005.jpg that is not taken yet in the folder."""
    stem = f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    path, n = folder / f"{stem}.jpg", 2
    while path.exists():
        path, n = folder / f"{stem}_{n}.jpg", n + 1
    return path


def read_key():
    """The key pressed in the window during the last millisecond, as a character, or "".

    cv2.waitKey(1) waits 1 ms and returns the key code, or -1 if no key was pressed.
    "& 0xFF" keeps only the last 8 bits (some systems add extra bits), and chr() turns
    the number into its character, so the "1" key gives "1".
    """
    code = cv2.waitKey(1)
    return "" if code == -1 else chr(code & 0xFF)


def draw(screen, box, dest, message, message_color, saved_counts):
    """Center box, help line and (for a moment after a key press) the message, on the screen copy."""
    x0, y0, x1, y1 = box
    cv2.rectangle(screen, (x0, y0), (x1, y1), GREEN, 2)
    h = screen.shape[0]
    help_text = f"{dest}: 1 compost 2 hazardous 3 recycling 4 trash q quit"
    cv2.rectangle(screen, (0, h - 30), (screen.shape[1], h), DARK, -1)
    cv2.putText(screen, help_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    if message:
        # The second line is the warning text, or (after a save) the count per class this run.
        detail = message[1] or "this run: " + "  ".join(f"{c} {saved_counts[c]}" for c in KEYS.values())
        cv2.rectangle(screen, (10, 10), (min(screen.shape[1] - 10, max(420, 30 + 10 * len(detail))), 100), DARK, -1)
        cv2.putText(screen, message[0], (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, message_color, 2, cv2.LINE_AA)
        cv2.putText(screen, detail, (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Take labeled webcam photos into camera_test or team_photos.")
    parser.add_argument("--dest", required=True, choices=sorted(DESTINATIONS),
                        help="camera_test (demo-room photos, never trained on) or team_photos (joins training)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--press", choices=sorted(KEYS), help="for tests: act as if this key was pressed once")
    parser.add_argument("--max-frames", type=int, help="for tests: quit after this many frames")
    args = parser.parse_args()

    dest_dir = DESTINATIONS[args.dest]
    for label in KEYS.values():
        (dest_dir / label).mkdir(parents=True, exist_ok=True)  # the class folders exist in git already
    rules = look_alike_rules()

    # Fingerprints of every photo in the OTHER folder: a frame that looks like one of them is refused.
    other = OTHER[args.dest]
    known_other = folder_fingerprints(DESTINATIONS[other])
    print(f"Checking every frame against {len(known_other)} {other} photos "
          f"(pHash <= {rules['max_phash']} and dHash <= {rules['max_dhash']} = look-alike).")

    camera = open_camera(args.camera)
    print(f"Camera open. Saving to {rel(dest_dir)}/<class>/. Keys: "
          + ", ".join(f"{k} = {v}" for k, v in KEYS.items()) + ", q = quit.")

    saved_counts = Counter()  # photos saved this run, per class (a missing class counts as 0)
    message, message_color, message_until = None, GREEN, 0.0
    frames = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.")
                break
            frames += 1
            screen = frame.copy()  # draw on a copy, so the saved photo has no box or text on it

            key = read_key()
            if args.press and frames == PRESS_AFTER_FRAMES:
                key = args.press  # test mode: pretend the key was pressed now
            if key in ("q", "Q"):
                break

            if key in KEYS:
                label = KEYS[key]
                # Turn the frame into JPG bytes in memory; these exact bytes are checked and saved.
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                jpeg_bytes = encoded.tobytes()
                match = find_match(jpeg_bytes, known_other, rules)
                if match:
                    # Refuse: this picture is already in the other folder.
                    print(f"NOT SAVED: this frame looks the same as {other} photo {match}. "
                          f"A photo must never be in both camera_test and team_photos.")
                    message = ("NOT SAVED", f"Looks like a {other} photo. Move the item.")
                    message_color, message_until = RED, time.time() + WARNING_SECONDS
                else:
                    path = free_name(dest_dir / label, label)
                    path.write_bytes(jpeg_bytes)
                    saved_counts[label] += 1
                    print(f"Saved {rel(path)}   this run: "
                          + ", ".join(f"{c} {saved_counts[c]}" for c in KEYS.values()))
                    message = (f"Saved: {label}", "")
                    message_color, message_until = GREEN, time.time() + FLASH_SECONDS

            # The message shows only until its time is up; after that, pass None (no message).
            showing = message if time.time() < message_until else None
            draw(screen, center_box(frame), args.dest, showing, message_color, saved_counts)
            cv2.imshow(f"Capture to {args.dest} - 1/2/3/4 save, q quit", screen)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Closed after {frames} frames. Saved this run: "
              + ", ".join(f"{c} {saved_counts[c]}" for c in KEYS.values()) + ".")


if __name__ == "__main__":
    main()
