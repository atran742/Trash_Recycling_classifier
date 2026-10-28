"""
capture_photos.py - take labeled photos with the webcam, straight into the right folder.

It opens the webcam and shows the live feed with the same center box as the demo
(src/webcam_smooth.py), so you hold each item where the demo will look for it.
Press a number key to save the current frame:

    1 = compost    2 = hazardous    3 = recycling    4 = trash    q = quit

Hold Shift with the number key to record that item for that bin instead (video mode):

    Shift+1 = record compost   Shift+2 = hazardous   Shift+3 = recycling   Shift+4 = trash

A recording lasts 10 seconds, or stops early when you press any other key (q too: press q
again to quit), or the same Shift+number again after letting go for 2 seconds. A red "REC" and the seconds left show on screen. While recording, one
frame is saved every 2 seconds (at 0, 2, 4, 6 and 8 seconds, so 5 frames for a full
recording), named <class>_item<NN>_f<frame>.jpg, for example trash_item03_f2.jpg. Every
recording gets the next item number in the --dest folder (counted over all four classes),
so all frames of one recording share one item number and compare_models.py can score them
as one item. The numbers count only the photos on THIS computer: record camera_test on one
computer, or rename items before two people's folders are put together (two item01s would
clash, and a copy renamed to "trash_item01_f1 (1).jpg" no longer counts as a recording frame).
Shift+1 sends "!", Shift+2 "@", Shift+3 "#" and Shift+4 "$" on a US keyboard;
on another keyboard layout Shift+number may send a different character and do nothing.

--dest picks the folder, and there is no default, so you always choose on purpose:

    camera_test  -> data/raw/camera_test/<class>/   demo-room photos, NEVER trained on
    team_photos  -> data/raw/team_photos/<class>/   our own photos, they join train/validation

A single photo is the FULL frame, saved as a JPG named <class>_<date>_<time>.jpg, for example
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

Recordings get one more check inside the --dest folder: a recording frame that looks the
same as a frame of a DIFFERENT recording (an earlier one this run, or one already on disk)
is not saved either, because one item recorded twice would count as two items. Frames of
the SAME recording are never checked against each other: they are meant to look alike.
A refused frame is skipped and the recording goes on. Single photos (plain number keys) are
not part of this check, and every single photo counts as an item of its own in
compare_models.py: for several angles of ONE object, record it instead of taking single
photos, and do not take both of the same object.
A key held down repeats by itself; a save or record key is read once while it is held, and
counts again only 2 seconds after you let go (so wait 2 seconds between two photos with one key).

Run from the repo root:
    python src/data/capture_photos.py --dest camera_test
    python src/data/capture_photos.py --dest team_photos --camera 1   # another camera
For tests only:
    python src/data/capture_photos.py --dest camera_test --press 4 --max-frames 60
    (acts as if 4 was pressed once after 30 frames, and quits after 60 frames)
    python src/data/capture_photos.py --dest camera_test --press shift+4 --press shift+1 --max-frames 800
    (each --press waits 30 frames after the last one, or after the recording it started ended)
"""

import argparse
import io
import sys
from time import monotonic as clock  # seconds that only ever go forward (unlike the wall clock)
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
from recording_names import ITEM_NAME  # noqa: E402  (trash_item03_f2.jpg; compare_models.py reads it too)
from webcam_smooth import center_box  # noqa: E402

# Number key -> class. Keys are the characters the keyboard sends.
KEYS = {"1": "compost", "2": "hazardous", "3": "recycling", "4": "trash"}
# Shift + number key -> record that class. A US keyboard sends "!" for Shift+1, and so on.
SHIFTED = {"1": "!", "2": "@", "3": "#", "4": "$"}
RECORD_KEYS = {SHIFTED[k]: label for k, label in KEYS.items()}
# The names --press accepts: "1" ... "4", and "shift+1" ... "shift+4" for a recording.
# {**a, **b} builds one dictionary from the entries of two.
PRESS_NAMES = {**{k: k for k in KEYS}, **{f"shift+{k}": SHIFTED[k] for k in KEYS}}
RECORD_SECONDS = 10.0     # a recording stops by itself after this long
FRAME_EVERY = 2.0         # one frame is saved every this many seconds while recording
FRAMES_PER_RECORDING = int(RECORD_SECONDS / FRAME_EVERY)  # 5: at 0, 2, 4, 6 and 8 seconds
# A key held down repeats by itself (!!!!!... or 44444...). The same save or record key seen again
# within this many seconds of its last repeat counts as still held, not as a new press. So holding
# Shift+4 a bit long does not stop the recording it just started, and holding 4 saves ONE photo,
# not twenty near copies (each would count as an item). 2 s is longer than the slowest
# "Delay until repeat" setting on a Mac.
HELD_KEY_SECONDS = 2.0
DESTINATIONS = {"camera_test": CAMERA_TEST_DIR, "team_photos": RAW_DIR / "team_photos"}
OTHER = {"camera_test": "team_photos", "team_photos": "camera_test"}  # the folder a photo must not also be in
PHOTO_SUFFIXES = IMAGE_EXTENSIONS | CONVERT_TO_JPEG  # "|" joins two sets: .jpg, .png, ... plus iPhone .heic
FLASH_SECONDS = 1.0       # how long the "saved" message stays on screen
WARNING_SECONDS = 2.5     # a "not saved" warning stays a bit longer
JPEG_QUALITY = 95
PRESS_AFTER_FRAMES = 30   # each --press waits this long, so the camera has adjusted its exposure

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


def recording_fingerprints(folder):
    """(pHash, dHash, path, item number) of every recording frame already in the folder."""
    out = []
    for path in sorted(folder.rglob("*")):
        found = ITEM_NAME.fullmatch(path.name)
        if found and path.is_file():
            try:
                p, d = image_hashes(path)
            except Exception as err:  # a broken file
                print(f"WARNING: could not read {rel(path)} for the look-alike check ({err}).")
                continue
            out.append((p, d, rel(path), int(found["item"])))
    return out


def next_item_number(folder):
    """One more than the highest item number in the folder (all classes), or 1 if there is none."""
    highest = 0
    for path in folder.rglob("*"):
        found = ITEM_NAME.fullmatch(path.name)
        if found:
            highest = max(highest, int(found["item"]))
    return highest + 1


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


def draw(screen, box, dest, message, message_color, saved_counts, recording=None, seconds_left=0):
    """Center box, help line, REC sign while recording and (for a moment after a key press)
    the message, on the screen copy."""
    x0, y0, x1, y1 = box
    cv2.rectangle(screen, (x0, y0), (x1, y1), RED if recording else GREEN, 2)
    h = screen.shape[0]
    help_text = f"{dest}: 1 compost 2 hazardous 3 recycling 4 trash (Shift = record) q quit"
    cv2.rectangle(screen, (0, h - 30), (screen.shape[1], h), DARK, -1)
    cv2.putText(screen, help_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    if recording:
        # A red dot, "REC", the seconds left, and which item is being recorded, just above the help line.
        rec_text = (f"REC {seconds_left}s  {recording['label']} item{recording['item']:02d}  "
                    f"frames {recording['saved']}/{FRAMES_PER_RECORDING}  other key = stop")
        cv2.rectangle(screen, (0, h - 62), (min(screen.shape[1], 40 + 9 * len(rec_text)), h - 32), DARK, -1)
        cv2.circle(screen, (18, h - 47), 8, RED, -1)
        cv2.putText(screen, rec_text, (34, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
    if message:
        # The second line is the warning text, or (after a save) the count per class this run.
        detail = message[1] or "this run: " + "  ".join(f"{c} {saved_counts[c]}" for c in KEYS.values())
        cv2.rectangle(screen, (10, 10), (min(screen.shape[1] - 10, max(420, 30 + 10 * len(detail))), 100), DARK, -1)
        cv2.putText(screen, message[0], (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, message_color, 2, cv2.LINE_AA)
        cv2.putText(screen, detail, (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def encode(frame):
    """The frame as JPG bytes in memory; these exact bytes are checked and saved."""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return encoded.tobytes()


def refuse(text, why):
    """Print why a frame was not saved, and return the red on-screen warning for it."""
    print(f"NOT SAVED: {text}")
    return ("NOT SAVED", why), RED, clock() + WARNING_SECONDS


def main():
    parser = argparse.ArgumentParser(description="Take labeled webcam photos into camera_test or team_photos.")
    parser.add_argument("--dest", required=True, choices=sorted(DESTINATIONS),
                        help="camera_test (demo-room photos, never trained on) or team_photos (joins training)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--press", action="append", choices=sorted(PRESS_NAMES),
                        help="for tests: act as if this key was pressed (repeat it for more presses)")
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
    # Fingerprints of every recording frame in THIS folder: a recording frame that looks like a
    # frame of a different recording is refused too. Frames saved this run are added as we go.
    known_recordings = recording_fingerprints(dest_dir)
    if known_recordings:
        print(f"Recording frames are also checked against {len(known_recordings)} frames of earlier "
              f"recordings in {rel(dest_dir)}/.")

    camera = open_camera(args.camera)
    print(f"Camera open. Saving to {rel(dest_dir)}/<class>/. Keys: "
          + ", ".join(f"{k} = {v}" for k, v in KEYS.items()) + ", q = quit. Hold Shift to record instead.")

    saved_counts = Counter()  # photos saved this run, per class (recording frames too; a missing class counts as 0)
    items_recorded = 0        # recordings that saved at least one frame this run
    message, message_color, message_until = None, GREEN, 0.0
    recording = None          # while recording: the label, item number, start time, frames tried and saved
    presses = [PRESS_NAMES[name] for name in args.press or []]  # test mode: keys still to "press"
    frames = idle_frames = 0  # idle_frames = frames since the last test press or the last recording ended
    last_key, last_key_at = "", -HELD_KEY_SECONDS  # the last key seen and when, to spot a held-down key
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.")
                break
            frames += 1
            screen = frame.copy()  # draw on a copy, so the saved photo has no box or text on it

            key = read_key()
            if recording is None:
                idle_frames += 1
                if presses and idle_frames >= PRESS_AFTER_FRAMES:
                    key, idle_frames = presses.pop(0), 0  # test mode: pretend the next key was pressed now
            now = clock()
            if key:
                # The same save or record key again, soon after we last saw it = the key is still held down.
                held = (key in KEYS or key in RECORD_KEYS) and key == last_key and now - last_key_at < HELD_KEY_SECONDS
                last_key, last_key_at = key, now  # every repeat moves the time on, for as long as it is held
                if held:
                    key = ""

            if recording is None:
                if key in ("q", "Q"):
                    break
                if key in KEYS:
                    label = KEYS[key]
                    jpeg_bytes = encode(frame)
                    match = find_match(jpeg_bytes, known_other, rules)
                    if match:
                        # Refuse: this picture is already in the other folder.
                        message, message_color, message_until = refuse(
                            f"this frame looks the same as {other} photo {match}. "
                            f"A photo must never be in both camera_test and team_photos.",
                            f"Looks like a {other} photo. Move the item.")
                    else:
                        path = free_name(dest_dir / label, label)
                        path.write_bytes(jpeg_bytes)
                        saved_counts[label] += 1
                        print(f"Saved {rel(path)}   this run: "
                              + ", ".join(f"{c} {saved_counts[c]}" for c in KEYS.values()))
                        message = (f"Saved: {label}", "")
                        message_color, message_until = GREEN, now + FLASH_SECONDS
                elif key in RECORD_KEYS:
                    # Start a recording. The item number is worked out from the files on disk, so it
                    # carries on from earlier runs, and a recording that saved nothing frees its number.
                    recording = {"label": RECORD_KEYS[key], "item": next_item_number(dest_dir),
                                 "started": now, "tried": 0, "saved": 0}
                    print(f"Recording {recording['label']} item{recording['item']:02d} "
                          f"for {RECORD_SECONDS:.0f} s (any other key stops it).")
            elif key or now - recording["started"] >= RECORD_SECONDS:
                # Stop: time is up, or a key was pressed (that key only stops the recording).
                label, item, saved = recording["label"], recording["item"], recording["saved"]
                items_recorded += saved > 0  # True counts as 1, False as 0
                print(f"Recording stopped: {label} item{item:02d}, {saved} of {recording['tried']} frames saved.")
                message = (f"Recorded: {label}", f"item{item:02d}: {saved} frames saved")
                message_color, message_until = (GREEN if saved else RED), now + FLASH_SECONDS
                recording, idle_frames = None, 0

            # While recording, save a frame when the next 2-second mark is reached (0, 2, 4, 6, 8 s).
            # int(seconds // FRAME_EVERY) + 1 = how many marks have passed, e.g. 4.3 s -> 3 marks.
            if recording is not None:
                due = min(int((now - recording["started"]) // FRAME_EVERY) + 1, FRAMES_PER_RECORDING)
                if recording["tried"] < due:
                    # Jump to the newest mark: if the camera stalled past a mark, that frame is skipped
                    # instead of being caught up a split second later (it would be a near copy).
                    recording["tried"] = due
                    label, item, n = recording["label"], recording["item"], recording["tried"]
                    jpeg_bytes = encode(frame)
                    match = find_match(jpeg_bytes, known_other, rules)
                    # Only frames of OTHER recordings count here; this recording's own frames are skipped.
                    match_rec = None if match else find_match(
                        jpeg_bytes, [k for k in known_recordings if k[3] != item], rules)
                    if match:
                        message, message_color, message_until = refuse(
                            f"frame {n} of {label} item{item:02d} looks the same as {other} photo {match}. "
                            f"A photo must never be in both camera_test and team_photos.",
                            f"Looks like a {other} photo. Move the item.")
                    elif match_rec:
                        message, message_color, message_until = refuse(
                            f"frame {n} of {label} item{item:02d} looks the same as {match_rec}, "
                            f"a frame of another recording. One item must not be recorded twice.",
                            "Looks like another recording. New item?")
                    else:
                        path = dest_dir / label / f"{label}_item{item:02d}_f{n}.jpg"
                        path.write_bytes(jpeg_bytes)
                        p_hash, d_hash = image_hashes(io.BytesIO(jpeg_bytes))
                        known_recordings.append((p_hash, d_hash, rel(path), item))
                        recording["saved"] += 1
                        saved_counts[label] += 1
                        print(f"Saved {rel(path)}")

            # The message shows only until its time is up; after that, pass None (no message).
            showing = message if clock() < message_until else None
            seconds_left = 0
            if recording is not None:
                seconds_left = int(RECORD_SECONDS - (now - recording["started"]) + 0.999)  # rounded up: 10 ... 1
            draw(screen, center_box(frame), args.dest, showing, message_color, saved_counts, recording, seconds_left)
            cv2.imshow(f"Capture to {args.dest} - 1/2/3/4 save, Shift+1/2/3/4 record, q quit", screen)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if recording is not None:
            print(f"Window closed while recording {recording['label']} item{recording['item']:02d} "
                  f"({recording['saved']} frames saved).")
            items_recorded += recording["saved"] > 0
        print(f"Closed after {frames} frames. Saved this run: "
              + ", ".join(f"{c} {saved_counts[c]}" for c in KEYS.values())
              + f" (recordings: {items_recorded}).")


if __name__ == "__main__":
    main()
