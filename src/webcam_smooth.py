"""
webcam_smooth.py - the demo loop: a steady answer, a center box, and "I'm not sure".

Builds on src/webcam_test.py (Amy's script, reused as is for loading the model)
and adds three things the demo needs:
  1. Smoothing: the last HISTORY predictions are averaged, so the label does not
     flicker between two classes from frame to frame.
  2. A center box: only the middle of the frame is classified, so the person,
     the table and the room do not count. Hold the item inside the box.
  3. "I'm not sure": when the averaged confidence is below CUTOFF the screen says
     so instead of guessing. CUTOFF 0.55 is the lowest cutoff reaching 85%
     accuracy on answered validation items for Amy's first model (it answers 93%
     of them); on the test set it answers 93% at 87%. See docs/evaluation.md.
Below the label, one line says what to do with the item under San Diego rules.

Run from the repo root:
    python src/webcam_smooth.py
    python src/webcam_smooth.py --camera 1          # another camera
    python src/webcam_smooth.py --max-frames 60     # quit by itself (for tests)
Press q to quit.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Reuse Amy's loader, so the model, classes, device and preprocessing are exactly hers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from webcam_test import DEVICE, load_model  # noqa: E402

# ---- Settings. Change these, nothing else, to tune the demo. ----
CUTOFF = 0.55         # below this averaged confidence the screen says "I'm not sure"
HISTORY = 8           # how many of the latest predictions are averaged
BOX_FRACTION = 0.55   # side of the center box as a share of the shorter frame side
PREDICT_EVERY = 2     # run the model every N frames (1 = every frame)
CAMERA_INDEX = 0      # 0 is usually the built-in / default webcam

# One line per class, San Diego rules. Keys are lowercase class names.
RULES = {
    "recycling": "Blue bin. Empty and rinse first.",
    "compost": "Green bin.",
    "trash": "Black bin.",
    "hazardous": "Not curbside. Take to a hazardous waste drop-off.",
}

GREEN, GREY, WHITE, DARK = (80, 255, 120), (170, 170, 170), (255, 255, 255), (25, 25, 25)


def center_box(frame):
    """(x0, y0, x1, y1) of the square box in the middle of the frame."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * BOX_FRACTION)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return x0, y0, x0 + side, y0 + side


def predict(model, preprocess, crop_bgr):
    """Class probabilities for one crop, as a numpy array in the model's class order."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)  # OpenCV is BGR; the model expects RGB
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def decide(history, classes):
    """Average the stored probability vectors; return (label or None, confidence)."""
    if not history:
        return None, 0.0
    mean = np.mean(history, axis=0)
    index = int(mean.argmax())
    confidence = float(mean[index])
    return (classes[index] if confidence >= CUTOFF else None), confidence


def draw(frame, box, label, confidence, n_averaged):
    """Box, label line and rule line on the frame."""
    x0, y0, x1, y1 = box
    sure = label is not None
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN if sure else GREY, 2)
    if sure:
        headline = f"{label.capitalize()}  {confidence:.0%}"
        rule = RULES.get(label.lower(), "")
    else:
        headline = f"I'm not sure  ({confidence:.0%})"
        rule = "Hold one item still inside the box."
    cv2.rectangle(frame, (10, 10), (max(560, 24 + 22 * len(rule)), 92), DARK, -1)
    cv2.putText(frame, headline, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, GREEN if sure else GREY, 2, cv2.LINE_AA)
    cv2.putText(frame, rule, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"avg of {n_averaged}/{HISTORY}  cutoff {CUTOFF:.2f}  q = quit",
                (20, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)


def open_camera(index):
    """The camera, or a clear message and exit code 1 (never a stack trace at the demo)."""
    camera = cv2.VideoCapture(index)
    if camera.isOpened():
        return camera
    print(f"Could not open camera {index}.\n"
          f"  Another camera:   python src/webcam_smooth.py --camera {index + 1}\n"
          f"  macOS:            allow camera access for Terminal in System Settings > Privacy & Security > Camera.\n"
          f"  Raspberry Pi:     see raspberry_pi/README.md (camera module setup).")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Webcam demo with smoothing, a center box and 'I'm not sure'.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help=f"camera index (default {CAMERA_INDEX})")
    parser.add_argument("--max-frames", type=int, help="quit after this many frames (for tests)")
    args = parser.parse_args()

    model, classes, preprocess = load_model()
    print(f"Model loaded on {DEVICE}. Classes: {list(classes)}")
    print(f"Cutoff {CUTOFF}, averaging the last {HISTORY} predictions, one every {PREDICT_EVERY} frames.")
    unknown = [c for c in classes if c.lower() not in RULES]
    if unknown:
        print(f"WARNING: no San Diego rule line for {unknown}; the rule line will be empty for them.")

    camera = open_camera(args.camera)
    print("Camera open. Hold one item inside the box. Press q to quit.")
    history = deque(maxlen=HISTORY)  # the latest probability vectors; old ones fall off the end
    frames, started = 0, time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.")
                break
            frames += 1
            box = center_box(frame)
            if frames % PREDICT_EVERY == 0:
                x0, y0, x1, y1 = box
                history.append(predict(model, preprocess, frame[y0:y1, x0:x1]))
            label, confidence = decide(history, classes)
            draw(frame, box, label, confidence, len(history))
            cv2.imshow("Trash / Recycling Classifier - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        seconds = time.time() - started
        print(f"Closed after {frames} frames in {seconds:.1f} s ({frames / seconds:.1f} fps)." if seconds else "Closed.")


if __name__ == "__main__":
    main()
