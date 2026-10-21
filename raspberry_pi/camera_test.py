"""
camera_test.py - take ONE photo to prove the camera works.

Saves it as raspberry_pi/test_photos/photo_YYYYMMDD_HHMMSS.jpg and prints
which camera path worked and the photo size. The folder is git-ignored.

Two camera paths, tried in this order:
  1. picamera2  - the Raspberry Pi camera module (ribbon cable). Comes with
                  Raspberry Pi OS: sudo apt install python3-picamera2
  2. OpenCV     - a USB webcam, or the built-in camera on a laptop, so the
                  same script also runs on a Mac or PC: pip install opencv-python

Run from the repo root:
    python raspberry_pi/camera_test.py
    python raspberry_pi/camera_test.py --index 1    # try the second webcam
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PHOTO_DIR = Path(__file__).resolve().parent / "test_photos"


def photo_path():
    """A new file name with the current date and time, e.g. photo_20261021_143005.jpg.

    The folder is only created here, right before a camera writes into it, so a
    failed run leaves nothing behind.
    """
    PHOTO_DIR.mkdir(exist_ok=True)
    return PHOTO_DIR / f"photo_{datetime.now():%Y%m%d_%H%M%S}.jpg"


def try_picamera2():
    """Take a photo with the Pi camera module. Returns (path, (width, height)) or None if unavailable."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed (normal on a Mac/PC). Trying OpenCV next.")
        return None
    cam = None
    try:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration())
        cam.start()
        time.sleep(2)  # let auto-exposure settle, or the photo comes out dark
        out = photo_path()
        cam.capture_file(str(out))
    except Exception as err:  # no camera plugged in, camera busy, ribbon cable loose...
        print(f"picamera2 is installed but failed: {err}. Trying OpenCV next.")
        return None
    finally:
        if cam is not None:
            cam.close()  # always release the camera, even after a failure
    from PIL import Image
    with Image.open(out) as img:
        return out, img.size


def try_opencv(index):
    """Take a photo with a USB webcam or laptop camera. Returns (path, (width, height)) or None."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Run: pip install opencv-python")
        return None
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print(f"OpenCV could not open camera index {index}.")
        return None
    # The first frames are often dark or empty while the camera warms up, so
    # read a few and keep the last one.
    frame = None
    for _ in range(10):
        ok, candidate = cam.read()
        if ok and candidate is not None:
            frame = candidate
        time.sleep(0.1)
    cam.release()
    if frame is None:
        print(f"OpenCV opened camera index {index} but got no frames.")
        return None
    out = photo_path()
    cv2.imwrite(str(out), frame)
    height, width = frame.shape[:2]
    return out, (width, height)


def main():
    parser = argparse.ArgumentParser(description="Take one test photo with the Pi camera or a webcam.")
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera number (default 0)")
    args = parser.parse_args()

    result = try_picamera2()
    used = "picamera2 (Pi camera module)"
    if result is None:
        result = try_opencv(args.index)
        used = f"OpenCV (camera index {args.index})"
    if result is None:
        print("\nNo camera worked. Check that a camera is connected and not in use by another app.\n"
              "  Pi camera module: sudo apt install python3-picamera2, then reboot.\n"
              "  USB webcam / laptop camera: pip install opencv-python, then try --index 1.\n"
              "  macOS: allow camera access for Terminal in System Settings > Privacy & Security > Camera.")
        sys.exit(1)
    out, size = result
    print(f"\nPhoto taken with {used}: {size[0]} x {size[1]} pixels")
    print(f"Saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
