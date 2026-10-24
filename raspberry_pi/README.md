# Raspberry Pi: first camera test

Five steps for a first-time Pi user. Nothing here needs the model yet.

1. **Set up the Pi.** Flash Raspberry Pi OS (64-bit) onto a microSD card with Raspberry Pi Imager, boot the Pi 5, connect it to Wi-Fi, and open a terminal.
2. **Connect the camera.** With the Pi powered off, plug the camera module's ribbon cable into a CAMERA port (metal contacts facing the port's contacts), then power on. A USB webcam also works: just plug it in.
3. **Install what the script needs.**
   ```bash
   sudo apt update && sudo apt install -y git python3-picamera2 python3-opencv
   ```
   `python3-picamera2` drives the camera module and `python3-opencv` drives a USB webcam.
4. **Get the code.**
   ```bash
   git clone https://github.com/atran742/Trash_Recycling_classifier.git
   cd Trash_Recycling_classifier
   ```
5. **Take a test photo.**
   ```bash
   python3 raspberry_pi/camera_test.py
   ```
   A good run prints the camera path that worked and the photo size, for example `Photo taken with picamera2 (Pi camera module): 4608 x 2592 pixels`, and saves the photo under `raspberry_pi/test_photos/` (git-ignored). If it prints `No camera worked`, follow the hints it gives.

The same script runs on a Mac or PC with a webcam through OpenCV, so you can try it before the Pi arrives. On macOS, allow camera access for your terminal app in System Settings > Privacy & Security > Camera.
