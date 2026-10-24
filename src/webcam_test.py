
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

# Paths are relative to the project root.
MODEL_PATH = Path("models/waste_classifier.pth")

# Use Apple Silicon MPS if available, otherwise CUDA or CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the model first or check the checkpoint path."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]

    # Recreate the exact architecture used during training.
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        in_features, len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Match the preprocessing used for validation and testing.
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()

    return model, classes, preprocess


def main():
    model, classes, preprocess = load_model()

    print(f"Model loaded on: {DEVICE}")
    print(f"Categories: {classes}")
    print("Starting webcam... Press 'q' to quit.")

    # 0 is usually the built-in/default webcam.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "or try changing VideoCapture(0) to VideoCapture(1)."
        )

    prediction = "Show an item to the camera"
    confidence = 0.0
    frame_count = 0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame_count += 1

            # Run inference every 5 frames to reduce CPU workload.
            if frame_count % 5 == 0:
                # OpenCV uses BGR; the model expects RGB.
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                pil_image = Image.fromarray(rgb_frame)
                input_tensor = preprocess(pil_image).unsqueeze(0)
                input_tensor = input_tensor.to(DEVICE)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence_tensor, predicted_index = (
                        probabilities.max(dim=1)
                    )

                prediction = classes[predicted_index.item()]
                confidence = confidence_tensor.item()

            # Display prediction and confidence.
            label = f"{prediction}: {confidence:.1%}"

            cv2.rectangle(
                frame, (10, 10), (440, 65), (25, 25, 25), -1
            )
            cv2.putText(
                frame,
                label,
                (20, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 255, 120),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Waste Classifier - Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
