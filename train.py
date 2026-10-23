
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

# Configuration
DATA_DIR = Path("dataset_split")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 8 # maybe change 
LEARNING_RATE = 0.001
NUM_CLASSES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Use the preprocessing expected by pretrained MobileNetV3.
weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()

# Add light augmentation to training images.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Validation and test data should not be augmented.
eval_transform = preprocess

train_data = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_data = datasets.ImageFolder(
    DATA_DIR / "val", transform=eval_transform
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False
)

print("Classes:", train_data.classes)
assert train_data.classes == val_data.classes

# Load pretrained CNN.
model = models.mobilenet_v3_small(weights=weights)

# Freeze pretrained feature extractor for the baseline.
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classifier layer for our four waste categories.
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(), lr=LEARNING_RATE
)

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss / train_total:.4f} | "
        f"Train acc: {train_accuracy:.2%} | "
        f"Val loss: {val_loss / val_total:.4f} | "
        f"Val acc: {val_accuracy:.2%}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes,
                "best_val_accuracy": best_val_accuracy,
            },
            MODEL_DIR / "waste_classifier.pth",
        )

print(f"\nBest validation accuracy: {best_val_accuracy:.2%}")
print(f"Saved best model to {MODEL_DIR / 'waste_classifier.pth'}")
