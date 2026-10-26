"""
train_clean.py - train a candidate model on the cleaned split from build_splits.py.

This is Amy's train.py with five changes, so the two models can be compared fairly:

  1. It reads the images from data/train and data/validation, which
     src/data/build_splits.py writes (instead of the old dataset_split/ folder),
     and checks that those folders hold exactly the images listed in
     data/splits/train.csv and data/splits/validation.csv.
  2. The loss uses the "sqrt" class weights from results/class_weights.json, so the
     big recycling class does not drown out compost, hazardous and trash.
  3. It trains in two stages:
       stage 1: 5 epochs, only the classifier head learns (like Amy's script),
       stage 2: 10 epochs, the last 3 feature blocks learn too, at a 10x lower
                learning rate so the pretrained features are only nudged.
  4. It keeps the epoch with the best validation MACRO F1 (the average F1 over
     the four classes), not the best accuracy. Accuracy mostly measures recycling,
     because recycling is most of the data.
  5. It saves to models/candidate.pth. It never writes models/waste_classifier.pth.

Same as Amy's script: MobileNetV3-small with the same pretrained ImageNet weights,
224 x 224 images, the same augmentation, batch size 32, Adam, learning rate 0.001
in stage 1, and the same checkpoint format (model_state_dict, classes,
best_val_accuracy). A few extra keys are added for traceability (see the end).

Smaller additions: it also uses the Apple GPU (MPS) on a Mac, loads images with
4 background workers, retries the pretrained-weights download with the certifi
certificates if the Mac's Python has none, and stops if data/train or
data/validation holds different images than the split CSVs.

Run from the repo root, after python src/data/build_splits.py:
    python src/train/train_clean.py
"""

import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# ---------------------------------------------------------------------------
# Paths. Path(__file__) is this file; parents[2] goes up from src/train/ to the repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"                      # written by build_splits.py
WEIGHTS_JSON = REPO_ROOT / "results" / "class_weights.json"
MODEL_DIR = REPO_ROOT / "models"
OUTPUT_PATH = MODEL_DIR / "candidate.pth"          # the new model goes here
PROTECTED_PATH = MODEL_DIR / "waste_classifier.pth"  # Amy's model: never overwritten

# build_splits.py lives in src/data/. Adding that folder to sys.path lets us import
# its functions that compute the split stamp, so this script prints the same stamp.
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
from build_splits import get_split_version, manifest_fingerprint  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 4
STAGE1_EPOCHS = 5              # head only
STAGE2_EPOCHS = 10             # head + last 3 feature blocks
STAGE1_LR = 0.001              # same as Amy's LEARNING_RATE
STAGE2_LR = STAGE1_LR / 10     # 10 times lower for the fine-tuning stage
UNFROZEN_BLOCKS = 3            # how many feature blocks (counted from the end) learn in stage 2
NUM_WORKERS = 4                # background processes that load images while the model trains

# Use an NVIDIA GPU if there is one, else the Apple GPU (MPS) on a Mac, else the CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed):
    """Fix every random number generator, so a rerun shuffles and augments the same way.

    On a GPU a few operations are still not bit-for-bit repeatable, so two runs can
    differ in the last decimals; the seed keeps them as close as PyTorch allows.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    """Each image-loading worker gets its own, but fixed, random seed (for the augmentation)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_stamp_and_weights():
    """Print the split stamp and return the sqrt class weights from results/class_weights.json.

    The stamp says exactly which split this model is trained on:
      split_version   = the last commit that changed build_splits.py or configs/
      manifest_sha256 = a fingerprint of the split CSVs in data/splits/
    The class weights were computed from that same split, so the script stops if the
    split on disk is not the one recorded in class_weights.json.
    """
    with open(WEIGHTS_JSON) as f:
        info = json.load(f)
    split_version, fingerprint = get_split_version(), manifest_fingerprint()
    print(f"Split version: {split_version}  manifest_sha256: {fingerprint}")
    if fingerprint != info["manifest_sha256"] or split_version != info["split_version"]:
        raise SystemExit(
            f"The split on disk ({split_version} / {fingerprint}) is not the one in "
            f"{WEIGHTS_JSON.relative_to(REPO_ROOT)} ({info['split_version']} / {info['manifest_sha256']}).\n"
            f"Run python src/data/build_splits.py first, so the class weights match the split.")
    sqrt = info["weights"]["sqrt_inverse_frequency"]
    return info["class_order"], sqrt["in_class_order"]


def load_pretrained_mobilenet(weights):
    """MobileNetV3-small with its pretrained ImageNet weights (downloaded once, then cached by torchvision).

    Python from python.org on a Mac often has no certificate list, so the download
    fails with CERTIFICATE_VERIFY_FAILED. Then it is retried with the certifi
    certificate list, the same fix src/data/download.py uses.
    """
    try:
        return models.mobilenet_v3_small(weights=weights)
    except urllib.error.URLError as err:
        if "CERTIFICATE_VERIFY_FAILED" not in str(err):
            raise
        import certifi
        print("Certificate check failed; retrying the weights download with the certifi certificates.")
        context = ssl.create_default_context(cafile=certifi.where())
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)))
        return models.mobilenet_v3_small(weights=weights)


def check_folder_matches_manifest(dataset, split):
    """Stop if data/<split>/ holds a different set of images than data/splits/<split>.csv.

    ImageFolder reads whatever files are in the folder. The CSV is the official list
    that the stamp fingerprints. If someone dropped an extra file into the folder, or
    deleted one, the model would no longer be trained on the stamped split.
    """
    import csv
    with open(DATA_DIR / "splits" / f"{split}.csv", newline="") as f:
        listed = {(row["path"], row["label"]) for row in csv.DictReader(f)}
    on_disk = {(Path(path).resolve().relative_to(REPO_ROOT).as_posix(), dataset.classes[label])
               for path, label in dataset.samples}
    if listed != on_disk:
        extra, missing = sorted(on_disk - listed), sorted(listed - on_disk)
        raise SystemExit(f"data/{split}/ does not match data/splits/{split}.csv: "
                         f"{len(extra)} extra files (e.g. {extra[:1]}), {len(missing)} missing (e.g. {missing[:1]}).\n"
                         f"Run python src/data/build_splits.py again.")


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over a data loader. With an optimizer it trains; without one it only evaluates.

    Returns (average loss, accuracy, macro F1, number of images).
    """
    training = optimizer is not None
    model.train(training)  # train mode turns on dropout; eval mode turns it off
    total_loss, all_true, all_pred = 0.0, [], []

    # torch.set_grad_enabled(False) during evaluation skips the gradient bookkeeping (faster, less memory).
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)            # one score per class for every image
            loss = criterion(outputs, labels)  # how wrong the scores are (weighted by class)
            if training:
                optimizer.zero_grad()          # clear the gradients from the previous batch
                loss.backward()                # work out how each weight should change
                optimizer.step()               # change the weights a little
            total_loss += loss.item() * images.size(0)
            all_true.extend(labels.tolist())
            all_pred.extend(outputs.argmax(1).tolist())  # the class with the highest score

    n = len(all_true)
    accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / n
    # Macro F1: the F1 score of each class, averaged with equal weight per class.
    macro_f1 = f1_score(all_true, all_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    return total_loss / n, accuracy, macro_f1, n


def main():
    start = time.perf_counter()
    set_seed(SEED)
    print(f"Seed: {SEED}   device: {DEVICE}")
    class_order, class_weights = read_stamp_and_weights()

    # ---- Images and augmentation (identical to Amy's train.py) ----
    weights = MobileNet_V3_Small_Weights.DEFAULT
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # random zoom/crop to 224 x 224
        transforms.RandomHorizontalFlip(),                     # mirror half the images
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = weights.transforms()  # resize to 256, center-crop 224: no randomness

    # ImageFolder treats each sub-folder (compost, hazardous, recycling, trash) as a class,
    # in alphabetical order, so output 0 = compost, 1 = hazardous, 2 = recycling, 3 = trash.
    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_data = datasets.ImageFolder(DATA_DIR / "validation", transform=eval_transform)
    if train_data.classes != class_order or val_data.classes != class_order:
        raise SystemExit(f"Class folders {train_data.classes} / {val_data.classes} do not match the "
                         f"class order in class_weights.json {class_order}.")
    check_folder_matches_manifest(train_data, "train")
    check_folder_matches_manifest(val_data, "validation")
    print(f"Classes: {train_data.classes}")
    print(f"Images: {len(train_data)} train, {len(val_data)} validation")

    # The generator fixes the shuffle order (seed 42); seed_worker fixes each worker's augmentation.
    generator = torch.Generator().manual_seed(SEED)
    loader_options = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                          persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    # ---- Model: pretrained MobileNetV3-small with a new 4-class last layer (like Amy's) ----
    model = load_pretrained_mobilenet(weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # ---- Loss with sqrt class weights ----
    # A mistake on a rare class (hazardous 1.37) costs more than one on recycling (0.54).
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    print("Loss weights (sqrt): " + ", ".join(f"{c} {w:.4f}" for c, w in zip(class_order, class_weights)))

    # ---- Two training stages ----
    # Each stage: (name, epochs, learning rate, which feature blocks can learn).
    # model.features is a list of 13 blocks; features[-3:] are the last 3.
    stages = [
        ("stage 1 (head only)", STAGE1_EPOCHS, STAGE1_LR, []),
        (f"stage 2 (head + last {UNFROZEN_BLOCKS} feature blocks)", STAGE2_EPOCHS, STAGE2_LR,
         list(model.features[-UNFROZEN_BLOCKS:])),
    ]

    best = {"macro_f1": -1.0}
    epoch_number, total_epochs = 0, STAGE1_EPOCHS + STAGE2_EPOCHS
    for stage_name, epochs, lr, unfrozen_blocks in stages:
        # Freeze every feature weight (requires_grad = False means "do not change this")...
        # Note: like in Amy's train.py, the BatchNorm layers of frozen blocks still update
        # their running averages in train mode; only their learned weights stay fixed.
        for param in model.features.parameters():
            param.requires_grad = False
        # ...then unfreeze the blocks this stage trains. The classifier head always learns.
        for block in unfrozen_blocks:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

        # A fresh optimizer per stage, given only the weights that are allowed to learn.
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        print(f"\n{stage_name}: {epochs} epochs, learning rate {lr:g}, "
              f"{sum(p.numel() for p in trainable):,} trainable weights")

        for _ in range(epochs):
            epoch_number += 1
            epoch_start = time.perf_counter()
            train_loss, train_acc, train_f1, _ = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1, _ = run_epoch(model, val_loader, criterion)
            line = (f"Epoch {epoch_number}/{total_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2%} | "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%} | Val macro F1: {val_f1:.4f} | "
                    f"{time.perf_counter() - epoch_start:.0f}s")

            # Keep this epoch only if its validation macro F1 is the best so far.
            if val_f1 > best["macro_f1"]:
                best = {"macro_f1": val_f1, "accuracy": val_acc, "epoch": epoch_number}
                if OUTPUT_PATH.resolve() == PROTECTED_PATH.resolve():  # belt and suspenders
                    raise SystemExit("Refusing to overwrite models/waste_classifier.pth.")
                MODEL_DIR.mkdir(exist_ok=True)
                torch.save(
                    {
                        # Same three keys as Amy's train.py, so webcam_test.py and
                        # export_predictions.py load it the same way.
                        "model_state_dict": model.state_dict(),
                        "classes": [c.capitalize() for c in train_data.classes],  # "Compost", like Amy's
                        "best_val_accuracy": val_acc,  # accuracy at the best-macro-F1 epoch
                        # Extra keys, for traceability only.
                        "best_val_macro_f1": val_f1,
                        "best_epoch": epoch_number,
                        "split_version": get_split_version(),
                        "manifest_sha256": manifest_fingerprint(),
                        "seed": SEED,
                    },
                    OUTPUT_PATH,
                )
                line += "  <- best so far, saved"
            print(line, flush=True)

    minutes = (time.perf_counter() - start) / 60
    print(f"\nBest validation macro F1: {best['macro_f1']:.4f} (accuracy {best['accuracy']:.2%}) at epoch {best['epoch']}")
    print(f"Saved best model to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Training time: {minutes:.1f} minutes on {DEVICE}")


# On macOS the image-loading workers start by re-importing this file; this guard keeps
# them from starting a second training run.
if __name__ == "__main__":
    main()
