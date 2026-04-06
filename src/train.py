"""
===============================================================================
FACIAL EMOTION RECOGNITION — PYTORCH TRAINING PIPELINE (L6 STANDARD)

Purpose:
    Train either EmotionCNN or ResNetEmotion on FER-2013 dataset.

Design Features:
    - Architecture selection via CLI
    - Deterministic training
    - GPU auto-detection
    - Early stopping
    - Checkpointing best model
    - Metric logging
    - Clean separation of data + model + training loop

Usage:
    python train.py --model cnn
    python train.py --model resnet
===============================================================================
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.model import EmotionCNN, ResNetEmotion


# ==============================================================================
# CONFIG
# ==============================================================================

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
NUM_CLASSES = 7
IMAGE_SIZE = 48


# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# DATASET
# ==============================================================================

class FERDataset(Dataset):
    """
    Custom PyTorch Dataset for FER-2013.
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype="float32")
        image = pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE) / 255.0
        label = row["emotion"]

        return torch.tensor(image), torch.tensor(label)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(model, train_loader, val_loader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 5

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("artifacts/models", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/models/best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "resnet"],
                        help="Select model architecture")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv("fer2013.csv")
    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=RANDOM_SEED, stratify=df["emotion"]
    )

    train_dataset = FERDataset(train_df)
    val_dataset = FERDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Select model
    if args.model == "resnet":
        model = ResNetEmotion(NUM_CLASSES)
    else:
        model = EmotionCNN(NUM_CLASSES)

    model.to(device)

    train(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
