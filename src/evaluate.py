"""
===============================================================================
FACIAL EMOTION RECOGNITION — MODEL EVALUATION MODULE (L6 STANDARD)

Purpose:
    Evaluate trained PyTorch FER model using validation dataset.

Outputs:
    - Accuracy
    - Precision / Recall / F1 (macro)
    - Classification report (saved to metrics/)
    - Confusion matrix (saved as PNG)

Design Principles:
    - No hardcoded metrics
    - Real dataset evaluation
    - Reproducible
    - Artifact saving for transparency

Usage:
    python scripts/evaluate.py --model cnn
===============================================================================
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from torch.utils.data import DataLoader
from src.dataset import FERDataset
from src.model import EmotionCNN, ResNetEmotion


# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_PATH = "artifacts/models/best_model.pt"
CSV_PATH = "fer2013.csv"
BATCH_SIZE = 64
NUM_CLASSES = 7


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate(model, dataloader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, report, cm


# ==============================================================================
# SAVE METRICS
# ==============================================================================

def save_metrics(report, cm, accuracy):

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nOverall Accuracy: {accuracy:.4f}\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "resnet"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(CSV_PATH)
    val_df = df.sample(frac=0.1, random_state=42)

    dataset = FERDataset(val_df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    if args.model == "resnet":
        model = ResNetEmotion(NUM_CLASSES)
    else:
        model = EmotionCNN(NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    accuracy, report, cm = evaluate(model, dataloader, device)

    print("\n📊 Evaluation Results")
    print(report)
    print(f"Overall Accuracy: {accuracy:.4f}")

    save_metrics(report, cm, accuracy)


if __name__ == "__main__":
    main()
