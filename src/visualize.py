# src/visualize.py

import matplotlib.pyplot as plt

# Simulated training history for example purpose (replace with actual history data if available)
history = {
    'accuracy': [0.55, 0.62, 0.69, 0.73, 0.76],
    'val_accuracy': [0.53, 0.60, 0.65, 0.70, 0.72],
    'loss': [1.0, 0.85, 0.70, 0.60, 0.52],
    'val_loss': [1.1, 0.95, 0.78, 0.65, 0.60]
}

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
print("✅ Saved accuracy_plot.png")

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
print("✅ Saved loss_plot.png")


import yaml
import json
import os

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

version = config["model"]["name"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load actual training history
with open("history.json", "r") as f:
    history = json.load(f)

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
acc_path = os.path.join(results_dir, f"accuracy_plot_{version}.png")
plt.savefig(acc_path)
print(f"✅ Saved {acc_path}")

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_path = os.path.join(results_dir, f"loss_plot_{version}.png")
plt.savefig(loss_path)
print(f"✅ Saved {loss_path}")
