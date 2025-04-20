# src/evaluate.py

import numpy as np
import os
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load preprocessed data and model
X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")
model = load_model(config["train"]["model_save_path"])

# Predict
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification report
report = classification_report(y_true, y_pred, target_names=[
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
])
print("📊 Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            yticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

import datetime

# Evaluate overall accuracy
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

# Save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
version = config["model"]["name"]  # Automatically pulls config name

results_path = f"results/results_{version}.txt"
os.makedirs("results", exist_ok=True)

with open(results_path, "w") as f:
    f.write(f"Facial Emotion Recognition - {version}\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    f.write(f"Loss: {loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    
print(f"✅ Evaluation metrics saved to {results_path}")

plt.savefig(f"results/confusion_matrix_{version}.png")
print(f"✅ Confusion matrix saved as 'confusion_matrix_{version}.png'")


