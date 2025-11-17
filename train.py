"""
=========================================================
FACIAL EMOTION RECOGNITION — TRAINING SCRIPT (L5/L6 Level)
Author: Trojan3877 (Corey Leath)
Description:
    - Loads FER-2013 dataset
    - Preprocesses pixel strings into 48x48 grayscale images
    - Builds a CNN model (L5/L6 quality)
    - Trains with callbacks (EarlyStopping, Checkpoints)
    - Saves trained model + history.json
=========================================================
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================
tf.random.set_seed(42)
np.random.seed(42)

# =========================================================
# 2. LOAD DATASET
# =========================================================
CSV_PATH = "fer2013.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Dataset not found! Expected at: {CSV_PATH}")

print("📥 Loading FER-2013 dataset...")
df = pd.read_csv(CSV_PATH)

# Validate required columns
required_cols = {"emotion", "pixels", "Usage"}
if not required_cols.issubset(df.columns):
    raise ValueError("❌ CSV missing required columns: emotion, pixels, Usage")

print("✅ Dataset loaded successfully.")

# =========================================================
# 3. PREPROCESSING
# =========================================================
print("🧹 Preprocessing pixel data...")

def preprocess_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype="float32")
    return pixels.reshape(48, 48, 1) / 255.0

X = np.array(list(map(preprocess_pixels, df["pixels"])))
y = tf.keras.utils.to_categorical(df["emotion"], num_classes=7)

print(f"📊 Dataset shape: {X.shape}, Labels: {y.shape}")

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"📚 Train: {X_train.shape}, Validation: {X_val.shape}")

# =========================================================
# 4. MODEL ARCHITECTURE (L6 QUALITY)
# =========================================================
print("🏗 Building CNN model...")

def build_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

model = build_model()
model.summary()

# =========================================================
# 5. CALLBACKS
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "emotion_model_best.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================================================
# 6. TRAINING
# =========================================================
print("🚀 Starting training...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

print("🎉 Training complete!")

# =========================================================
# 7. SAVE MODEL + HISTORY
# =========================================================
model.save("emotion_model_final.h5")
print("💾 Final model saved as 'emotion_model_final.h5'")

# Save training history
with open("history.json", "w") as f:
    json.dump(history.history, f, indent=4)

print("📈 Training history saved to 'history.json'")

# =========================================================
print("\n✅ Training workflow complete! Your model is L5/L6 ready.\n")
