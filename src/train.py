# src/train.py

import os
import numpy as np
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Parameters
INPUT_SHAPE = tuple(config["model"]["input_shape"])
EPOCHS = config["model"]["epochs"]
BATCH_SIZE = config["model"]["batch_size"]
LEARNING_RATE = config["model"]["learning_rate"]
DROPOUT_RATE = config["model"]["dropout_rate"]
MODEL_SAVE_PATH = config["train"]["model_save_path"]
EARLY_STOPPING_PATIENCE = config["train"]["early_stopping_patience"]

# Load preprocessed data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")

# Define CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(2, 2),
        Dropout(DROPOUT_RATE),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(DROPOUT_RATE),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(config["data"]["num_classes"], activation='softmax')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train model
model = build_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
