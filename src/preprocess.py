# src/preprocess.py

import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Parameters
RAW_DATA_PATH = config["data"]["raw_data_path"]
PROCESSED_DIR = config["data"]["processed_data_dir"]
IMAGE_SIZE = config["data"]["image_size"]
NUM_CLASSES = config["data"]["num_classes"]
VALIDATION_SPLIT = config["model"]["validation_split"]
RANDOM_STATE = config["train"]["random_state"]

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Extract pixel data and convert to array
    pixels = df['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    images = np.stack(pixels.to_numpy())
    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0

    # One-hot encode labels
    labels = df['emotion'].values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    labels_encoded = encoder.fit_transform(labels)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_encoded, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=labels
    )

    # Save processed arrays
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

    print("✅ Data preprocessing complete. Files saved to:", PROCESSED_DIR)

if __name__ == "__main__":
    load_and_preprocess()
