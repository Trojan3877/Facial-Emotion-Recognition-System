import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "fer2013.csv"

def preprocess_image(image_array):
    """
    Accepts a flattened 48x48 grayscale array and reshapes, normalizes it.
    """
    image = np.reshape(image_array, (48, 48)).astype("uint8")
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    return image.flatten()

def load_dataset():
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    
    pixels = df["pixels"].tolist()
    emotions = df["emotion"].tolist()

    X = []
    for p in pixels:
        vals = np.array(p.split(), dtype="float32")
        X.append(preprocess_image(vals))

    X = np.array(X)
    y = np.array(emotions)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    print("📥 Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("🤖 Training SVM model...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    print("📊 Evaluating model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print("✔️ Accuracy:", acc)
    print(classification_report(y_test, predictions))

    print("💾 Saving model to model.joblib...")
    joblib.dump(model, "model.joblib")
    print("🎉 Training complete!")

if __name__ == "__main__":
    train()
