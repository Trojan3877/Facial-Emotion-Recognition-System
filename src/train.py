import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

def load_dataset(path="data"):
    X, y = [], []
    for emotion in os.listdir(path):
        folder = os.path.join(path, emotion)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            X.append(extract_features(img))
            y.append(emotion)
    return np.array(X), np.array(y)

def train():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()
