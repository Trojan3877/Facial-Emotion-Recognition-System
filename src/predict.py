"""
=========================================================
FACIAL EMOTION RECOGNITION — INFERENCE SCRIPT (L5/L6 Level)
Author: Trojan3877 (Corey Leath)
Description:
    - Loads trained FER model
    - Accepts an image path
    - Preprocesses and predicts the emotion
    - Returns readable label + confidence
=========================================================
"""

import cv2
import numpy as np
import tensorflow as tf

# Emotion categories for FER-2013 dataset
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

MODEL_PATH = "emotion_model_final.h5"

# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


# ---------------------------------------------------------
# Preprocess image (L6 production style)
# ---------------------------------------------------------
def preprocess_image(img_path: str):
    """Loads and preprocesses an image for FER model."""

    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")

    # Detect face (optional improvement: Haar cascades)
    # Resize to 48x48 for FER-2013
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0

    # Expand dims → (1, 48, 48, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


# ---------------------------------------------------------
# Predict emotion from image
# ---------------------------------------------------------
def predict_emotion(img_path: str):
    """Runs prediction and returns label + confidence."""
    img = preprocess_image(img_path)

    preds = model.predict(img)
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return EMOTION_LABELS[pred_idx], confidence


# ---------------------------------------------------------
# Main execution (CLI)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("⚠️ Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        label, conf = predict_emotion(image_path)
        print(f"\n🎯 Predicted Emotion: **{label}**")
        print(f"📊 Confidence: {conf:.4f}\n")

    except Exception as e:
        print(f"❌ Error: {e}")
