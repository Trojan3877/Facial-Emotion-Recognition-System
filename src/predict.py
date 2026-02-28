"""
===============================================================================
FACIAL EMOTION RECOGNITION — INFERENCE SCRIPT (PYTORCH L6 STANDARD)

Purpose:
    CLI-based emotion prediction from image input.

Responsibilities:
    - Load trained PyTorch model
    - Preprocess grayscale image
    - Perform inference
    - Return structured output

Design Principles:
    - No TensorFlow
    - Model logic separated (EmotionModel wrapper)
    - Explicit tensor shape enforcement
    - Deterministic inference mode

Future Improvements:
    - Add face detection preprocessing (MTCNN / Haar cascade)
    - Add batch inference support
    - Add API wrapper (FastAPI)
===============================================================================
"""

import sys
import cv2
import torch
import numpy as np
from src.model import EmotionModel
from src.config.settings import settings


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Loads image and converts to tensor suitable for EmotionCNN.

    Expected Output Shape:
        (1, 1, 48, 48)
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    # Resize to FER standard
    img = cv2.resize(img, (48, 48))

    # Normalize
    img = img.astype("float32") / 255.0

    # Convert to tensor shape (B, C, H, W)
    img = np.expand_dims(img, axis=0)  # (1,48,48)
    img = np.expand_dims(img, axis=0)  # (1,1,48,48)

    tensor = torch.tensor(img, dtype=torch.float32)

    return tensor


# ==============================================================================
# PREDICTION
# ==============================================================================

def predict_emotion(img_path: str):
    """
    Performs full inference pipeline.
    Returns:
        {
            emotion: str,
            confidence: float
        }
    """

    model = EmotionModel(model_path=settings.MODEL_PATH)
    face_tensor = preprocess_image(img_path)

    result = model.predict(face_tensor)

    return result


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        result = predict_emotion(image_path)

        print("\n🎯 Predicted Emotion:", result["emotion"])
        print("📊 Confidence:", result["confidence"], "\n")

    except Exception as e:
        print("❌ Error:", e)
