# =====================================================
# FastAPI Inference API for Facial Emotion Recognition
# Author: Corey Leath (Trojan3877)
# =====================================================

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("🚀 GPU DETECTED — Using GPU for inference.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("⚠ No GPU detected — defaulting to CPU.")


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("emotion_model_final.h5")

# Emotion labels (same order used during training)
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Upload an image and receive an emotion prediction.",
    version="1.0",
)

# --------------- Helper: Preprocess the image ----------------
def preprocess_image(image_bytes):
    # Convert to numpy array
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize to 48x48 for FER
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 48, 48, 1)
    return img

# ----------------------- API ROUTES --------------------------

@app.get("/")
def root():
    return {"message": "Facial Emotion Recognition API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        pred = model.predict(img)[0]
        emotion_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))

        return JSONResponse({
            "emotion": emotion_labels[emotion_idx],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
