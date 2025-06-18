from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
import numpy as np
import cv2
import io

app = FastAPI(title="Facial Emotion Recognition API")

# Load the trained model once on startup
model = load_model("model/emotion_model.h5")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = preprocess_image(content)
        prediction = model.predict(image)
        confidence = float(np.max(prediction))
        label = class_names[np.argmax(prediction)]

        return JSONResponse({
            "predicted_emotion": label,
            "confidence": round(confidence, 4),
            "status": "success"
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "failure"
        }, status_code=500)
