from fastapi import FastAPI, UploadFile
from src.predict import predict_emotion
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    emotion = predict_emotion(temp_path)
    return {"emotion": emotion}
