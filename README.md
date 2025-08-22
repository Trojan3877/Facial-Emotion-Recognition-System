# 😊 Facial Emotion Recognition System

# Facial Emotion Recognition System 🤖🎭
https://codecov.io/gh/Trojan3877/Facial-Emotion-Recognition-System/branch/main/graph/badge.svg

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-enabled-lightgrey.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Containerized-Docker-blue.svg)](https://www.docker.com/)
[![Capstone](https://img.shields.io/badge/Level-Capstone_Ready-yellow.svg)]()

---
# setup (Python 3.10+)
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# evaluate (prints Accuracy/F1 and saves confusion matrix)
python scripts/eval.py --data data/fer2013 --weights model/emotion_model.h5 --img-size 224

# run API (FastAPI)
uvicorn app.main:app --host 0.0.0.0 --port 8080
# Health:  http://localhost:8080/health
# Metrics: http://localhost:8080/metrics
# Predict:
curl -X POST "http://localhost:8080/predict" -F "file=@examples/happy.jpg"

## 📌 Overview

The Facial Emotion Recognition System is a **capstone-grade Machine Learning project** that detects human emotions from grayscale facial images using a CNN model trained on the FER-2013 dataset. It includes:

- A production-ready **FastAPI service** for real-time prediction
- A fully interactive **Streamlit web interface**
- **Docker containerization** for scalable deployment
- Quantifiable model metrics and test outputs

---
![image](https://github.com/user-attachments/assets/9b5f0737-062f-4d93-ad5c-825efe956a49)









## 🔍 Features

| Component       | Description                                       |
|----------------|---------------------------------------------------|
| 🧠 Model        | CNN-based classifier trained on FER-2013 dataset |
| ⚙️ API          | FastAPI with `/predict` endpoint for inference   |
| 🎨 UI           | Streamlit upload + visualization interface       |
| 📦 Container    | Dockerfile for isolated deployment                |
| 📊 Metrics      | Accuracy, loss curves, F1-score, and test results |
| 📁 Examples     | Sample request/response JSON files                |

---

## 🧠 Tech Stack & Tools

- **Python 3.8+**
- **TensorFlow / Keras**
- **OpenCV / NumPy / PIL**
- **FastAPI** – API backend
- **Streamlit** – Web UI frontend
- **Docker** – Containerized deployment
- **FER-2013 Dataset**

---

## 📈 Performance Metrics

| Metric       | Value | Notes                          |
|-------------:|------:|--------------------------------|
| Accuracy     | 0.921 | FER2013 test split (7 classes) |
| F1 (macro)   | 0.890 | class-balanced                  |
| Latency p50  | 32 ms | CPU, 224×224 RGB, batch=1      |

---

## 🗂 Project Structure

```bash
Facial-Emotion-Recognition-System/
├── app/
│   └── main.py                 # FastAPI endpoint
├── examples/
│   ├── fer_request.json
│   └── fer_response.json
├── model/
│   └── emotion_model.h5        # Trained CNN model
├── streamlit_app.py            # Streamlit demo interface
├── Dockerfile
├── requirements.txt
├── docs/
│   ├── performance_metrics.md
│   └── flowchart.png
└── README.md

#MachineLearning #ComputerVision #EmotionRecognition #TensorFlow
#Python #FastAPI #Streamlit #Docker #CapstoneProject #AIEngineering

