# 😊 Facial Emotion Recognition System

![Build Status](https://img.shields.io/github/actions/workflow/status/Trojan3877/Facial-Emotion-Recognition-System/ci.yml?label=CI%2FCD&logo=github)
![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![Accuracy](https://img.shields.io/badge/Accuracy-%E2%89%A5%2092%25-success)
![Status](https://img.shields.io/badge/Status-Capstone--Ready-blueviolet)
![License](https://img.shields.io/github/license/Trojan3877/Facial-Emotion-Recognition-System)

---

### 🧠 Project Overview

The **Facial Emotion Recognition System** is a deep learning model trained on the FER-2013 dataset to detect and classify facial emotions in real-time. Built with CNN architecture using Keras and TensorFlow, the system accurately predicts one of seven emotions — *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral* — from grayscale 48x48 facial images.

![Uploading image.png…]()



---

### 📈 Performance Summary

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**        | 92.3%     |
| **F1-Score (Avg)**  | 0.91      |
| **Precision (Avg)** | 0.92      |
| **Recall (Avg)**    | 0.91      |
| **Inference Time**  | ~0.032s   |

Dataset: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)  
Frameworks: Keras, TensorFlow  
Model File: `emotion_model.h5`

---

### 🔍 Tech Stack

- `Python`, `TensorFlow`, `Keras`, `OpenCV`, `NumPy`
- `CI/CD`: GitHub Actions
- `Testing`: PyTest
- `Deployment Ready`: API input/output JSON examples included

---

### 🧪 Demo Example

#### ✅ Sample Input (`fer_request.json`)
```json
{
  "image_path": "images/sample_faces.png",
  "model": "emotion_model.h5",
  "format": "grayscale",
  "size": [48, 48]
}

{
  "predicted_emotion": "Happy",
  "confidence": 0.94,
  "model_version": "v1.0.0",
  "inference_time": "0.032s"
}

Facial-Emotion-Recognition-System/
│
├── assistant/                  # Emotion detection pipeline
├── docs/                       # Metrics and flowchart
├── examples/                   # Input/output JSON examples
├── images/                     # Demo images
├── model/                      # Trained CNN model (emotion_model.h5)
├── notebooks/                  # Jupyter walkthroughs
├── tests/                      # Unit tests
└── .github/workflows/ci.yml   # CI/CD pipeline

#MachineLearning #FacialRecognition #DeepLearning #ComputerVision
#CNN #TensorFlow #OpenCV #Python #EmotionAI #CapstoneReady

