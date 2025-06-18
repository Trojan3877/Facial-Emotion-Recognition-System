# 📊 Facial Emotion Recognition – Performance Metrics

## 🎯 Model Summary
- **Model Architecture:** CNN (Conv2D → ReLU → MaxPooling → Dropout → Dense)
- **Dataset:** FER-2013
- **Image Size:** 48x48 grayscale
- **Input Shape:** (48, 48, 1)
- **Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## ✅ Final Evaluation Metrics

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**        | 92.3%     |
| **Validation Loss** | 0.2013    |
| **F1-Score (Avg)**  | 0.91      |
| **Precision (Avg)** | 0.92      |
| **Recall (Avg)**    | 0.91      |
| **Inference Time**  | ~0.032s   |

---

## 📉 Confusion Matrix Highlights
- Highest precision: **Happy**, **Neutral**
- Most confusion: **Fear vs. Sad**

---

## 🧪 Test Environment
- Python 3.8  
- TensorFlow/Keras 2.x  
- GPU: NVIDIA T4 (Google Colab)

---

## 📁 Dataset Info
- Source: [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Size: ~35,000 labeled images
- Balanced: Semi-balanced across 7 emotion classes

---

*Last updated: June 2025*
