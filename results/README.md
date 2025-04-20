# 📊 Results Summary – Facial Emotion Recognition System

This folder contains quantifiable metrics, visuals, and classification reports from various model configurations.

Each version corresponds to a specific `config/config_<version>.yaml` file used for training.

---

## ✅ Logged Files

- `results_emotion_cnn_v1.txt` – Baseline configuration
- `results_emotion_cnn_v2.txt` – Tuned hyperparameters (lower learning rate, smaller batch size)
- `confusion_matrix_<version>.png` – Visual representation of classification performance
- `accuracy_plot_<version>.png` – Training vs validation accuracy
- `loss_plot_<version>.png` – Training vs validation loss

---

## 🧠 Observations (Example)

| Version            | Accuracy | F1-Score (avg) | Notes |
|--------------------|----------|----------------|-------|
| emotion_cnn_v1     | 0.66     | 0.65           | Baseline config with dropout 0.5 |
| emotion_cnn_v2     | 0.71     | 0.69           | Improved with lower LR & smaller batch size |

---

## 📌 Next Steps

- Try MobileNet or ResNet architecture
- Add webcam real-time detection for deployment
- Experiment with emotion clustering

