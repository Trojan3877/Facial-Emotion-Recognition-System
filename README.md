# 🎭 Facial Emotion Recognition System  
# 🎭 Facial Emotion Recognition System (Scikit-Learn Edition)
# 🎭 Facial Emotion Recognition System (FER-2013)
### **Author:** Corey Leath (GitHub: [Trojan3877](https://github.com/Trojan3877))  
**Status:** Production-Ready | L5/L6 Quality | Deployable | GPU-Accelerated  
<p align="center">

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python">

  <!-- TensorFlow -->
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">

  <!-- Streamlit -->
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">

  <!-- OpenCV -->
  <img src="https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">

  <!-- NumPy -->
  <img src="https://img.shields.io/badge/NumPy-1.26.4-4D77CF?logo=numpy&logoColor=white" alt="NumPy">

  <!-- Pandas -->
  <img src="https://img.shields.io/badge/Pandas-2.2.1-150458?logo=pandas&logoColor=white" alt="Pandas">

  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">

  <!-- Status -->
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Status">

  <!-- GitHub Repo Size -->
  <img src="https://img.shields.io/github/repo-size/Trojan3877/Facial-Emotion-Recognition-System?color=blue" alt="Repo Size">

</p>

---

## 📘 Overview  
This project is a full **Facial Emotion Recognition (FER)** system built using the **FER-2013 dataset**.  
It includes:

- A complete **training pipeline** (preprocessing → model → metrics)
- A production-level **CNN architecture**
- Inference via **predict.py**
- A clean UI via **Streamlit** (`streamlit_app.py`)
- Full visualization (accuracy curves, confusion matrix)
- A professional ML engineering structure  
- Deployment-ready components for portfolio use

This project demonstrates **Machine Learning Engineer (L5/L6) capabilities**, including:
- Model design  
- Training with callbacks  
- Data validation  
- Visualization  
- Inference API  
- Deployment readiness  
- Reproducibility (`requirements.txt`)  

---

# 🧠 Architecture  
Facial Emotion Recognition System
│
├── src/
│ ├── train.py # Train CNN model (L6 quality)
│ ├── predict.py # Run inference on images
│ ├── streamlit_app.py # Web UI for predictions
│ ├── visualize.py # Plots + confusion matrix
│
├── emotion_model_final.h5 # Saved trained model (if included)
├── fer2013.csv # Training dataset
├── history.json # Training history
├── requirements.txt # Reproducible environment
└── README.md # You are here


---

# 🚀 Features  
### ✔ **7-class emotion detection**
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

### ✔ **L6-quality CNN model**
- 3 convolutional blocks  
- BatchNorm + Dropout for stability  
- Final Dense classifier  
- Adaptive learning rate  
- EarlyStopping + Checkpointing  

### ✔ **Training visualizations**
Saved in `/plots/`:
- accuracy_curve.png  
- loss_curve.png  
- confusion_matrix.png  
- classification_report.txt  

### ✔ **Streamlit Web App**
Run with:

