# 🎭 Facial Emotion Recognition System  
> Production-Ready AI/ML Pipeline with Apache Airflow, MLflow, and MetrixFlow

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-green)
![Apache Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-blue)
![MetrixFlow](https://img.shields.io/badge/Monitoring-MetrixFlow-purple)
![Docker](https://img.shields.io/badge/Containerized-Yes-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🧠 Overview
The **Facial Emotion Recognition System** uses deep learning to classify human emotions from images.  
This version is integrated with full **MLOps tracking** using:
- **Apache Airflow** for orchestration  
- **MLflow** for experiment tracking  
- **MetrixFlow** for real-time performance visualization  

The system demonstrates an end-to-end AI pipeline suitable for **AI Research Engineers** and **ML Researchers** — with metrics reproducibility, containerized environments, and model lineage.

---

## 🧰 Tech Stack
| Category | Tools |
|-----------|-------|
| Frameworks | PyTorch, FastAPI, Streamlit |
| Orchestration | Apache Airflow |
| Experiment Tracking | MLflow |
| Monitoring | MetrixFlow |
| Containerization | Docker |
| Hardware | NVIDIA RTX GPU (CUDA 12.1) |

---

## 🧱 Architecture
                ┌──────────────────────────────┐
                │         Apache Airflow        │
                │    (Schedules + DAGs)         │
                └──────────────┬────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │           MLflow             │
                │ Logs params, metrics, models │
                └──────────────┬────────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │         MetrixFlow           │
                │ Real-time dashboard tracking │
                └──────────────────────────────┘

---

## 🧪 Pipeline Stages
| Stage | Description |
|--------|--------------|
| **1. Data Preprocessing** | Cleans and augments facial image datasets |
| **2. Model Training** | Uses CNN for emotion classification (Happy, Sad, Angry, etc.) |
| **3. Evaluation** | Calculates metrics (Accuracy, Precision, F1, Recall) |
| **4. Logging** | Sends results to MLflow + MetrixFlow |
| **5. Orchestration** | Scheduled daily via Apache Airflow |

---

## ⚙️ How to Run (Locally)

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Trojan3877/Facial_Emotion_Recognition_System.git
cd Facial_Emotion_Recognition_System

