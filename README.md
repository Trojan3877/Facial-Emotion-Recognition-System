<p align="center">

  <!-- General Repo Badges -->
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Contributions-Welcome-orange?style=for-the-badge" />

  <br/>

  <!-- Tech Stack Badges -->
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-GPT--4o--mini-412991?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/RAG-Enabled-7b42f6?style=for-the-badge&logo=googlecloud" />
  <img src="https://img.shields.io/badge/Docker-0db7ed?style=for-the-badge&logo=docker&logoColor=white" />

  <br/>

  <!-- CI/CD Badges -->
  <img src="https://img.shields.io/github/actions/workflow/status/Trojan3877/Disease_Prediction_Capstone/ci.yml?style=for-the-badge&label=CI%20Pipeline" />
  <img src="https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge&logo=pytest" />

  <br/>

  <!-- Python & API Badges -->
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/API-FastAPI-green?style=for-the-badge" />

</p>

## 📌 **Project Overview**

This project is a complete **emotion-analysis AI microservice**, combining:

1. **CNN Emotion Recognition** – A trained PyTorch model predicts human emotion from images.  
2. **RAG Psychology Retrieval** – Retrieves behavioral-science context for each detected emotion.  
3. **LLM Explanation Engine** – Uses GPT-4o-mini (or any MCP-compatible model) to generate friendly, psychologically accurate interpretations.  
4. **FastAPI Microservice** – Production-ready REST API.  
5. **Dockerized Deployment** – Fully containerized with optional Docker Compose.  
6. **CI/CD Pipeline** – Automated testing with GitHub Actions.

This repository demonstrates skills in **Machine Learning Engineering, Applied AI, LLM Systems, RAG, FastAPI, Docker, and CI/CD** — the same tech used across Big Tech AI teams.

---

## 🧩 **System Architecture**

           ┌──────────────────┐
           │   User Uploads   │
           │      Image       │
           └───────┬──────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   CNN Emotion Classifier (PyTorch)│
    └──────────────────┬───────────────┘
                       │ top emotions
                       ▼
      ┌──────────────────────────────────────┐
      │ Psychology RAG Retriever             │
      │ (context lookup for emotions)        │
      └──────────────────┬───────────────────┘
                         │ context
                         ▼
    ┌──────────────────────────────────────────┐
    │    LLM Explanation Module (GPT-4o)       │
    │ Combines predictions + psychology info   │
    └──────────────────┬───────────────────────┘
                       ▼
              JSON Response Output

---
## 🧩 System Design Diagram
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/97912859-76ce-400d-9318-a208c9984a22" />


## 🚀 **Key Features**

### ✔ CNN-Based Emotion Recognition  
- Preprocessing pipeline (Resize → Normalize → ToTensor)  
- Softmax output with top-k emotion ranking  
- Supports CPU & GPU  

### ✔ RAG (Retrieval-Augmented Generation)  
- Psychology-focused emotion database  
- Modular for FAISS / Pinecone upgrades  
- Interpretable scientific grounding  

### ✔ LLM Interpretation Layer  
- GPT-4o-mini or any MCP-compatible model  
- Converts raw predictions into understandable emotional insights  

### ✔ FastAPI Backend  
- Clean route design  
- Automatic input validation  
- CORS-enabled  
- Production-ready  

### ✔ MLOps Quality  
- Dockerfile + docker-compose  
- requirements.txt (fully pinned)  
- `.env.example` security practice  
- Pytest test suite  
- GitHub Actions CI pipeline  

---

## ⚙️ **Tech Stack**

| Layer | Technology |
|-------|------------|
| ML Inference | PyTorch, TorchVision |
| API Framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini / MCP compatible |
| RAG | Custom Python retriever |
| Infrastructure | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | PyTest |

---

## 📦 **Installation**

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Trojan3877/Disease_Prediction_Capstone
cd Disease_Prediction_Capstone
pip install -r requirements.txt
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
MODEL_PATH=models/emotion_cnn.pth
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
docker-compose up --build
{
  "status": "online"
}
pytest -q

