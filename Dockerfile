# ============================================================
# Facial Emotion Recognition System — Production Dockerfile
# Author: Corey Leath (Trojan3877)
# ============================================================

# ----- Base Image (CPU version of TensorFlow) -----
FROM python:3.10-slim

# ----- Set working directory -----
WORKDIR /app

# ----- Install system dependencies -----
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ----- Copy project files -----
COPY requirements.txt .
COPY src/ ./src/
COPY fer2013.csv ./  # Optional if you include dataset locally
COPY emotion_model_final.h5 ./  # Include trained model

# ----- Install Python dependencies -----
RUN pip install --no-cache-dir -r requirements.txt

# ----- Streamlit config: prevents network prompts -----
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml

# ----- Expose Streamlit port -----
EXPOSE 8501

# ----- Default command: Launch Streamlit UI -----
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
