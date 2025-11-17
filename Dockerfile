# ============================================================
# GPU-ENABLED DOCKERFILE FOR FACIAL EMOTION RECOGNITION SYSTEM
# Base image includes CUDA 11.8 + cuDNN 8
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev python3-venv \
    libglib2.0-0 libsm6 libxext6 libxrender-dev wget git && \
    rm -rf /var/lib/apt/lists/*

# Copy project
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.15.0
RUN pip3 install -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Launch FastAPI + Streamlit
CMD uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
