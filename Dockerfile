# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install tensorflow keras opencv-python fastapi uvicorn streamlit

# Expose port for FastAPI
EXPOSE 8000

# Default command to run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
