# ---------------------------------------------------------
# 1. Base Image — Official Python 3.10 Slim
# ---------------------------------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# 2. Install System Dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------
# 3. Set Working Directory
# ---------------------------------------------------------
WORKDIR /app


# ---------------------------------------------------------
# 4. Copy Requirements & Install
# ---------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ---------------------------------------------------------
# 5. Copy Project Files
# ---------------------------------------------------------
COPY . .


# ---------------------------------------------------------
# 6. Expose API Port
# ---------------------------------------------------------
EXPOSE 8000


# ---------------------------------------------------------
# 7. Run FastAPI with Uvicorn
# ---------------------------------------------------------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
