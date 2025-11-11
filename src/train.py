import torch
import torch.nn as nn
from mlflow_tracking import log_experiment
from metrix_tracking import log_metric

def train_model():
    model_name = "FacialEmotionRecognition"
    params = {"epochs": 10, "lr": 0.001}

    # Dummy training loop example
    acc, loss = 0.92, 0.08
    model = nn.Linear(10, 2)  # replace with your actual model

    # Log to MLflow
    metrics = {"accuracy": acc, "loss": loss}
    log_experiment(model_name, model, params, metrics)

    # Log to MetrixFlow
    log_metric("accuracy", acc)
    log_metric("loss", loss)

    print(f"🏁 Training complete with acc={acc} loss={loss}")
    return acc
