import mlflow
import mlflow.pytorch
from datetime import datetime

def log_experiment(model_name, model, params, metrics):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.pytorch.log_model(model, "model")
        print(f"✅ Logged {model_name} to MLflow successfully.")
