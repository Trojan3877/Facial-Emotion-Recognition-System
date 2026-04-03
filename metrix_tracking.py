import mlflow

def log_metric(metric_name, value):
    mlflow.log_metric(metric_name, value)
    print(f"📈 {metric_name}: {value}")
