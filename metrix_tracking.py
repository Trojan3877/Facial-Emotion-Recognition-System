from metrixflow import Tracker

tracker = Tracker(project="FacialEmotionRecognition", uri="./metrics.db")

def log_metric(metric_name, value):
    tracker.log_metric(metric_name, value)
    tracker.commit()
    print(f"📈 {metric_name}: {value}")
