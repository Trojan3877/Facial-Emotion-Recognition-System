from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from train import train_model
from evaluate import evaluate_model

default_args = {"owner": "corey", "start_date": datetime(2025, 1, 1)}

with DAG(
    "facial_emotion_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "emotion"],
) as dag:

    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    eval_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    train_task >> eval_task
