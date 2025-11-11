#!/bin/bash
echo "🚀 Starting Facial Emotion Recognition pipeline..."
export AIRFLOW_HOME=./airflow

airflow db init
airflow scheduler &
airflow webserver -p 8080 &

mlflow ui --port 5000 --backend-store-uri ./mlruns &
metrixflow ui --port 8081 &

wait
