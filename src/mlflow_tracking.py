import mlflow
import os

def setup_mlflow(experiment_name="HeartDiseasePrediction"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

def start_run(run_name="RandomForest_Run"):
    return mlflow.start_run(run_name=run_name)
