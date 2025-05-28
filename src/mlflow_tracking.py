import mlflow
import os

def setup_mlflow(experiment_name="HeartDiseasePrediction"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def start_run(run_name="RandomForest_Run"):
    return mlflow.start_run(run_name=run_name)
