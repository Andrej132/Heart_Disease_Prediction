import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from config import DATA_DIR, LOG_DIR, MODEL_DIR
from sklearn.model_selection import StratifiedKFold, cross_val_score
from data_preprocessing import preprocess_data, load_data
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow_tracking import setup_mlflow, start_run

logging.basicConfig(filename=f"{LOG_DIR}/pipeline.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Starting the training process.")
    setup_mlflow()

    with start_run("RandomForest_Training") as run:
        data_path = f"{DATA_DIR}/heart.csv"
        logging.info(f"Loading data from {data_path}.")
        data = load_data(data_path)

        logging.info("Preprocessing the data.")
        preprocess_data(data)

        logging.info("Splitting the data into input features and target variable.")
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

        X_train = train_df.drop(columns=['HeartDisease'])
        y_train = train_df['HeartDisease']

        rf_params = {
            "max_depth": 10,
            "min_samples_leaf": 4,
            "min_samples_split": 10,
            "n_estimators": 50,
            "random_state": 42
        }

        mlflow.log_params(rf_params)
        logging.info("Initializing the RandomForestClassifier.")
        rf = RandomForestClassifier(**rf_params)

        logging.info("Performing cross-validation.")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')

        mean_score = np.mean(scores)
        mlflow.log_metric("cv_accuracy_mean", mean_score)
        logging.info(f"Cross-validation scores: {scores}")
        logging.info(f"Mean accuracy: {mean_score:.2f}")

        logging.info("Fitting the model on the training data.")
        rf.fit(X_train, y_train)

        signature = infer_signature(X_train, rf.predict(X_train))
        input_example = X_train.head(1)

        mlflow.sklearn.log_model(rf, "model", signature=signature, input_example=input_example)

        model_path = os.path.join(MODEL_DIR, "model.pkl")
        joblib.dump(rf, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        logging.info(f"Model saved and logged to MLflow at {model_path}.")

        train_csv_path = os.path.join(DATA_DIR, "train.csv")
        mlflow.log_artifact(train_csv_path, artifact_path="data")

        logging.info("Training process completed successfully.")


if __name__ == "__main__":
    train_model()



