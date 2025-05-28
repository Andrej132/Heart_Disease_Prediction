import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import mlflow
from mlflow_tracking import setup_mlflow, start_run
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import DATA_DIR, MODEL_DIR, IMAGES_DIR, LOG_DIR

logging.basicConfig(
    filename=f"{LOG_DIR}/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_model():
    setup_mlflow(experiment_name="HeartDiseasePrediction")
    print("Current experiment:", mlflow.get_experiment_by_name("HeartDiseasePrediction"))
    with start_run("RandomForest_Evaluation"):
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR, exist_ok=True)

        test_path = os.path.join(DATA_DIR, "test.csv")
        logging.info(f"Loading test data from {test_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found at {test_path}")
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=["HeartDisease"])
        y_test = test_df["HeartDisease"]

        model_path = os.path.join(MODEL_DIR, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_txt = classification_report(y_test, y_pred)
        logging.info("Classification report generated.")

        report_txt_path = os.path.join(IMAGES_DIR, "classification_report.txt")
        with open(report_txt_path, "w") as f:
            f.write(report_txt)
        mlflow.log_artifact(report_txt_path, artifact_path="eval")

        mlflow.log_metrics({
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "accuracy": report["accuracy"]
        })

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="eval")
        logging.info(f"Confusion matrix saved to {cm_path}.")

        importances = model.feature_importances_
        feature_names = X_test.columns
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fi_path = os.path.join(IMAGES_DIR, "feature_importance.png")
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path, artifact_path="eval")
        logging.info(f"Feature importance plot saved to {fi_path}.")

if __name__ == "__main__":
    evaluate_model()

