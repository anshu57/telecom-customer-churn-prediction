import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

def evaluate_model(run_id):
    """
    Loads a model and test data from an MLflow run, evaluates it,
    and logs the evaluation metrics and a confusion matrix.

    Args:
        run_id (str): The ID of the MLflow run containing the trained model and test data.
    """
    warnings.filterwarnings('ignore')
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    
    # Load the model and test data as artifacts from the specified run
    try:
        # Load the trained model
        model_uri = f"runs:/{run_id}/telco_churn_model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load the test data
        X_test = pd.read_csv(mlflow.get_artifact_uri('X_test.csv').replace('file://', ''))
        y_test = pd.read_csv(mlflow.get_artifact_uri('y_test.csv').replace('file://', '')).squeeze()

    except Exception as e:
        print(f"Error loading model or artifacts from run ID '{run_id}': {e}")
        return

    # Start a new MLflow run for evaluation (optional, but good practice)
    with mlflow.start_run(nested=True):
        y_pred = model.predict(X_test)

        # Log metrics from evaluation
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric('eval_accuracy', report['accuracy'])
        mlflow.log_metric('eval_precision_churn', report['1']['precision'])
        mlflow.log_metric('eval_recall_churn', report['1']['recall'])
        mlflow.log_metric('eval_f1_churn', report['1']['f1-score'])
        
        # Generate and log the confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix on Test Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig("eval_confusion_matrix.png")
        mlflow.log_artifact("eval_confusion_matrix.png")
        plt.close()

    print("Model evaluation complete. Metrics and confusion matrix logged to MLflow.")

if __name__ == "__main__":
    # You need to get the run ID of the latest training run.
    # A common way is to find it from the MLflow UI or programmatically.
    # For a simple example, let's assume you've already run train_model.py
    # and know the run ID. Replace 'YOUR_TRAINING_RUN_ID' with the actual ID.
    latest_run_id = 'YOUR_TRAINING_RUN_ID' 
    evaluate_model(latest_run_id)