import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
import yaml
import dagshub
import pickle

# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_TOKEN env variable is not set")


# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

print("experimenting LR model")
# DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner = 'anshu57'
repo_name = 'telecom-customer-churn-prediction'
dagshub.init(repo_owner='anshu57', repo_name='telecom-customer-churn-prediction', mlflow=True)
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Logistic Regression")
print("experimenting LR model")

def train_multiple_models_with_penalties(features_path, target_path, params_path='params.yaml'):
    """
    Loads prepared data, applies SMOTE, trains a Logistic Regression model
    for each specified penalty, and logs the results to MLflow.

    Args:
        features_path (str): The file path to the prepared features CSV.
        target_path (str): The file path to the prepared target CSV.
        params_path (str): The file path to the parameters YAML file.
    """
    warnings.filterwarnings('ignore')
    
    with open(params_path) as f:
        params = yaml.safe_load(f)['train']
    
    try:
        X = pd.read_csv(features_path)
        y = pd.read_csv(target_path).squeeze()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{features_path}' and '{target_path}' exist.")
        return


    # Split the data once for all models to ensure a fair comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y)
    
    # Apply SMOTE to the training data only
    smote = SMOTE(random_state=params['smote_random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    for penalty in params['penalties']:
        with mlflow.start_run(run_name=f"LogisticRegression_{penalty}"):
            print(f"Starting MLflow run for Logistic Regression with penalty: {penalty}")

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param('penalty', penalty)
            
            # Initialize the Logistic Regression model with the specified penalty
            if penalty == 'elasticnet':
                # 'saga' solver is required for elasticnet penalty
                model = LogisticRegression(penalty=penalty, l1_ratio=params['l1_ratio'], solver='saga', random_state=params['random_state'])
            else:
                # 'liblinear' solver is required for l1 and l2 penalties
                model = LogisticRegression(penalty=penalty, solver='liblinear', random_state=params['random_state'])

            model.fit(X_train_resampled, y_train_resampled)
            
            y_pred = model.predict(X_test)
            # try:
            #     with open(model_name, "wb") as file:
            #         pickle.dump(model, file)
            # except Exception as e:
            #     raise Exception(f"Error saving model to {model_name}: {e}")

            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metric('accuracy', report['accuracy'])
            mlflow.log_metric('precision_no_churn', report['0']['precision'])
            mlflow.log_metric('recall_no_churn', report['0']['recall'])
            mlflow.log_metric('f1_no_churn', report['0']['f1-score'])
            mlflow.log_metric('precision_churn', report['1']['precision'])
            mlflow.log_metric('recall_churn', report['1']['recall'])
            mlflow.log_metric('f1_churn', report['1']['f1-score'])
            
            signature = infer_signature(X_test,model.predict(X_test))
            mlflow.sklearn.log_model(sk_model=model, artifact_path=f"model_{penalty}", signature=signature)

            # Generate and log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
            plt.title(f'Confusion Matrix for Logistic Regression ({penalty})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(f"confusion_matrix_{penalty}.png")
            mlflow.log_artifact(f"confusion_matrix_{penalty}.png")
            plt.close()

    print("All models trained and evaluated. Results logged to MLflow/DagsHub.")

if __name__ == "__main__":
    params_path = "params.yaml"
    data_path = "./data/processed"
    train_multiple_models_with_penalties(os.path.join(data_path,'x_features.csv'), os.path.join(data_path,'y_target.csv'))