import pandas as pd
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
import json

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
mlflow.set_experiment("Logistic_Regression")


def train_multiple_models_with_penalties(training_data_path, params_path = 'params.yaml'):
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
        train_df = pd.read_csv(os.path.join(training_data_path,'train_processed.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{training_data_path}' exist.")
        return


 
    
    # Apply SMOTE to the training data only
    smote = SMOTE(random_state=params['smote_random_state'])
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    best_metric = 0
    #for penalty in params['penalties']:
    with mlflow.start_run(run_name=f"LogisticRegression") as parent_run:
        parent_run_id = parent_run.info.run_id
        for penalty in params['penalties']:
            with mlflow.start_run(nested=True, run_name=f"LogisticRegression_{penalty}"):
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
                
                y_pred_train = model.predict(X_train)
                # try:
                #     with open(model_name, "wb") as file:
                #         pickle.dump(model, file)
                # except Exception as e:
                #     raise Exception(f"Error saving model to {model_name}: {e}")

                report = classification_report(y_train, y_pred_train, output_dict=True)
                mlflow.log_metric('train_accuracy', report['accuracy'])
                mlflow.log_metric('train_precision_no_churn', report['0']['precision'])
                mlflow.log_metric('train_recall_no_churn', report['0']['recall'])
                mlflow.log_metric('train_f1_no_churn', report['0']['f1-score'])
                mlflow.log_metric('train_precision_churn', report['1']['precision'])
                mlflow.log_metric('train_recall_churn', report['1']['recall'])
                mlflow.log_metric('train_f1_churn', report['1']['f1-score'])
                
                signature = infer_signature(X_train,model.predict(X_train))
                mlflow.sklearn.log_model(sk_model=model, artifact_path=f"model_{penalty}", signature=signature)

                # Generate and log confusion matrix
                cm = confusion_matrix(y_train, y_pred_train)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
                plt.title(f'Confusion Matrix for Logistic Regression ({penalty})')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                if not os.path.isdir('./reports'):
                    os.makedirs('./reports')
                file_name = f'train_{penalty}.json'
                with open(os.path.join('./reports',file_name), 'w') as file:
                        json.dump(report, file, indent=4)
                
                plt.savefig(f"reports/confusion_matrix_{penalty}.png")
                mlflow.log_artifact(f"reports/confusion_matrix_{penalty}.png")
                plt.close()

    print("All models trained and evaluated. Results logged to MLflow/DagsHub.")

if __name__ == "__main__":
    params_path = "params.yaml"
    data_path = "./data/processed"
    train_multiple_models_with_penalties(data_path)