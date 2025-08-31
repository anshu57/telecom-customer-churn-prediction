import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub

# Initialize Dags hub and set up mlflow experiment tracking
# dagshub.init(repo_owner='anshu57', repo_name='telecom-customer-churn-prediction', mlflow=True)

import os
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN env variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner = 'anshu57'
repo_name = 'telecom-customer-churn-prediction'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Logistic Regression")




# Load the run ID and model name from the saved JSON file
reports_path = "reports/best_model.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['best_run_id'] # Fetch run id from the JSON file
model_name = run_info['best_model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Create the model URI
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

# Register the model
reg = mlflow.register_model(model_uri, model_name)

# Get the model version
model_version = reg.version  # Get the registered model version

# Transition the model version to Staging
new_stage = "Staging"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")