import mlflow
from mlflow.tracking import MlflowClient
import json
import dagshub


def transition_best_model_to_production(model_name: str, best_model_info_file: str):
    """
    Transitions the best-performing model to the 'Production' stage
    in the MLflow Model Registry.

    Args:
        model_name (str): The name of the registered model.
        best_model_info_file (str): The path to the JSON file containing the best
                                    model's run ID and metrics.
    """
    print("Connecting to MLflow Tracking Server...")

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
        
    # Initialize the MLflow client
    client = MlflowClient()
    
    # Check if the model exists in the registry
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.RestException:
        print(f"Model '{model_name}' not found in the registry. Exiting.")
        return

    # Load the best model's run ID from the JSON file
    try:
        with open(best_model_info_file, 'r') as f:
            best_model_info = json.load(f)
        best_run_id = best_model_info['best_run_id']
        best_model_name = best_model_info['best_model_name']
        print(f"Found best model info for Run ID: {best_run_id}")
    except FileNotFoundError:
        print(f"Error: The file '{best_model_info_file}' was not found. Exiting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{best_model_info_file}'. Exiting.")
        return

    # Find the model version that corresponds to the best run
    model_versions = client.search_model_versions(f"run_id='{best_run_id}'")
    if not model_versions:
        print(f"No model version found for run_id '{best_run_id}'. Exiting.")
        return

    # Get the latest version for the best run ID
    latest_version = max(model_versions, key=lambda mv: mv.version)
    new_version = latest_version.version
    
    print(f"Best model's version is: {new_version}")

    # Transition any existing Production models to Staging
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == 'Production':
            print(f"Transitioning existing Production model (version {mv.version}) to 'Staging'.")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage='Staging',
                archive_existing_versions=False
            )

    # Transition the new best model to Production
    print(f"Transitioning new model (version {new_version}) to 'Production'.")
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage='Production',
        archive_existing_versions=False
    )

    print(f"Model '{model_name}' version {new_version} is now in 'Production'.")

if __name__ == "__main__":
    # Define the name of the registered model and the info file
    registered_model_name = "TelcoChurnPredictionModel"
    best_model_json_file = "reports/best_model.json"
    
    transition_best_model_to_production(registered_model_name, best_model_json_file)