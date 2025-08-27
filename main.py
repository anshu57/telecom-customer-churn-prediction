import mlflow
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction",
    description="An API to predict whether water is potable (safe to drink) or not."
)

# Set the MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = 'anshu57'
repo_name = 'telecom-customer-churn-prediction'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Load the latest model from MLflow
def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("Best Model", stages=["Production"])
    run_id = versions[0].run_id
    print(run_id)
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")

model = load_model()

class Water(BaseModel):
    SeniorCitizen: bool
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: bool
    Partner_Yes: bool
    Dependents_Yes: bool
    PhoneService_Yes: bool
    MultipleLines_No_phone_service: bool
    MultipleLines_Yes: bool
    InternetService_Fiber_optic: bool
    InternetService_No: bool
    OnlineSecurity_No_internet_service: bool
    OnlineSecurity_Yes: bool
    OnlineBackup_No_internet_service: bool
    OnlineBackup_Yes: bool
    DeviceProtection_No_internet_service: bool
    DeviceProtection_Yes: bool
    TechSupport_No_internet_service: bool
    TechSupport_Yes: bool
    StreamingTV_No_internet_service: bool
    StreamingTV_Yes: bool
    StreamingMovies_No_internet_service: bool
    StreamingMovies_Yes: bool
    Contract_One_year: bool
    Contract_Two_year: bool
    PaperlessBilling_Yes: bool
    PaymentMethod_Credit_card__automatic_: bool
    PaymentMethod_Electronic_check: bool
    PaymentMethod_Mailed_check: bool


# Home Page Route
@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}

# Prediction Endpoint
@app.post("/predict")
def predict(water: Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })
    predicted_value = model.predict(sample)
    
    if predicted_value[0] == 1:
        return {"result": "Water is Consumable"}
    else:
        return {"result": "Water is not Consumable"}
    