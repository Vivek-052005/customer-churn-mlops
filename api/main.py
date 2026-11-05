
import logging
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "logistic_regression.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessor"
    / "preprocessor.joblib"
)


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to leave.",
    version="1.0.0"
)


# --------------------------------------------------
# 4. LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

model = None
preprocessor = None


@app.on_event("startup")
def load_model_artifacts():

    global model, preprocessor

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    logger.info("Model and preprocessor loaded successfully")


# --------------------------------------------------
# 5. CUSTOMER INPUT SCHEMA
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(ge=0, le=1, examples=[0])
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(ge=0, examples=[12])
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["Yes"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(
        examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(ge=0, examples=[70.50])
    TotalCharges: float = Field(ge=0, examples=[850.00])


# --------------------------------------------------
# 6. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running",
        "model": "Logistic Regression"
    }


# --------------------------------------------------
# 7. HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


# --------------------------------------------------
# 8. MODEL INFORMATION ENDPOINT
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model_name": "Logistic Regression",
        "model_file": MODEL_PATH.name,
        "preprocessor_file": PREPROCESSOR_PATH.name
    }


# --------------------------------------------------
# 9. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        # Convert customer input into a DataFrame
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # Apply the saved preprocessing transformations
        processed_data = preprocessor.transform(customer_df)

        # Predict churn
        prediction = int(model.predict(processed_data)[0])

        # Calculate churn probability
        probability = float(
            model.predict_proba(processed_data)[0][1]
        )

        # Convert prediction into readable output
        result = (
            "Customer likely to churn"
            if prediction == 1
            else "Customer likely to stay"
        )

        return {
            "prediction": prediction,
            "result": result,
            "churn_probability": round(probability, 4),
            "churn_percentage": round(probability * 100, 2)
        }

    except Exception as error:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


