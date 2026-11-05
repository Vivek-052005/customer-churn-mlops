
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    # This context manager triggers FastAPI startup and shutdown events.
    with TestClient(app) as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API is running"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert result["preprocessor_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_name"] == "Logistic Regression"


def test_prediction_endpoint(client):
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 850.0
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["churn_percentage"] <= 100
    assert "result" in result
