
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessor"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def build_preprocessor():
    """Create numerical and categorical preprocessing pipelines."""

    # Numerical columns
    numerical_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Categorical columns
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    # Numerical preprocessing:
    # 1. Fill missing numerical values with median
    # 2. Scale numerical values
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing:
    # 1. Fill missing values with most frequent category
    # 2. Convert categories into numerical columns
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def process_features():
    """Fit preprocessing on training data and transform both datasets."""

    logger.info("Starting feature engineering")

    # Check input files
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate features and target
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    logger.info("Training features shape: %s", X_train.shape)
    logger.info("Testing features shape: %s", X_test.shape)

    # Create the preprocessor
    preprocessor = build_preprocessor()

    # IMPORTANT:
    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the training-fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    # Save the fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    logger.info("Preprocessor saved: %s", PREPROCESSOR_PATH)
    logger.info("Processed training shape: %s", X_train_processed.shape)
    logger.info("Processed testing shape: %s", X_test_processed.shape)

    logger.info("Feature engineering completed successfully")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )


if __name__ == "__main__":
    process_features()
