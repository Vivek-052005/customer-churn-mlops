
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# Create required directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def preprocess_data():
    """Clean the raw dataset and create train/test datasets."""

    logger.info("Starting data preprocessing")

    # 1. Check whether the raw dataset exists
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    # 2. Load the raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Raw dataset loaded successfully")
    logger.info("Original dataset shape: %s", df.shape)

    # 3. Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    logger.info("Duplicate rows removed: %s", duplicate_count)

    # 4. Remove customerID because it is an identifier,
    # not a useful predictive feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 5. Convert TotalCharges into numeric values
    # Blank strings become NaN and will be handled later
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # 6. Clean the target column
    df["Churn"] = df["Churn"].astype(str).str.strip()

    # Convert Yes/No into 1/0
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # 7. Validate target values
    if df["Churn"].isna().any():
        raise ValueError(
            "Invalid or missing values found in Churn column"
        )

    df["Churn"] = df["Churn"].astype(int)

    logger.info("Target column encoded successfully")

    # 8. Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 9. Split data before fitting any preprocessing steps
    # Stratification preserves the Yes/No churn proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 10. Combine features and target for saving
    train_data = X_train.copy()
    train_data["Churn"] = y_train

    test_data = X_test.copy()
    test_data["Churn"] = y_test

    # 11. Save the datasets
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    logger.info("Training dataset saved: %s", TRAIN_PATH)
    logger.info("Testing dataset saved: %s", TEST_PATH)

    logger.info("Training dataset shape: %s", train_data.shape)
    logger.info("Testing dataset shape: %s", test_data.shape)

    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    preprocess_data()
