
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessor"
    / "preprocessor.joblib"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. CONFIGURE MLFLOW
# --------------------------------------------------

# Store MLflow experiment information inside the project
# Use SQLite as the MLflow tracking backend
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_experiment("Customer_Churn_Prediction")


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

def load_data():

    logger.info("Loading training and testing datasets")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError("Training dataset not found")

    if not TEST_PATH.exists():
        raise FileNotFoundError("Testing dataset not found")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    logger.info("Training data shape: %s", X_train.shape)
    logger.info("Testing data shape: %s", X_test.shape)

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 5. LOAD PREPROCESSOR
# --------------------------------------------------

def load_preprocessor():

    logger.info("Loading saved preprocessor")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Preprocessor not found")

    return joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# 6. DEFINE MODELS
# --------------------------------------------------

def get_models():

    return {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

    }


# --------------------------------------------------
# 7. TRAIN AND TRACK MODELS
# --------------------------------------------------

def train_models():

    logger.info("Starting MLflow model training")

    X_train, X_test, y_train, y_test = load_data()

    preprocessor = load_preprocessor()

    # Transform training and testing data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = get_models()

    for model_name, model in models.items():

        logger.info("Training model: %s", model_name)

        # Every model gets a separate MLflow run
        with mlflow.start_run(run_name=model_name):

            # Record model information
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("testing_rows", len(X_test))

            # Record model hyperparameters
            mlflow.log_params(model.get_params())

            # Train model
            model.fit(X_train_processed, y_train)

            # Predict test data
            y_pred = model.predict(X_test_processed)

            y_probability = model.predict_proba(
                X_test_processed
            )[:, 1]

            # Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),

                "precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),

                "recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),

                "f1_score": f1_score(
                    y_test, y_pred, zero_division=0
                ),

                "roc_auc": roc_auc_score(
                    y_test, y_probability
                )
            }

            # Record metrics in MLflow
            mlflow.log_metrics({
                name: float(value)
                for name, value in metrics.items()
            })

            # Save model in project artifacts
            model_path = MODEL_DIR / f"{model_name}.joblib"

            joblib.dump(model, model_path)

            model_log_kwargs = {}
            if model_name in ["decision_tree", "random_forest"]:
                model_log_kwargs["skops_trusted_types"] = [
                    "sklearn.tree._tree.Tree"
                    ]
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    **model_log_kwargs
                    )


            # Log the saved preprocessor
            mlflow.log_artifact(
                str(PREPROCESSOR_PATH),
                artifact_path="preprocessor"
            )

            logger.info(
                "%s | Accuracy: %.4f | F1-score: %.4f",
                model_name,
                metrics["accuracy"],
                metrics["f1_score"]
            )

            logger.info(
                "MLflow run ID: %s",
                mlflow.active_run().info.run_id
            )

    logger.info("All MLflow experiments completed successfully")


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    train_models()
