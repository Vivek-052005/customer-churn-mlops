
import logging
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessor"
    / "preprocessor.joblib"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

RESULT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = RESULT_DIR / "evaluation_results.json"


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

def load_test_data():

    logger.info("Loading test dataset")

    if not TEST_PATH.exists():
        raise FileNotFoundError("Test dataset not found")

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_test, y_test


# --------------------------------------------------
# 4. LOAD PREPROCESSOR
# --------------------------------------------------

def load_preprocessor():

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Preprocessor not found")

    return joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# 5. EVALUATE MODELS
# --------------------------------------------------

def evaluate_models():

    logger.info("Starting model evaluation")

    X_test, y_test = load_test_data()

    preprocessor = load_preprocessor()

    # Transform test data using the already fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    results = {}

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest"
    ]

    for model_name in model_names:

        model_path = MODEL_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        logger.info("Evaluating model: %s", model_name)

        model = joblib.load(model_path)

        # Predict customer churn
        y_pred = model.predict(X_test_processed)

        # Predict probabilities for ROC-AUC
        y_probability = model.predict_proba(
            X_test_processed
        )[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test, y_probability
        )

        # Save metrics
        results[model_name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }

        logger.info(
            "%s | Accuracy: %.4f | Precision: %.4f | "
            "Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
            model_name,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        )

        # Print detailed report
        print(f"\nClassification Report: {model_name}")

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # --------------------------------------------------
    # 6. SELECT BEST MODEL USING F1-SCORE
    # --------------------------------------------------

    best_model_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    results["best_model"] = best_model_name

    logger.info("Best model based on F1-score: %s", best_model_name)

    # --------------------------------------------------
    # 7. SAVE EVALUATION RESULTS
    # --------------------------------------------------

    with open(RESULT_PATH, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info("Evaluation results saved: %s", RESULT_PATH)

    print("\n========== MODEL COMPARISON ==========")

    comparison_df = pd.DataFrame(results).T

    print(comparison_df)

    print("\nBest model:", best_model_name)

    logger.info("Model evaluation completed successfully")


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    evaluate_models()
