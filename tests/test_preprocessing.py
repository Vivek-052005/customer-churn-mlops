
import pandas as pd

from src.data import preprocess


def test_preprocessing_creates_train_and_test_files(
    tmp_path, monkeypatch
):
    # Create a temporary raw dataset
    data = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(20)],
        "gender": ["Female", "Male"] * 10,
        "tenure": list(range(20)),
        "TotalCharges": [
            "100", "200", "", "400", "500",
            "600", "700", "800", "900", "1000",
            "1100", "1200", "1300", "1400", "1500",
            "1600", "1700", "1800", "1900", "2000"
        ],
        "Churn": ["Yes", "No"] * 10
    })

    # Save temporary raw dataset
    raw_file = tmp_path / "customer_churn.csv"
    data.to_csv(raw_file, index=False)

    # Create temporary output paths
    processed_dir = tmp_path / "processed"

    train_file = processed_dir / "train.csv"
    test_file = processed_dir / "test.csv"

    # Replace actual project paths with temporary paths
    monkeypatch.setattr(
        preprocess, "RAW_DATA_PATH", raw_file
    )

    monkeypatch.setattr(
        preprocess, "PROCESSED_DIR", processed_dir
    )

    monkeypatch.setattr(
        preprocess, "TRAIN_PATH", train_file
    )

    monkeypatch.setattr(
        preprocess, "TEST_PATH", test_file
    )

    # Execute preprocessing
    preprocess.preprocess_data()

    # Check output files
    assert train_file.exists()
    assert test_file.exists()

    # Load generated datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Check target column
    assert "Churn" in train_df.columns
    assert "Churn" in test_df.columns

    # Check customerID was removed
    assert "customerID" not in train_df.columns
    assert "customerID" not in test_df.columns

    # Check target encoding
    assert set(train_df["Churn"].unique()).issubset({0, 1})
    assert set(test_df["Churn"].unique()).issubset({0, 1})

    # Check train/test split
    assert len(train_df) + len(test_df) == 20

    # Check missing TotalCharges remains available
    # for later pipeline imputation
    assert (
        train_df["TotalCharges"].isna().sum()
        + test_df["TotalCharges"].isna().sum()
    ) == 1


def test_missing_dataset_raises_error(
    tmp_path, monkeypatch
):
    missing_file = tmp_path / "missing.csv"

    monkeypatch.setattr(
        preprocess, "RAW_DATA_PATH", missing_file
    )

    import pytest

    with pytest.raises(FileNotFoundError):
        preprocess.preprocess_data()
