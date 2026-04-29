"""
Diabetes Risk Prediction Project

This script trains a Random Forest classifier to predict diabetes risk using
diagnostic health measurements.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"


def load_data(path: Path) -> pd.DataFrame:
    """Load the diabetes dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Make sure diabetes.csv is inside the data folder."
        )
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> None:
    """Train, evaluate, and save a Random Forest model."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("DIABETES RISK PREDICTION MODEL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
    plt.title("Confusion Matrix - Diabetes Prediction")
    plt.tight_layout()
    plt.show()

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def main() -> None:
    df = load_data(DATA_PATH)

    print("Dataset preview:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())

    train_model(df)


if __name__ == "__main__":
    main()
