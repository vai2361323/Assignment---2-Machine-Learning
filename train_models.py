"""Train and save the five required classifiers on a reproducible UCI dataset split."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import calculate_metrics

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_models():
    """Create fresh model pipelines; scale distance- and margin-based models."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(kernel="rbf")),
            ]
        ),
    }


def load_dataset():
    """Load the Wisconsin Diagnostic dataset and use human-readable labels."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.map({0: "malignant", 1: "benign"}).rename("diagnosis")
    return features, target


def train_and_evaluate():
    """Fit each model on the same training rows and save its holdout results."""
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test_data = x_test.copy()
    test_data["diagnosis"] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    results = []
    model_files = {}
    for model_name, model in make_models().items():
        model.fit(x_train, y_train)
        metrics = calculate_metrics(model, x_test, y_test)
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)
        model_files[model_name] = filename
        results.append({"Model": model_name, **metrics})

    results_frame = pd.DataFrame(results).set_index("Model")
    results_frame.to_csv(BASE_DIR / "model_comparison.csv", float_format="%.6f")
    (MODEL_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "UCI Wisconsin Diagnostic Breast Cancer",
                "target": "diagnosis",
                "positive_label": "malignant",
                "feature_columns": list(features.columns),
                "models": model_files,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Dataset rows/features: {len(features)} / {features.shape[1]}")
    print(f"Train/test rows: {len(x_train)} / {len(x_test)}")
    print("\nHoldout metrics (positive class: malignant):")
    print(results_frame.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved test data to: {BASE_DIR / 'test_data.csv'}")


if __name__ == "__main__":
    train_and_evaluate()
