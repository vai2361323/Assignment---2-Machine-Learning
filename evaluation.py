"""Shared evaluation utilities for the assignment models."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = "malignant"
CLASS_LABELS = ("malignant", "benign")


def calculate_metrics(model, features, actual):
    """Return the required binary-classification metrics."""
    predicted = model.predict(features)
    class_names = list(model.classes_)
    if POSITIVE_LABEL not in class_names:
        raise ValueError(f"Model does not contain the positive class {POSITIVE_LABEL!r}.")

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, class_names.index(POSITIVE_LABEL)]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        scores = raw_scores if class_names[1] == POSITIVE_LABEL else -raw_scores
    else:
        raise TypeError("Model must provide predict_proba or decision_function for AUC.")

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(
            actual,
            scores,
            labels=class_names,
        ),
        "Precision": precision_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }
