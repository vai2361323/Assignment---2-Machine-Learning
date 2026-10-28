"""Streamlit interface for evaluating the saved assignment classifiers."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import CLASS_LABELS, calculate_metrics

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Classifier Evaluation Lab", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classifier Evaluation")
st.caption(
    "Five models trained on the UCI Wisconsin Diagnostic dataset. "
    "Upload labeled holdout data to compare their predictions."
)


@st.cache_resource
def load_saved_models():
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Model manifest is missing. Run `python train_models.py` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in manifest["models"].items()
    }
    return manifest, models


try:
    manifest, models = load_saved_models()
except (FileNotFoundError, OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the saved models: {error}")
    st.stop()

sample_path = BASE_DIR / "test_data.csv"
if sample_path.is_file():
    st.download_button(
        "Download the sample holdout CSV",
        data=sample_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
st.info(
    "The CSV must include all 30 model input columns and a `diagnosis` column "
    "containing `malignant` or `benign`. Models are loaded from disk; this app does not retrain."
)

if uploaded_file is None:
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

feature_columns = manifest["feature_columns"]
required_columns = feature_columns + [manifest["target"]]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error("CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()
if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()
if data[required_columns].isna().any().any():
    st.error("Required columns contain missing values. Clean the CSV and upload it again.")
    st.stop()

labels = data[manifest["target"]].astype(str)
unexpected_labels = sorted(set(labels) - set(CLASS_LABELS))
if unexpected_labels:
    st.error(
        "The diagnosis column has unsupported labels: "
        + ", ".join(unexpected_labels)
        + ". Expected `malignant` and `benign`."
    )
    st.stop()
if labels.nunique() != 2:
    st.error("AUC requires both diagnosis classes to be present in the uploaded test data.")
    st.stop()

try:
    numeric_features = data[feature_columns].apply(pd.to_numeric, errors="raise")
except (TypeError, ValueError) as error:
    st.error(f"Every feature must be numeric: {error}")
    st.stop()

if not np.isfinite(numeric_features.to_numpy()).all():
    st.error("Feature values must be finite numbers.")
    st.stop()

selected_model_name = st.selectbox("Model for detailed diagnostics", list(models))
all_metrics = {}
all_predictions = {}
try:
    for name, model in models.items():
        all_metrics[name] = calculate_metrics(model, numeric_features, labels)
        all_predictions[name] = model.predict(numeric_features)
except (TypeError, ValueError) as error:
    st.error(f"Could not evaluate the uploaded data: {error}")
    st.stop()

metrics_frame = pd.DataFrame.from_dict(all_metrics, orient="index")
metrics_frame.index.name = "Model"
st.subheader("Model comparison on uploaded data")
st.dataframe(metrics_frame.style.format("{:.4f}"))
st.bar_chart(metrics_frame[["Accuracy", "AUC", "F1", "MCC"]])

st.subheader(f"Diagnostics: {selected_model_name}")
predictions = all_predictions[selected_model_name]
matrix = confusion_matrix(labels, predictions, labels=list(CLASS_LABELS))
st.write("Rows are actual labels; columns are predicted labels.")
st.dataframe(
    pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in CLASS_LABELS],
        columns=[f"Predicted {label}" for label in CLASS_LABELS],
    )
)
st.text(
    classification_report(
        labels,
        predictions,
        labels=list(CLASS_LABELS),
        zero_division=0,
        digits=4,
    )
)
