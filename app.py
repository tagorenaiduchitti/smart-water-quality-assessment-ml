"""Streamlit dashboard for the expanded 20-feature water-quality benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "waterQuality1.csv",
    ROOT / "data" / "water_quality.csv",
    ROOT / "data" / "water-quality.csv",
]
MODEL_PATH = ROOT / "models" / "water_quality_models.joblib"
METRICS_PATH = ROOT / "results" / "model_comparison.csv"
SUMMARY_PATH = ROOT / "results" / "run_summary.json"
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="Smart Water Quality Assessment",
    page_icon="💧",
    layout="wide",
)
st.title("💧 Smart Water Quality Assessment")
st.caption(
    "Machine-learning demonstration using an expanded 20-parameter water-quality benchmark dataset"
)
st.warning(
    "This application predicts the source dataset's is_safe label (0 = not safe, 1 = safe). "
    "The CSV does not provide independently verifiable sample locations, collection dates, or laboratory records. "
    "A model prediction is not a drinking-water safety certification; use accredited laboratory testing "
    "to determine whether water is safe to consume."
)

# Locate project files.
data_path = next((path for path in DATA_CANDIDATES if path.is_file()), None)
if data_path is None:
    st.error("Dataset not found. Place waterQuality1.csv inside the project's data/ folder.")
    st.stop()
if not MODEL_PATH.is_file():
    st.error("Trained model bundle not found. Run train.py before launching the dashboard.")
    st.code(r".\.venv\Scripts\python.exe train.py", language="powershell")
    st.stop()
if not METRICS_PATH.is_file():
    st.error("Model comparison file not found. Run train.py to generate results/model_comparison.csv.")
    st.stop()

# Load and validate the model bundle.
try:
    bundle = joblib.load(MODEL_PATH)
    FEATURES = list(bundle["features"])
    TARGET = str(bundle["target"])
    models = bundle["models"]
    best_model_name = str(bundle["best_model_name"])
except (OSError, KeyError, ValueError, TypeError) as exc:
    st.error(f"Could not load the trained model bundle: {exc}")
    st.stop()

if not FEATURES or TARGET == "" or not models:
    st.error("The model bundle is incomplete. Please rerun train.py.")
    st.stop()
if best_model_name not in models:
    st.error("The selected model named in the bundle is missing from its model collection.")
    st.stop()

# Apply the same target validation, numeric conversion, feature selection and
# full-row duplicate removal used by the training pipeline.
try:
    raw_df = pd.read_csv(data_path)
    raw_rows = len(raw_df)
    raw_df.columns = [str(column).strip().lower() for column in raw_df.columns]
    if "fluoride" in raw_df.columns and "flouride" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"fluoride": "flouride"})

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        st.error(f"Dataset is missing columns required by the trained model: {missing_columns}")
        st.stop()

    df = raw_df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column].replace({"#NUM!": pd.NA, "": pd.NA}), errors="coerce"
        )

    invalid_target_rows = int((~df[TARGET].isin([0, 1])).sum())
    df = df.loc[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Could not read or prepare the dataset: {exc}")
    st.stop()

if df.empty or df[TARGET].nunique() != 2:
    st.error("The cleaned dataset must contain both target classes (0 and 1).")
    st.stop()

try:
    metrics = pd.read_csv(METRICS_PATH)
except (OSError, pd.errors.ParserError) as exc:
    st.error(f"Could not read model-comparison results: {exc}")
    st.stop()

summary = {}
if SUMMARY_PATH.is_file():
    try:
        with SUMMARY_PATH.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, json.JSONDecodeError):
        summary = {}

# Detect an old result file or a dataset/model mismatch instead of silently
# presenting metrics that may belong to a different training run.
summary_rows = summary.get("rows_after_cleaning")
if summary_rows is not None and int(summary_rows) != len(df):
    st.error(
        "Dataset/result mismatch: the current CSV has "
        f"{len(df):,} cleaned rows, but run_summary.json records {int(summary_rows):,}. "
        "Rerun train.py before using this dashboard."
    )
    st.stop()
if "Model" not in metrics.columns:
    st.error("model_comparison.csv does not contain the expected Model column. Rerun train.py.")
    st.stop()

# Header metrics.
class_counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
safe_rate = class_counts.loc[1] / len(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid dataset samples", f"{len(df):,}")
c2.metric("Input parameters", str(len(FEATURES)))
c3.metric("ML models compared", str(len(models)))
c4.metric("Selected model", best_model_name)

st.caption(
    f"Dataset file: {data_path.name} · Raw rows: {raw_rows:,} · "
    f"Invalid target rows removed: {invalid_target_rows:,} · "
    f"Duplicate rows removed: {duplicate_rows:,}"
)

tab_overview, tab_explore, tab_predict, tab_models = st.tabs(
    ["Overview", "Explore Data", "Test a Sample", "Model Results"]
)

with tab_overview:
    st.subheader("Project overview")
    st.write(
        f"The project analyses {len(FEATURES)} chemical and biological water-quality indicators "
        f"and predicts the dataset's binary {TARGET} label (0 = not safe, 1 = safe). "
        "Missing numerical values are imputed inside each model pipeline. "
        "The data is divided into stratified training and held-out test sets before model fitting."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Source-label distribution**")
        distribution = pd.DataFrame(
            {"Samples": class_counts.values},
            index=["Not safe (0)", "Safe (1)"],
        )
        st.bar_chart(distribution)
        st.caption(
            f"Safe-label share: {safe_rate:.1%}. The target classes are imbalanced; "
            "accuracy should be interpreted alongside balanced accuracy, macro F1 and class recall."
        )
    with right:
        st.markdown("**Dataset preview**")
        st.dataframe(df.head(8), use_container_width=True)

    with st.expander("Data preparation summary"):
        st.write(
            {
                "Raw rows": raw_rows,
                "Invalid target rows removed": invalid_target_rows,
                "Duplicate rows removed": duplicate_rows,
                "Final valid rows": len(df),
                "Predictor columns": len(FEATURES),
                "Missing predictor cells": int(df[FEATURES].isna().sum().sum()),
                "Negative values by predictor": {
                    feature: int((df[feature] < 0).sum())
                    for feature in FEATURES
                    if int((df[feature] < 0).sum()) > 0
                },
                "Not-safe labels (0)": int(class_counts.loc[0]),
                "Safe labels (1)": int(class_counts.loc[1]),
            }
        )

with tab_explore:
    st.subheader("Exploratory data analysis")
    missing_counts = df[FEATURES].isna().sum().sort_values(ascending=False)
    negative_counts = (df[FEATURES] < 0).sum().sort_values(ascending=False)
    st.markdown("**Missing values by parameter (before model imputation)**")
    if int(missing_counts.sum()) == 0:
        st.success("No missing predictor values remain in the cleaned dataset.")
    else:
        st.bar_chart(missing_counts[missing_counts > 0])

    if int(negative_counts.sum()) > 0:
        nonzero_negative_counts = negative_counts[negative_counts > 0]
        st.warning(
            "Potentially unusual negative measurements were found in these predictors: "
            + ", ".join(
                f"{feature} ({int(count)})" for feature, count in nonzero_negative_counts.items()
            )
            + ". They are shown for transparency and have not been silently removed. "
            "Check the source dataset's units and data-generation notes before interpreting them."
        )

    st.markdown("**Descriptive statistics**")
    st.dataframe(df[FEATURES].describe().T.round(4), use_container_width=True)

    st.markdown("**Correlation matrix**")
    correlation = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(correlation, aspect="auto", vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(correlation.index)), correlation.index, fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_predict:
    st.subheader("Predict the benchmark dataset label")
    st.write(
        "Enter the 20 measured parameters. Initial values are the dataset medians for demonstration. "
        "Use the units and ranges documented by the dataset provider when interpreting inputs."
    )
    medians = df[FEATURES].median(numeric_only=True)
    with st.form("prediction_form"):
        values = {}
        columns = st.columns(4)
        for index, feature in enumerate(FEATURES):
            series = pd.to_numeric(df[feature], errors="coerce").dropna()
            if series.empty:
                low, high, default = 0.0, 100.0, 50.0
            else:
                low = float(min(0.0, series.min()))
                high = float(series.max() * 1.25) if series.max() > 0 else float(series.max() + 1.0)
                if high <= low:
                    high = low + 1.0
                default_value = medians.get(feature)
                default = float(default_value) if pd.notna(default_value) else (low + high) / 2
                default = min(max(default, low), high)
            with columns[index % 4]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=low,
                    max_value=high,
                    value=default,
                    step=max((high - low) / 100, 0.001),
                    format="%.4f",
                )
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(best_model_name),
        )
        submitted = st.form_submit_button("Assess sample")

    if submitted:
        sample = pd.DataFrame([values], columns=FEATURES)
        model = models[selected_model]
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)
        safe_probability = float(probabilities[classes.index(1)])

        left, right = st.columns(2)
        with left:
            if prediction == 1:
                st.success("Model output: Safe source-dataset label (1)")
            else:
                st.warning("Model output: Not-safe source-dataset label (0)")
        with right:
            st.metric("Model probability for source label 1", f"{safe_probability:.1%}")
        st.dataframe(sample, use_container_width=True)
        st.warning(
            "This is a prediction of the benchmark dataset label, not a real-world water-safety verdict. "
            "It must not be used as the basis for drinking-water decisions."
        )

with tab_models:
    st.subheader("Held-out test-set performance")
    if metrics.empty:
        st.warning("Run train.py to generate model metrics.")
    else:
        display = metrics.copy()
        numeric_columns = display.select_dtypes(include="number").columns
        display[numeric_columns] = display[numeric_columns].round(3)
        st.dataframe(display, use_container_width=True)

        selection_note = (
            "The table reports held-out test-set metrics. The CV_F1_macro_mean and "
            "CV_F1_macro_std columns, when present, summarize five-fold cross-validation "
            "on the training split. These CV figures are separate from held-out test metrics."
            if "CV_F1_macro_mean" in metrics.columns
            else "The table reports held-out test-set metrics from the latest training run."
        )
        st.caption(selection_note)

        st.markdown("**Macro F1-score comparison**")
        chart_data = display[display["Model"] != "Majority Baseline (not ML)"]
        if not chart_data.empty and "F1_macro" in chart_data.columns:
            st.bar_chart(chart_data.set_index("Model")[["F1_macro"]])

        st.markdown("**Confusion matrices**")
        names = ["Logistic Regression", "Decision Tree", "Random Forest"]
        matrix_columns = st.columns(len(names))
        for index, name in enumerate(names):
            image_path = RESULTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
            with matrix_columns[index]:
                st.markdown(f"**{name}**")
                if image_path.is_file():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.info("Confusion matrix image not found. Rerun train.py.")

        st.caption(
            "The majority-class baseline illustrates why raw accuracy alone is insufficient for this imbalanced dataset. "
            "All reported metrics describe prediction of the dataset's labels, not verified drinking-water safety."
        )
