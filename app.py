"""Streamlit demo for the Water Quality Potability ML project."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "models" / "water_quality_models.joblib"
METRICS_PATH = ROOT / "results" / "model_comparison.csv"
MATRICES_PATH = ROOT / "results" / "confusion_matrices.json"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
LABELS = {
    "ph": "pH",
    "Hardness": "Hardness (mg/L as CaCO₃)",
    "Solids": "Total dissolved solids (ppm)",
    "Chloramines": "Chloramines (ppm)",
    "Sulfate": "Sulfate (mg/L)",
    "Conductivity": "Conductivity (µS/cm)",
    "Organic_carbon": "Organic carbon (ppm)",
    "Trihalomethanes": "Trihalomethanes (µg/L)",
    "Turbidity": "Turbidity (NTU)",
}

st.set_page_config(page_title="Smart Water Quality Assessment", page_icon="💧", layout="wide")
st.title("💧 Smart Water Quality Assessment")
st.caption("Machine-learning demonstration for exploratory water-quality assessment")
st.info(
    "This application predicts the label in an educational benchmark dataset. "
    "It is not a certified drinking-water test and must not be used to decide whether water is safe to consume. "
    "Only an accredited laboratory can confirm potability."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Project files are not ready yet. Run `python train.py` once, then restart the Streamlit app.")
    st.code("python train.py\nstreamlit run app.py", language="bash")
    st.stop()

bundle = joblib.load(MODEL_PATH)
models = bundle["models"]
best_model_name = bundle["best_model_name"]
best_model = models[best_model_name]
df = pd.read_csv(DATA_PATH)
metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()

# Header KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset samples", f"{len(df):,}")
c2.metric("Input parameters", str(len(FEATURES)))
c3.metric("Models compared", str(len(models)))
c4.metric("Selected model", best_model_name)

tab_overview, tab_explore, tab_predict, tab_models = st.tabs(
    ["Overview", "Explore Data", "Test a Sample", "Model Results"]
)

with tab_overview:
    st.subheader("Project overview")
    st.write(
        "The project uses nine measured water-quality parameters to learn the relationship "
        "with the dataset's binary Potability label (0 or 1). Missing numerical values are "
        "imputed inside each model pipeline, and the data is split into stratified training "
        "and test sets before model fitting."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Potability label distribution**")
        class_counts = df["Potability"].value_counts().reindex([0, 1], fill_value=0)
        class_chart = pd.DataFrame({"Samples": class_counts.values}, index=["Not potable (0)", "Potable (1)"])
        st.bar_chart(class_chart)
    with right:
        st.markdown("**Dataset preview**")
        st.dataframe(df.head(8), use_container_width=True)
    st.caption("The labels are used as supplied in the benchmark dataset; they are not independent laboratory certifications.")

with tab_explore:
    st.subheader("Exploratory data analysis")
    st.markdown("**Missing values by parameter**")
    missing = df[FEATURES].isna().sum().sort_values(ascending=False)
    st.bar_chart(missing)
    st.markdown("**Descriptive statistics**")
    st.dataframe(df[FEATURES].describe().T.round(3), use_container_width=True)
    st.markdown("**Correlation matrix**")
    corr = df[FEATURES + ["Potability"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_predict:
    st.subheader("Predict the benchmark potability label")
    st.write("Enter the nine parameters from a water sample. Default values are dataset medians for demonstration.")
    medians = df[FEATURES].median(numeric_only=True)
    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        bounds = {
            "ph": (0.0, 14.0), "Hardness": (0.0, 1000.0), "Solids": (0.0, 100000.0),
            "Chloramines": (0.0, 30.0), "Sulfate": (0.0, 2000.0), "Conductivity": (0.0, 3000.0),
            "Organic_carbon": (0.0, 100.0), "Trihalomethanes": (0.0, 500.0), "Turbidity": (0.0, 50.0),
        }
        for index, feature in enumerate(FEATURES):
            low, high = bounds[feature]
            with cols[index % 3]:
                values[feature] = st.number_input(
                    LABELS[feature], min_value=low, max_value=high,
                    value=float(min(max(medians[feature], low), high)), step=0.1, format="%.3f"
                )
        selected_model = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index(best_model_name))
        submitted = st.form_submit_button("Assess sample")

    if submitted:
        sample = pd.DataFrame([values], columns=FEATURES)
        model = models[selected_model]
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        positive_probability = float(probabilities[list(model.classes_).index(1)])
        a, b = st.columns(2)
        with a:
            if prediction == 1:
                st.success("Model output: Potable label (1)")
            else:
                st.warning("Model output: Not-potable label (0)")
        with b:
            st.metric("Model probability for label 1", f"{positive_probability:.1%}")
        st.write("**Submitted measurements**")
        st.dataframe(sample.rename(columns=LABELS), use_container_width=True)
        st.warning("This is a model prediction against the dataset's labels, not a real-world safety verdict.")

with tab_models:
    st.subheader("Held-out test-set performance")
    if metrics.empty:
        st.warning("Run `python train.py` to generate model metrics.")
    else:
        display = metrics.copy()
        numeric_cols = display.select_dtypes(include="number").columns
        display[numeric_cols] = display[numeric_cols].round(3)
        st.dataframe(display, use_container_width=True)
        st.markdown("**Macro F1-score comparison**")
        st.bar_chart(display.set_index("Model")[["F1_macro"]])
        st.markdown("**Why macro metrics?** They give equal weight to both classes, which is useful when the class counts are uneven.")
        st.caption("Metrics are generated by train.py using a stratified 80/20 split and random_state=42.")

        st.divider()
        st.markdown("### Confusion matrices")
        st.caption(
            "Rows show the actual dataset label; columns show the model's predicted label. "
            "Class 0 = Not potable; Class 1 = Potable."
        )
        if MATRICES_PATH.exists():
            with open(MATRICES_PATH, "r", encoding="utf-8") as matrix_file:
                confusion_matrices = json.load(matrix_file)

            # Follow the same model order as the performance table above.
            model_order = display["Model"].tolist()
            matrix_columns = st.columns(len(model_order))
            for column, model_name in zip(matrix_columns, model_order):
                matrix_values = confusion_matrices.get(model_name)
                with column:
                    st.markdown(f"**{model_name}**")
                    if matrix_values is None:
                        st.warning("No saved confusion matrix was found for this model.")
                        continue

                    fig, ax = plt.subplots(figsize=(4.2, 3.8))
                    image = ax.imshow(matrix_values, cmap="Blues")
                    ax.set_xticks([0, 1], ["0: Not potable", "1: Potable"], rotation=20, ha="right")
                    ax.set_yticks([0, 1], ["0: Not potable", "1: Potable"])
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("Actual label")
                    ax.set_title("Confusion matrix")

                    threshold = max(max(row) for row in matrix_values) / 2
                    for row_index, row in enumerate(matrix_values):
                        for col_index, count in enumerate(row):
                            text_color = "white" if count > threshold else "black"
                            ax.text(
                                col_index, row_index, str(count),
                                ha="center", va="center", color=text_color, fontweight="bold"
                            )
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
        else:
            st.warning(
                "Confusion-matrix data was not found. Run `python train.py` once to regenerate the evaluation results."
            )
