"""Preprocess data, run EDA, train three classifiers, and save results."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from download_data import download_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET = "Potability"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
RANDOM_STATE = 42


def load_and_check_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        download_dataset()
    df = pd.read_csv(DATA_PATH)
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")
    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates().reset_index(drop=True)
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Potability target must contain only 0 and 1.")
    df[TARGET] = df[TARGET].astype(int)
    if df.empty:
        raise ValueError("The dataset is empty after validation.")
    return df


def save_eda(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Target distribution
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Not potable (0)", "Potable (1)"], [counts.get(0, 0), counts.get(1, 0)])
    ax.set_title("Distribution of Dataset Potability Labels")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Missing data before imputation
    missing = df[FEATURES].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(missing.index, missing.values)
    ax.set_title("Missing Values by Feature (Before Imputation)")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    # Correlation heatmap
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    df[FEATURES + [TARGET]].describe().T.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    missing.rename("missing_count").to_csv(RESULTS_DIR / "missing_value_summary.csv")


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_check_data()
    save_eda(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    trained_models = {}
    metric_rows = []
    reports = {}
    matrices = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "Recall_non_potable_class_0": recall_score(y_test, predictions, pos_label=0, zero_division=0),
        }
        metric_rows.append(metrics)
        reports[name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        matrices[name] = confusion_matrix(y_test, predictions).tolist()
        trained_models[name] = model

        # Save one confusion matrix for every classifier
        fig, ax = plt.subplots(figsize=(4.5, 4))
        matrix = confusion_matrix(y_test, predictions)
        image = ax.imshow(matrix)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        fig.savefig(RESULTS_DIR / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows).sort_values("F1_macro", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_model_name = str(metrics_df.iloc[0]["Model"])
    joblib.dump({
        "models": trained_models,
        "best_model_name": best_model_name,
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
    }, MODELS_DIR / "water_quality_models.joblib")
    with open(RESULTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(RESULTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f, indent=2)
    metadata = {
        "rows_after_basic_checks": int(len(df)),
        "feature_count": len(FEATURES),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_values_by_feature": {k: int(v) for k, v in df[FEATURES].isna().sum().items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model_by_macro_f1": best_model_name,
        "split": "80% train / 20% test, stratified",
        "random_state": RANDOM_STATE,
    }
    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDATA SUMMARY")
    print(json.dumps(metadata, indent=2))
    print("\nMODEL COMPARISON")
    print(metrics_df.to_string(index=False, formatters={c: (lambda x: f"{x:.3f}") for c in metrics_df.columns if c != "Model"}))
    print(f"\nBest model by macro F1: {best_model_name}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Model bundle saved in: {MODELS_DIR / 'water_quality_models.joblib'}")


if __name__ == "__main__":
    main()
