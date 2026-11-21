
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET = "is_safe"
FEATURES = [
    "aluminium", "ammonia", "arsenic", "barium", "cadmium",
    "chloramine", "chromium", "copper", "flouride", "bacteria",
    "viruses", "lead", "nitrates", "nitrites", "mercury",
    "perchlorate", "radium", "selenium", "silver", "uranium",
]
RANDOM_STATE = 42


def locate_csv() -> Path:
    candidates = [
        DATA_DIR / "waterQuality1.csv",
        DATA_DIR / "water_quality.csv",
        DATA_DIR / "water-quality.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1_000:
            return path
    raise FileNotFoundError(
        "The larger Kaggle dataset was not found. Download waterQuality1.csv from "
        "https://www.kaggle.com/datasets/mssmartypants/water-quality and place it in "
        f"{DATA_DIR}"
    )


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    path = locate_csv()
    df = pd.read_csv(path)
    # Normalize column spelling/capitalization while preserving the source file.
    df.columns = [str(c).strip().lower() for c in df.columns]
    # The Kaggle file is commonly spelled 'flouride'; accept the standard spelling too.
    if "fluoride" in df.columns and "flouride" not in df.columns:
        df = df.rename(columns={"fluoride": "flouride"})
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df = df[required].copy()
    original_rows = len(df)
    for col in required:
        df[col] = pd.to_numeric(df[col].replace({"#NUM!": np.nan, "": np.nan}), errors="coerce")

    # Remove rows without a valid binary target. Do not infer/recreate labels.
    valid_target = df[TARGET].isin([0, 1])
    invalid_target_rows = int((~valid_target).sum())
    df = df.loc[valid_target].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Audit repeated feature vectors separately from full-row duplicates.
    # Conflicting labels for identical measurements can inflate random-split scores.
    duplicate_feature_rows = int(df.duplicated(subset=FEATURES).sum())
    predictor_label_groups = df.groupby(FEATURES, dropna=False)[TARGET].nunique()
    conflicting_predictor_groups = int((predictor_label_groups > 1).sum())

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows remain after validating the is_safe target.")
    if df[TARGET].nunique() != 2:
        raise ValueError("The cleaned is_safe target must contain both 0 and 1 classes.")

    metadata = {
        "source_file": path.name,
        "original_rows": int(original_rows),
        "invalid_target_rows_removed": invalid_target_rows,
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_feature_rows": duplicate_feature_rows,
        "conflicting_predictor_label_groups": conflicting_predictor_groups,
        "rows_after_cleaning": int(len(df)),
        "feature_count": len(FEATURES),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "missing_values_by_feature": {k: int(v) for k, v in df[FEATURES].isna().sum().items()},
    }
    return df, metadata

#Data Cleaning
def save_eda(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Not safe (0)", "Safe (1)"], counts.values)
    ax.set_title("Distribution of Water Safety Labels")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    missing = df[FEATURES].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(missing.index, missing.values)
    ax.set_title("Missing Values by Feature (Before Imputation)")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 11))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    df[FEATURES].describe().T.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    missing.rename("missing_count").to_csv(RESULTS_DIR / "missing_value_summary.csv")

#MODELS USED
def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=5, class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata = load_and_validate()
    save_eda(df)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Verify that no identical predictor vectors cross the train/test boundary.
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False).tolist())
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).tolist())
    metadata["identical_feature_vectors_shared_between_train_and_test"] = int(
        len(train_hashes.intersection(test_hashes))
    )

    # A majority-class baseline shows why raw accuracy alone is misleading here.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    rows = [{
        "Model": "Majority Baseline (not ML)",
        "Accuracy": accuracy_score(y_test, baseline_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, baseline_pred),
        "Precision_macro": precision_score(y_test, baseline_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_test, baseline_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, baseline_pred, average="macro", zero_division=0),
        "ROC_AUC": np.nan,
        "Safe_class_recall": recall_score(y_test, baseline_pred, pos_label=1, zero_division=0),
        "Safe_class_average_precision": np.nan,
        "CV_F1_macro_mean": np.nan,
        "CV_F1_macro_std": np.nan,
    }]

    trained_models = {}
    reports = {}
    matrices = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = "f1_macro"

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, predictions),
            "Precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "Safe_class_recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
            "Safe_class_average_precision": average_precision_score(y_test, probabilities),
            "CV_F1_macro_mean": float(np.mean(cv_scores)),
            "CV_F1_macro_std": float(np.std(cv_scores)),
        })
        reports[name] = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        matrices[name] = matrix.tolist()

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Not safe (0)", "Safe (1)"]
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)
        trained_models[name] = model

    metrics_df = pd.DataFrame(rows)
    # Select using cross-validation on the training split only.
    # The held-out test set is reserved for final reporting and must not choose the model.
    ml_metrics = metrics_df[metrics_df["Model"] != "Majority Baseline (not ML)"]
    best_model_name = str(
        ml_metrics.sort_values("CV_F1_macro_mean", ascending=False).iloc[0]["Model"]
    )
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

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

    metadata.update({
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "80% train / 20% test, stratified",
        "random_state": RANDOM_STATE,
        "majority_class_test_accuracy_baseline": float(accuracy_score(y_test, baseline_pred)),
        "best_model_by_macro_f1": best_model_name,
        "note": (
            "This is a public Kaggle benchmark dataset. The CSV does not provide "
            "verifiable sample locations, collection dates, or laboratory provenance. "
            "Metrics describe prediction of the dataset's is_safe labels only and do "
            "not establish real-world drinking-water safety."
        ),
    })
    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDATA SUMMARY")
    print(json.dumps(metadata, indent=2))
    print("\nMODEL COMPARISON")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nBest model by macro F1: {best_model_name}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Model bundle saved in: {MODELS_DIR / 'water_quality_models.joblib'}")


if __name__ == "__main__":
    main()
