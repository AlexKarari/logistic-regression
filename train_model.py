"""
Train Model: Logistic Regression (Telco Customer Churn)
===============================================================

End-to-end pipeline: load real Telco churn data, clean, encode,
train Logistic Regression from scratch, validate against sklearn,
and save model artifact for FastAPI deployment.

Dataset
-------
    Kaggle — Telco Customer Churn
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    Place `Telco-Customer-Churn.csv` in the `data/` folder.

Usage
-----
    python train_model.py

Output
------
    - models/churn_model.pkl  (model + scaler + metadata)
    - Console metrics report
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logistic_regression import LogisticRegressionScratch


# ── Configuration ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
N_ITERATIONS = 10000
LAMBDA_REG = 0.01
THRESHOLD = 0.5
DATA_PATH = "data/Telco-Customer-Churn.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the Telco dataset and perform cleaning.

    Steps:
        1. Drop customerID (not a feature)
        2. Fix TotalCharges (empty strings -> 0.0 for new customers)
        3. Encode target (Yes->1, No->0)

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with string target converted to int.
    """
    df = pd.read_csv(path)
    df = df.drop("customerID", axis=1)

    # TotalCharges has empty strings for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode categorical features for logistic regression.

    - Binary columns (Yes/No, Male/Female) -> label encoded (0/1)
    - Multi-category columns -> one-hot encoded (drop_first=True)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    tuple
        (encoded_dataframe, feature_column_names, new_one_hot_column_names)
    """
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]

    # Binary encoding
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)

    feature_cols = [c for c in df_encoded.columns if c != "Churn"]
    new_cols = [c for c in df_encoded.columns if c not in df.columns]

    return df_encoded, feature_cols, new_cols


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute and print classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n  {label}:")
    for name, value in metrics.items():
        print(f"    {name:<12}: {value:.4f}")
    return metrics


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("Day 3: Logistic Regression — Telco Customer Churn")
    print("=" * 60)

    # ── Load & clean ─────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n✗ Dataset not found at: {DATA_PATH}")
        print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("  Telco-Customer-Churn.csv in the data/ folder.")
        return

    df = load_and_clean(DATA_PATH)
    print(f"\nLoaded: {len(df):,} customers")
    print(f"Churn rate: {df['Churn'].mean():.1%}")

    # ── Encode ───────────────────────────────────────────────────────────
    df_encoded, feature_cols, new_cols = encode_features(df)
    print(f"Features after encoding: {len(feature_cols)}")

    X = df_encoded[feature_cols].values
    y = df_encoded["Churn"].values

    # ── Split & scale ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Train scratch model ──────────────────────────────────────────────
    print(f"\n--- Training LogisticRegressionScratch ---")
    print(f"  lr={LEARNING_RATE}, iterations={N_ITERATIONS}, λ={LAMBDA_REG}")

    model = LogisticRegressionScratch(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        lambda_reg=LAMBDA_REG,
        threshold=THRESHOLD,
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_metrics = evaluate_model(y_train, y_train_pred, "Train")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

    # ── Validate against sklearn ─────────────────────────────────────────
    print(f"\n--- Validating against sklearn ---")
    sklearn_model = SklearnLR(
        C=1 / LAMBDA_REG, max_iter=N_ITERATIONS, random_state=RANDOM_SEED
    )
    sklearn_model.fit(X_train_scaled, y_train)
    y_test_pred_sk = sklearn_model.predict(X_test_scaled)
    sk_metrics = evaluate_model(y_test, y_test_pred_sk, "sklearn Test")

    print(f"\n  Scratch F1: {test_metrics['f1']:.4f}  |  sklearn F1: {sk_metrics['f1']:.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    print(f"\n--- Top 10 Features (|weight|, scaled) ---")
    importance = sorted(
        zip(feature_cols, model.weights), key=lambda x: abs(x[1]), reverse=True
    )
    for feat, w in importance[:10]:
        direction = "↑ churn" if w > 0 else "↓ churn"
        print(f"    {feat:>40s}: {w:+.4f}  ({direction})")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "threshold": THRESHOLD,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "encoding_info": {
            "binary_cols": ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"],
            "multi_cat_cols": [
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaymentMethod",
            ],
            "numeric_cols": ["tenure", "MonthlyCharges", "TotalCharges"],
            "one_hot_cols": new_cols,
        },
    }
    artifact_path = "models/churn_model.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n✓ Model saved to {artifact_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()