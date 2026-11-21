"""Trening modelu predykcji churnu."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(
    train_path: str,
    model_path: str,
    params_path: str = "params.yaml",
) -> dict:
    """Trenuje model i zapisuje artefakty."""

    # Wczytaj parametry
    with open(params_path) as f:
        params = yaml.safe_load(f)

    model_params = params.get("model", {})
    feature_cols = params.get(
        "features",
        [
            "age",
            "income",
            "tenure_months",
            "num_products",
            "has_credit_card",
            "is_active",
            "balance",
            "num_transactions_30d",
            "income_per_age",
            "balance_per_product",
            "is_long_tenure",
        ],
    )

    # Wczytaj dane
    train_df = pd.read_parquet(train_path)
    X_train = train_df[feature_cols]
    y_train = train_df["churn"]

    print(f"Trening na {len(X_train):,} próbkach, {len(feature_cols)} cechach")

    # Buduj pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=model_params.get("n_estimators", 100),
                    max_depth=model_params.get("max_depth", 8),
                    min_samples_split=model_params.get("min_samples_split", 10),
                    random_state=model_params.get("random_state", 42),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Ewaluacja na zbiorze treningowym
    y_pred = pipeline.predict(X_train)
    y_prob = pipeline.predict_proba(X_train)[:, 1]

    metrics = {
        "train_auc": float(roc_auc_score(y_train, y_prob)),
        "train_f1": float(f1_score(y_train, y_pred)),
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "n_estimators": model_params.get("n_estimators", 100),
        "max_depth": model_params.get("max_depth", 8),
    }

    print(f"  Train AUC: {metrics['train_auc']:.4f}")

    # Zapis modelu
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"  Model zapisany: {model_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/train.parquet")
    parser.add_argument("--model-output", default="models/churn_model.pkl")
    args = parser.parse_args()

    metrics = train_model(args.train_data, args.model_output)

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
