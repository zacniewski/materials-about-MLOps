"""Run one parameterized training experiment and record it in MLflow."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def run_experiment(
    n_estimators: int = 100,
    max_depth: int = 8,
    experiment_name: str = "churn-baseline",
    tracking_uri: str | None = None,
) -> str:
    """Log cross-validation metrics on training data and return the MLflow run ID."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    train = pd.read_parquet("data/processed/train.parquet")
    target = "churn"
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    features = params["features"]
    model_params = params["model"]
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=model_params["min_samples_split"],
        random_state=model_params["random_state"],
        n_jobs=-1,
    )
    with mlflow.start_run() as run:
        scores = cross_validate(
            model,
            train[features],
            train[target],
            cv=3,
            scoring=("accuracy", "f1", "roc_auc"),
        )
        metrics = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in ("accuracy", "f1", "roc_auc")
        }
        model.fit(train[features], train[target])
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": model_params["min_samples_split"],
                "random_state": model_params["random_state"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("data_version", "dvc-managed")
        mlflow.set_tag("evaluation_strategy", "3-fold CV on training split")
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id
