"""Lecture 3, section 4 – Comparing experiments programmatically.

Requires running 01_mlflow_experiments.py first.
Fetches runs via the API, displays a ranking, loads the best model
and registers it in the Model Registry (with the "champion" alias).

Usage:
    python lectures-EN/scripts/lecture_03/02_compare_runs.py [--metric roc_auc]
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "churn-prediction-v1"
MODEL_NAME = "churn-classifier"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="roc_auc")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                              filter_string="attributes.status = 'FINISHED'",
                              order_by=[f"metrics.{args.metric} DESC"])
    if runs.empty:
        raise SystemExit("No runs – run 01_mlflow_experiments.py first")

    cols = ["tags.mlflow.runName", "tags.model_type", "metrics.roc_auc", "metrics.f1",
            "metrics.accuracy", "metrics.cv_roc_auc_mean"]
    print(f"📋 Run ranking by {args.metric}:\n")
    print(runs[cols].round(4).to_string(index=False))

    # Filtering example (same syntax as in the UI)
    rf = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                            filter_string="tags.model_type = 'RandomForestClassifier' and metrics.roc_auc > 0.9")
    print(f"\n🔎 RandomForest runs with roc_auc > 0.9: {len(rf)}")

    best = runs.iloc[0]
    run_id = best["run_id"]
    print(f"\n🏆 Best: {best['tags.mlflow.runName']} (run_id={run_id})")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    sample = np.random.default_rng(0).normal(size=(3, model.n_features_in_))
    print(f"🔮 Prediction with the loaded model: {model.predict(sample).tolist()}")

    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"📦 Registered {MODEL_NAME} v{mv.version} with alias 'champion'")
    print(f"   Loading in a service: mlflow.pyfunc.load_model('models:/{MODEL_NAME}@champion')")


if __name__ == "__main__":
    main()
