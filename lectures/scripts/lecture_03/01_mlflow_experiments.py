"""Wykład 3, sekcja 3 – Pełny eksperyment z MLflow.

Trenuje kilka modeli z różnymi hiperparametrami i loguje do MLflow:
parametry, metryki (w tym CV), tagi, wykres (artefakt), raport i model.

Domyślnie zapisuje runy lokalnie (katalog ./mlruns). Aby użyć serwera:
    mlflow ui --port 5000                      # w osobnym terminalu
    export MLFLOW_TRACKING_URI=http://localhost:5000

Uruchomienie:
    python lectures/scripts/lecture_03/01_mlflow_experiments.py
    mlflow ui                                   # → http://localhost:5000
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402

EXPERIMENT_NAME = "churn-prediction-v1"

X, y = make_classification(n_samples=5_000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(model) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba)}


def run_experiment(model_cls, params: dict, run_name: str) -> dict:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_type": model_cls.__name__, "dataset": "synthetic-v1",
                         "author": os.getenv("USER", "student")})
        mlflow.log_params(params)
        mlflow.log_params({"n_train": len(X_train), "n_features": X.shape[1]})

        model = model_cls(**params)
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
        mlflow.log_metrics({"cv_roc_auc_mean": cv.mean(), "cv_roc_auc_std": cv.std()})

        model.fit(X_train, y_train)
        metrics = evaluate(model)
        mlflow.log_metrics(metrics)

        # krzywa uczenia dla GBM – metryka logowana krok po kroku
        if hasattr(model, "staged_predict_proba"):
            for step, proba in enumerate(model.staged_predict_proba(X_test)):
                if step % 10 == 0:
                    mlflow.log_metric("staged_roc_auc", roc_auc_score(y_test, proba[:, 1]), step=step)

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
            fig.savefig(Path(tmp) / "confusion_matrix.png", bbox_inches="tight")
            plt.close(fig)
            (Path(tmp) / "classification_report.json").write_text(
                json.dumps(classification_report(y_test, model.predict(X_test), output_dict=True), indent=2))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        mlflow.sklearn.log_model(model, artifact_path="model",
                                 signature=infer_signature(X_test, model.predict(X_test)),
                                 input_example=X_test[:3])
        print(f"✅ {run_name:<28} roc_auc={metrics['roc_auc']:.4f}  f1={metrics['f1']:.4f}")
        return metrics


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}  |  eksperyment: {EXPERIMENT_NAME}\n")

    configs = [
        (LogisticRegression, {"C": 0.1, "max_iter": 1000}, "LR-C0.1"),
        (LogisticRegression, {"C": 1.0, "max_iter": 1000}, "LR-C1.0"),
        (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "random_state": 42}, "RF-100-d5"),
        (RandomForestClassifier, {"n_estimators": 200, "max_depth": None, "random_state": 42}, "RF-200-dNone"),
        (GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}, "GBM-100-lr0.1"),
    ]
    results = {name: run_experiment(cls, params, name) for cls, params, name in configs}
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n🏆 Najlepszy run: {best} (roc_auc={results[best]['roc_auc']:.4f})")
    print("👉 Uruchom `mlflow ui` i otwórz http://localhost:5000, aby porównać runy.")


if __name__ == "__main__":
    main()
