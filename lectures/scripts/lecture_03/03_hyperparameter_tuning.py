"""Wykład 3, sekcja 6 – Automatyczne strojenie hiperparametrów + MLflow.

Każda próba (trial) jest zagnieżdżonym runem (nested run) w runie-rodzicu.
- Jeśli zainstalowana jest Optuna (`pip install optuna`) → użyty zostanie TPE.
- W przeciwnym razie → losowe przeszukiwanie (RandomSearch) z scikit-learn/numpy.

Uruchomienie:
    python lectures/scripts/lecture_03/03_hyperparameter_tuning.py [--trials 20]
"""

import argparse
import os

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=2_000, n_features=20, n_informative=10, random_state=42)


def score(params: dict) -> float:
    model = GradientBoostingClassifier(random_state=42, **params)
    return cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()


def log_trial(i: int, params: dict) -> float:
    with mlflow.start_run(run_name=f"trial-{i:03d}", nested=True):
        mlflow.log_params(params)
        value = score(params)
        mlflow.log_metric("cv_roc_auc", value)
    print(f"trial {i:3d}  roc_auc={value:.4f}  {params}")
    return value


def search_optuna(n_trials: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        return log_trial(trial.number, params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def search_random(n_trials: int):
    rng = np.random.default_rng(42)
    best_params, best_value = None, -np.inf
    for i in range(n_trials):
        params = {
            "n_estimators": int(rng.choice(range(20, 201, 20))),
            "learning_rate": float(10 ** rng.uniform(-2, np.log10(0.3))),
            "max_depth": int(rng.integers(2, 7)),
            "subsample": float(rng.uniform(0.6, 1.0)),
        }
        value = log_trial(i, params)
        if value > best_value:
            best_params, best_value = params, value
    return best_params, best_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    try:
        import optuna  # noqa: F401
        method, search = "optuna-tpe", search_optuna
    except ImportError:
        method, search = "random-search", search_random

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("churn-hpo")
    print(f"Metoda: {method}, liczba prób: {args.trials}\n")

    with mlflow.start_run(run_name=f"hpo-{method}"):
        mlflow.set_tag("search_method", method)
        best_params, best_value = search(args.trials)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", best_value)

    print(f"\n🏆 Najlepsze parametry: {best_params}  (roc_auc={best_value:.4f})")
    print("👉 W `mlflow ui` otwórz eksperyment 'churn-hpo' → widok 'Parallel Coordinates'.")


if __name__ == "__main__":
    main()
