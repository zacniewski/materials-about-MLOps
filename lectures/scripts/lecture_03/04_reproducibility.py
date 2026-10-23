"""Wykład 3, sekcje 8 i 10 – Reprodukowalność eksperymentu i logowanie porażek.

- set_seed(): ustala wszystkie źródła losowości,
- loguje do MLflow: commit git, hash danych, wersje bibliotek, środowisko,
- pokazuje, że dwa runy z tym samym seedem dają identyczne metryki,
- pokazuje logowanie NIEUDANEGO runu (status FAILED + tag z błędem).

Uruchomienie:
    python lectures/scripts/lecture_03/04_reproducibility.py
"""

import hashlib
import os
import platform
import random
import subprocess

import mlflow
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # opcjonalnie, jeśli zainstalowany jest PyTorch
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def data_hash(X: np.ndarray) -> str:
    return hashlib.sha256(X.tobytes()).hexdigest()[:12]


def train_run(seed: int, run_name: str) -> float:
    set_seed(seed)
    X, y = make_classification(n_samples=3000, n_features=20, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_commit": git_commit(), "data_hash": data_hash(X),
                         "python": platform.python_version(), "numpy": np.__version__,
                         "sklearn": sklearn.__version__, "os": platform.platform()})
        params = {"seed": seed, "n_estimators": 100, "max_depth": 6}
        mlflow.log_params(params)
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        mlflow.log_metric("roc_auc", auc)
    return auc


def failing_run() -> None:
    with mlflow.start_run(run_name="failing-run") as run:
        mlflow.log_param("learning_rate", 10.0)
        try:
            raise ValueError("Loss = NaN – zbyt duży learning rate")
        except ValueError as e:
            mlflow.set_tags({"failed": "true", "error": str(e)})
            mlflow.end_run(status="FAILED")
            print(f"⚠️  Run {run.info.run_id[:8]} zalogowany jako FAILED: {e}")


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("reproducibility-demo")

    a = train_run(42, "seed42-a")
    b = train_run(42, "seed42-b")
    c = train_run(7, "seed7")
    print(f"seed=42 run A: roc_auc={a:.6f}")
    print(f"seed=42 run B: roc_auc={b:.6f}  → identyczne: {a == b}")
    print(f"seed=7       : roc_auc={c:.6f}  → inny seed, inny wynik")
    failing_run()
    print("👉 W `mlflow ui` sprawdź tagi git_commit/data_hash oraz run ze statusem FAILED.")


if __name__ == "__main__":
    main()
