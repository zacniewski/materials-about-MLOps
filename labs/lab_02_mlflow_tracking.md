# Laboratorium 2: Śledzenie Eksperymentów z MLflow

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Podstawowy–Średni
- **Wymagania wstępne:** Lab 1 ukończony, podstawy scikit-learn

## Cele laboratorium
Po tym laboratorium student:
- uruchomi lokalny serwer MLflow i skonfiguruje tracking,
- zaloguje parametry, metryki i artefakty do MLflow,
- porówna eksperymenty w UI MLflow,
- zarejestruje model w MLflow Model Registry.

---

## Część 1: Uruchomienie MLflow (20 min)

### Krok 1.1: Start serwera MLflow

```bash
# Aktywuj środowisko
source mlops-env/bin/activate
cd mlops-project

# Uruchom MLflow UI (w osobnym terminalu lub w tle)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

# Sprawdź czy działa
curl http://localhost:5000/health
# Otwórz w przeglądarce: http://localhost:5000
```

### Krok 1.2: Konfiguracja klienta MLflow

```python
# configs/mlflow_config.py
import mlflow
import os

def setup_mlflow(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "churn-prediction"
) -> str:
    """Konfiguruje MLflow i zwraca ID eksperymentu."""
    mlflow.set_tracking_uri(tracking_uri)
    
    # Utwórz lub pobierz eksperyment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "churn-prediction",
                "team": "mlops-lab",
                "version": "1.0"
            }
        )
        print(f"Utworzono eksperyment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Używam eksperymentu: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id
```

---

## Część 2: Logowanie eksperymentów (40 min)

### Krok 2.1: Podstawowy eksperyment

Utwórz `src/models/experiment.py`:

```python
# src/models/experiment.py
"""Eksperymenty ML z pełnym logowaniem do MLflow."""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)

# Konfiguracja MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction")

FEATURE_COLS = [
    'age', 'income', 'tenure_months', 'num_products',
    'has_credit_card', 'is_active', 'balance',
    'num_transactions_30d', 'income_per_age',
    'balance_per_product', 'is_long_tenure'
]

def load_data(train_path: str, test_path: str):
    """Wczytuje dane treningowe i testowe."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['churn']
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['churn']
    
    return X_train, X_test, y_train, y_test

def compute_metrics(model, X_test, y_test) -> dict:
    """Oblicza kompletny zestaw metryk."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }

def plot_roc_curve(model, X_test, y_test, save_path: str):
    """Rysuje i zapisuje krzywą ROC."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve – Churn Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, save_path: str):
    """Rysuje i zapisuje macierz pomyłek."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Brak churnu', 'Churn'],
        ax=ax
    )
    ax.set_title("Macierz pomyłek")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def run_experiment(
    model_class,
    model_params: dict,
    run_name: str,
    X_train, X_test, y_train, y_test,
    tags: dict = None
) -> tuple[str, dict]:
    """
    Uruchamia jeden eksperyment i loguje wszystko do MLflow.
    
    Returns:
        (run_id, metrics)
    """
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        run_id = run.info.run_id
        
        # --- Logowanie parametrów ---
        mlflow.log_params(model_params)
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("churn_rate_train", float(y_train.mean()))
        
        # --- Trening ---
        t0 = time.time()
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_class(**model_params, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        
        mlflow.log_metric("training_time_seconds", train_time)
        
        # --- Ewaluacja ---
        metrics = compute_metrics(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # --- Cross-validation ---
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
        )
        mlflow.log_metric("cv_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_auc_std", float(cv_scores.std()))
        
        # --- Wykresy jako artefakty ---
        Path("/tmp/mlflow_artifacts").mkdir(exist_ok=True)
        
        plot_roc_curve(pipeline, X_test, y_test, "/tmp/mlflow_artifacts/roc_curve.png")
        plot_confusion_matrix(pipeline, X_test, y_test, "/tmp/mlflow_artifacts/confusion_matrix.png")
        
        mlflow.log_artifact("/tmp/mlflow_artifacts/roc_curve.png", "plots")
        mlflow.log_artifact("/tmp/mlflow_artifacts/confusion_matrix.png", "plots")
        
        # --- Feature importance (dla modeli drzewiastych) ---
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fi_dict = dict(zip(FEATURE_COLS, importances.tolist()))
            
            # Zapisz jako JSON
            with open("/tmp/mlflow_artifacts/feature_importance.json", "w") as f:
                json.dump(fi_dict, f, indent=2)
            mlflow.log_artifact("/tmp/mlflow_artifacts/feature_importance.json")
            
            # Wykres
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(importances)[::-1]
            ax.bar(range(len(FEATURE_COLS)), importances[sorted_idx])
            ax.set_xticks(range(len(FEATURE_COLS)))
            ax.set_xticklabels([FEATURE_COLS[i] for i in sorted_idx], rotation=45, ha='right')
            ax.set_title(f"Feature Importance – {model_class.__name__}")
            plt.tight_layout()
            plt.savefig("/tmp/mlflow_artifacts/feature_importance.png", dpi=100)
            plt.close()
            mlflow.log_artifact("/tmp/mlflow_artifacts/feature_importance.png", "plots")
        
        # --- Logowanie modelu ---
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=f"churn-{model_class.__name__.lower()}"
        )
        
        print(f"✅ {run_name}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1_score']:.4f}")
        print(f"   Run ID: {run_id}")
        
        return run_id, metrics
```

### Krok 2.2: Uruchomienie wielu eksperymentów

```python
# scripts/run_experiments.py
"""Uruchamia serię eksperymentów i porównuje wyniki."""

import sys
sys.path.insert(0, '.')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.experiment import load_data, run_experiment

# Wczytaj dane
X_train, X_test, y_train, y_test = load_data(
    "data/processed/train.parquet",
    "data/processed/test.parquet"
)

# Definicja eksperymentów
experiments = [
    # Baseline: Regresja logistyczna
    (LogisticRegression, {"C": 1.0, "max_iter": 1000}, "LR-baseline",
     {"model_type": "linear", "purpose": "baseline"}),
    
    # Random Forest – różne głębokości
    (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5}, "RF-shallow",
     {"model_type": "tree", "purpose": "experiment"}),
    
    (RandomForestClassifier, {"n_estimators": 100, "max_depth": 10}, "RF-medium",
     {"model_type": "tree", "purpose": "experiment"}),
    
    (RandomForestClassifier, {"n_estimators": 200, "max_depth": 15}, "RF-deep",
     {"model_type": "tree", "purpose": "experiment"}),
    
    # Gradient Boosting
    (GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
     "GBM-default", {"model_type": "boosting", "purpose": "experiment"}),
    
    (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5},
     "GBM-tuned", {"model_type": "boosting", "purpose": "experiment"}),
]

# Uruchom eksperymenty
results = []
for model_class, params, name, tags in experiments:
    run_id, metrics = run_experiment(
        model_class, params, name,
        X_train, X_test, y_train, y_test,
        tags=tags
    )
    results.append({"name": name, "run_id": run_id, **metrics})

# Podsumowanie
results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
print("\n=== Ranking eksperymentów ===")
print(results_df[["name", "auc_roc", "f1_score", "precision", "recall"]].to_string(index=False))
print(f"\nNajlepszy model: {results_df.iloc[0]['name']} (AUC={results_df.iloc[0]['auc_roc']:.4f})")
```

```bash
# Uruchom eksperymenty
python scripts/run_experiments.py

# Otwórz MLflow UI i porównaj wyniki
# http://localhost:5000
```

---

## Część 3: MLflow Model Registry (30 min)

### Krok 3.1: Rejestracja i zarządzanie modelami

```python
# scripts/manage_registry.py
"""Zarządzanie modelami w MLflow Model Registry."""

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "churn-randomforestclassifier"

def list_registered_models():
    """Wyświetla wszystkie zarejestrowane modele."""
    print("\n=== Zarejestrowane modele ===")
    for rm in client.search_registered_models():
        print(f"\nModel: {rm.name}")
        for mv in rm.latest_versions:
            print(f"  v{mv.version} | stage={mv.current_stage} | run={mv.run_id[:8]}...")

def promote_best_model(model_name: str, min_auc: float = 0.80):
    """Promuje najlepszy model do Staging."""
    
    # Znajdź wszystkie wersje modelu
    versions = client.search_model_versions(f"name='{model_name}'")
    
    best_version = None
    best_auc = 0
    
    for mv in versions:
        run = client.get_run(mv.run_id)
        auc = run.data.metrics.get("auc_roc", 0)
        
        if auc > best_auc and auc >= min_auc:
            best_auc = auc
            best_version = mv.version
    
    if best_version is None:
        print(f"❌ Brak modelu z AUC >= {min_auc}")
        return None
    
    # Przenieś do Staging
    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Staging",
        archive_existing_versions=False
    )
    
    # Dodaj opis
    client.update_model_version(
        name=model_name,
        version=best_version,
        description=f"Najlepszy model. AUC={best_auc:.4f}. Promowany automatycznie."
    )
    
    print(f"✅ Model v{best_version} promowany do Staging (AUC={best_auc:.4f})")
    return best_version

def promote_to_production(model_name: str, version: str):
    """Promuje model ze Staging do Production."""
    
    # Archiwizuj poprzedni model produkcyjny
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=pv.version,
            stage="Archived"
        )
        print(f"📦 Zarchiwizowano v{pv.version}")
    
    # Promuj nowy model
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"🚀 Model v{version} wdrożony do Production!")

def load_production_model(model_name: str):
    """Ładuje aktualny model produkcyjny."""
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Załadowano model produkcyjny: {model_uri}")
    return model

# Uruchomienie
list_registered_models()

staging_version = promote_best_model(MODEL_NAME, min_auc=0.75)

if staging_version:
    # Symulacja testów przed wdrożeniem
    print("\nUruchamianie testów integracyjnych...")
    # ... testy ...
    print("✅ Testy przeszły pomyślnie")
    
    promote_to_production(MODEL_NAME, staging_version)
    
    # Załaduj model produkcyjny
    prod_model = load_production_model(MODEL_NAME)
    print(f"Model załadowany: {type(prod_model)}")
```

```bash
python scripts/manage_registry.py
```

---

## Część 4: Porównanie eksperymentów (30 min)

### Krok 4.1: Programowe porównanie przez API

```python
# scripts/compare_experiments.py
"""Porównuje eksperymenty i generuje raport."""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

def get_experiment_results(experiment_name: str) -> pd.DataFrame:
    """Pobiera wyniki wszystkich runów z eksperymentu."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Eksperyment '{experiment_name}' nie istnieje")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.auc_roc DESC"]
    )
    
    rows = []
    for run in runs:
        rows.append({
            "run_name": run.data.tags.get("mlflow.runName", "N/A"),
            "model_class": run.data.params.get("model_class", "N/A"),
            "auc_roc": run.data.metrics.get("auc_roc", 0),
            "f1_score": run.data.metrics.get("f1_score", 0),
            "precision": run.data.metrics.get("precision", 0),
            "recall": run.data.metrics.get("recall", 0),
            "cv_auc_mean": run.data.metrics.get("cv_auc_mean", 0),
            "cv_auc_std": run.data.metrics.get("cv_auc_std", 0),
            "training_time": run.data.metrics.get("training_time_seconds", 0),
            "run_id": run.info.run_id,
            "status": run.info.status
        })
    
    return pd.DataFrame(rows)

# Pobierz wyniki
df = get_experiment_results("churn-prediction")

print("=== Wyniki eksperymentów ===")
print(df[["run_name", "auc_roc", "f1_score", "cv_auc_mean", "training_time"]].to_string(index=False))

# Wykres porównawczy
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AUC-ROC
axes[0].barh(df['run_name'], df['auc_roc'], color='steelblue')
axes[0].set_xlabel('AUC-ROC')
axes[0].set_title('AUC-ROC per model')
axes[0].axvline(x=0.80, color='red', linestyle='--', label='Próg 0.80')
axes[0].legend()

# F1 Score
axes[1].barh(df['run_name'], df['f1_score'], color='green')
axes[1].set_xlabel('F1 Score')
axes[1].set_title('F1 Score per model')

# Czas treningu
axes[2].barh(df['run_name'], df['training_time'], color='orange')
axes[2].set_xlabel('Czas treningu (s)')
axes[2].set_title('Czas treningu per model')

plt.tight_layout()
plt.savefig("reports/figures/experiment_comparison.png", dpi=100, bbox_inches='tight')
print("\nWykres zapisany: reports/figures/experiment_comparison.png")

# Najlepszy model
best = df.iloc[0]
print(f"\n🏆 Najlepszy model: {best['run_name']}")
print(f"   AUC-ROC: {best['auc_roc']:.4f}")
print(f"   F1:      {best['f1_score']:.4f}")
print(f"   CV AUC:  {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}")
```

```bash
python scripts/compare_experiments.py
```

---

## Zadania do samodzielnego wykonania

1. **Dodaj nowy eksperyment** z XGBoost (`pip install xgboost`) i porównaj z Random Forest.
2. **Zaloguj dodatkową metrykę** – AUC-PR (Precision-Recall AUC) dla niezbalansowanych klas.
3. **Dodaj tag** `data_version` do każdego runu i filtruj po nim w UI.
4. **Napisz skrypt** który automatycznie promuje model do Production jeśli jego AUC jest lepsze od aktualnego modelu produkcyjnego.

## Pytania kontrolne

1. Jaka jest różnica między `mlflow.log_param` a `mlflow.log_metric`?
2. Co to jest MLflow Model Registry i jakie stany może mieć model?
3. Dlaczego logujemy CV AUC zamiast tylko AUC na zbiorze testowym?
4. Jak załadować model z MLflow Registry bez znajomości ścieżki do pliku?

## Podsumowanie

W tym laboratorium:
- ✅ Uruchomiłeś lokalny serwer MLflow
- ✅ Zalogowałeś parametry, metryki, wykresy i modele
- ✅ Porównałeś 6 różnych modeli w UI MLflow
- ✅ Zarejestrowałeś model w Model Registry i zarządzałeś jego cyklem życia
