# Wykład 3: Eksperymentowanie i Śledzenie Eksperymentów

## Cel wykładu
Po tym wykładzie student:
- rozumie potrzebę systematycznego śledzenia eksperymentów ML,
- potrafi używać MLflow do logowania parametrów, metryk i artefaktów,
- zna wzorce organizacji eksperymentów,
- potrafi porównywać eksperymenty i wybierać najlepszy model.

## Skrypty do wykładu

Skrypty demonstracyjne znajdują się w `lectures/scripts/lecture_03/`. Pełną instrukcję znajdziesz w [`lectures/scripts/README.md`](scripts/README.md).

| Kiedy (sekcja) | Skrypt | Cel demonstracji |
|---|---|---|
| 3. Podstawy MLflow Tracking | `01_mlflow_experiments.py` | logowanie parametrów, metryk, artefaktów i modeli |
| 4. Porównywanie eksperymentów | `02_compare_runs.py` (po `01`) | `search_runs`, ładowanie modelu, Model Registry + alias |
| 6. Optuna | `03_hyperparameter_tuning.py --trials 15` | HPO z zagnieżdżonymi runami |
| 8 i 10. Reprodukowalność, pułapki | `04_reproducibility.py` | seed, commit git, hash danych, run FAILED |

**Jak:** przed wykładem uruchom `rm -rf mlruns`, żeby zacząć od pustego trackingu. Następnie uruchamiaj skrypty z katalogu głównego repozytorium (`python lectures/scripts/lecture_03/<skrypt>`). W drugim terminalu uruchom `mlflow ui --port 5000` i po każdym skrypcie pokazuj wyniki w przeglądarce (http://localhost:5000). Domyślnie skrypty zapisują runy w `./mlruns`. Żeby użyć serwera, ustaw `MLFLOW_TRACKING_URI`.

---

## 1. Problem niekontrolowanych eksperymentów

Typowy scenariusz bez śledzenia eksperymentów:

```
model_final.pkl
model_final_v2.pkl
model_final_v2_NAPRAWDE_FINAL.pkl
model_best_20240115.pkl
model_DO_UZYCIA.pkl   ← który to jest?!
```

**Problemy:**
- Nie wiadomo, jakie hiperparametry dały najlepszy wynik.
- Nie wiadomo, na jakich danych trenowano.
- Nie można odtworzyć eksperymentu.
- Trudno porównać różne podejścia.


### Co dokładnie należy logować?

Eksperyment jest powtarzalny tylko wtedy, gdy zapiszemy **wszystkie** jego wejścia i wyjścia:

```mermaid
flowchart LR
    subgraph Wejścia
        C[Kod\ncommit git]
        D[Dane\nwersja / hash]
        P[Parametry\nhiperparametry, seed]
        E[Środowisko\nwersje bibliotek, sprzęt]
    end
    C & D & P & E --> R((Run))
    subgraph Wyjścia
        M[Metryki\nliczby, krzywe]
        A[Artefakty\nwykresy, raporty]
        MD[Model\n+ sygnatura]
    end
    R --> M & A & MD
```

| Kategoria | Przykłady | MLflow API |
|---|---|---|
| Parametry | `learning_rate`, `max_depth`, `seed` | `log_param(s)` |
| Metryki | `roc_auc`, `loss` per epoka | `log_metric(s)` (z `step`) |
| Tagi | `git_commit`, `data_hash`, `author` | `set_tag(s)` |
| Artefakty | confusion matrix, feature importance | `log_artifact`, `log_figure` |
| Dane | nazwa, źródło, digest zbioru | `log_input` |
| Model | wytrenowany model, sygnatura | `mlflow.<flavor>.log_model` |

### Cykl pracy eksperymentalnej

```mermaid
flowchart LR
    H[Hipoteza\n„cecha X poprawi AUC”] --> EX[Eksperyment\nrun w MLflow]
    EX --> AN[Analiza\nporównanie z baseline]
    AN -->|odrzucona| H
    AN -->|potwierdzona| REG[Rejestracja modelu]
```

Dobra praktyka: **zawsze zaczynaj od baseline'u** (np. model większościowy, regresja logistyczna) – bez punktu odniesienia nie wiadomo, czy złożony model wnosi wartość.

---

## 2. MLflow – platforma do śledzenia eksperymentów

**MLflow** to open-source'owa platforma do zarządzania cyklem życia ML, składająca się z czterech komponentów:

```mermaid
graph TB
    MLflow --> TR[MLflow Tracking\nLogowanie eksperymentów]
    MLflow --> PR[MLflow Projects\nReprodukowalność]
    MLflow --> MO[MLflow Models\nFormat modeli]
    MLflow --> RE[MLflow Registry\nRejestr modeli]
```

### Instalacja i uruchomienie

```bash
pip install mlflow scikit-learn pandas

# Uruchomienie UI (domyślnie http://localhost:5000)
mlflow ui
```

---

## 3. Podstawy MLflow Tracking

### Kluczowe pojęcia

```mermaid
graph LR
    EXP[Experiment\nGrupa powiązanych\neksperymntów] --> RUN[Run\nJeden eksperyment]
    RUN --> PAR[Parameters\nHiperparametry]
    RUN --> MET[Metrics\nWyniki ewaluacji]
    RUN --> ART[Artifacts\nModele, wykresy, dane]
    RUN --> TAG[Tags\nMetadane]
```

### Przykład: Pełny eksperyment z MLflow

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                              f1_score, classification_report)
import matplotlib.pyplot as plt
import json

# Generowanie danych
X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Konfiguracja MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction-v1")

def evaluate_model(model, X_test, y_test) -> dict:
    """Oblicza metryki ewaluacji modelu."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred),
    }

def run_experiment(model_class, params: dict, run_name: str):
    """Uruchamia jeden eksperyment i loguje wyniki do MLflow."""
    
    with mlflow.start_run(run_name=run_name) as run:
        # Logowanie parametrów
        mlflow.log_params(params)
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Trening modelu
        model = model_class(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Ewaluacja
        metrics = evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics["cv_auc_mean"] = cv_scores.mean()
        metrics["cv_auc_std"] = cv_scores.std()
        
        # Logowanie metryk
        mlflow.log_metrics(metrics)
        
        # Logowanie modelu
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"churn-{model_class.__name__.lower()}"
        )
        
        # Logowanie wykresu feature importance (dla RF)
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            ax.bar(range(10), importances[indices])
            ax.set_title('Top 10 Feature Importances')
            ax.set_xlabel('Feature Index')
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close()
        
        # Logowanie raportu klasyfikacji jako artefakt
        report = classification_report(y_test, model.predict(X_test))
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt")
        
        print(f"Run: {run_name}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  F1:      {metrics['f1_score']:.4f}")
        print(f"  Run ID:  {run.info.run_id}")
        
        return run.info.run_id, metrics

# Eksperymenty z różnymi modelami i hiperparametrami
experiments = [
    (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5}, "RF-shallow"),
    (RandomForestClassifier, {"n_estimators": 200, "max_depth": 10}, "RF-deep"),
    (GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1}, "GBM-default"),
    (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05}, "GBM-slow"),
    (LogisticRegression, {"C": 1.0, "max_iter": 1000}, "LR-default"),
]

results = []
for model_class, params, name in experiments:
    run_id, metrics = run_experiment(model_class, params, name)
    results.append({"name": name, "run_id": run_id, **metrics})

# Podsumowanie wyników
results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
print("\n=== Ranking eksperymentów ===")
print(results_df[["name", "auc_roc", "f1_score", "cv_auc_mean"]].to_string(index=False))
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_03/01_mlflow_experiments.py`  
> **Kiedy:** po omówieniu kodu z sekcji 3. Trwa około 40 s, więc w tym czasie możesz omówić kluczowe pojęcia.  
> **Co pokazać:** w MLflow UI zaznacz 5 runów i kliknij **Compare** (wykres parallel coordinates). Otwórz run GBM i pokaż metrykę krokową `staged_roc_auc`, artefakty `evaluation/confusion_matrix.png` oraz model z sygnaturą i `input_example`.

---

## 4. Porównywanie eksperymentów

### Programowe porównanie przez MLflow API

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Pobierz wszystkie runy z eksperymentu
experiment = client.get_experiment_by_name("churn-prediction-v1")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.auc_roc DESC"],
    max_results=10
)

print("Top 5 eksperymentów wg AUC-ROC:")
for i, run in enumerate(runs[:5], 1):
    print(f"{i}. {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"   AUC-ROC: {run.data.metrics.get('auc_roc', 0):.4f}")
    print(f"   F1:      {run.data.metrics.get('f1_score', 0):.4f}")
    print(f"   Params:  {run.data.params}")
    print()

# Pobierz najlepszy model
best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"Najlepszy model: {best_run_id}")

# Załaduj najlepszy model
best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_03/02_compare_runs.py`  
> **Kiedy:** bezpośrednio po kodzie z sekcji 4.  
> **Co pokazać:** ranking runów w terminalu, filtr `tags.model_type = ... and metrics.roc_auc > 0.9` (ta sama składnia działa w polu wyszukiwania UI) oraz zakładkę **Models** w UI z modelem `churn-classifier` i aliasem `champion`.


### Czy różnica między runami jest istotna?

Różnica AUC 0.871 vs 0.874 może wynikać wyłącznie z losowości podziału danych. Zanim ogłosisz „lepszy model”:
1. Uruchom każdą konfigurację z **kilkoma seedami** lub w **k-fold CV** i porównuj średnią ± odchylenie.
2. Użyj testu sparowanego (te same foldy) lub **bootstrapu** na zbiorze testowym.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)   # te same foldy!
a = cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=cv, scoring="roc_auc")
b = cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=cv, scoring="roc_auc")

print(f"RF : {a.mean():.4f} ± {a.std():.4f}")
print(f"GBM: {b.mean():.4f} ± {b.std():.4f}")
t, p = stats.ttest_rel(a, b)
print(f"sparowany t-test: p = {p:.3f} → {'różnica istotna' if p < 0.05 else 'brak dowodu na różnicę'}")
```

> Wyniki z CV warto logować w MLflow jako `roc_auc_mean` i `roc_auc_std` – wtedy porównanie w UI uwzględnia niepewność.

---

## 5. Organizacja eksperymentów – dobre praktyki

### Hierarchia eksperymentów

```mermaid
graph TD
    PROJ[Projekt:\nchurn-prediction] --> EXP1[Eksperyment:\nbaseline-models]
    PROJ --> EXP2[Eksperyment:\nfeature-engineering-v2]
    PROJ --> EXP3[Eksperyment:\nhyperparameter-tuning]
    
    EXP1 --> R1[Run: LR]
    EXP1 --> R2[Run: RF]
    EXP1 --> R3[Run: GBM]
    
    EXP2 --> R4[Run: RF + new features]
    EXP2 --> R5[Run: GBM + new features]
    
    EXP3 --> R6[Run: GBM optuna trial 1]
    EXP3 --> R7[Run: GBM optuna trial 2]
    EXP3 --> R8[Run: GBM optuna trial N]
```

### Konwencje nazewnictwa

```python
# Dobra konwencja nazewnictwa runów
run_name = f"{model_type}-{feature_set}-{date}-{author}"
# Przykład: "GBM-v2features-20240115-jkowalski"

# Tagi dla łatwego filtrowania
tags = {
    "author": "jan.kowalski",
    "feature_set": "v2",
    "data_version": "2024-01-15",
    "purpose": "hyperparameter_search",
    "environment": "development"
}

with mlflow.start_run(run_name=run_name, tags=tags):
    pass  # ... trening
```

---

## 6. Optuna – automatyczne strojenie hiperparametrów

**Optuna** to framework do automatycznej optymalizacji hiperparametrów (AutoML).

```python
import optuna
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Integracja Optuna z MLflow
mlflow.set_experiment("hyperparameter-tuning-gbm")

def objective(trial: optuna.Trial) -> float:
    """Funkcja celu dla Optuna."""
    
    # Przestrzeń hiperparametrów
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "random_state": 42
    }
    
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        
        model = GradientBoostingClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        mean_auc = scores.mean()
        mlflow.log_metric("cv_auc_mean", mean_auc)
        mlflow.log_metric("cv_auc_std", scores.std())
    
    return mean_auc

# Uruchomienie optymalizacji
with mlflow.start_run(run_name="optuna-study"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=50, timeout=300)
    
    # Logowanie najlepszych wyników
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_auc", study.best_value)
    
    print(f"Najlepsze hiperparametry: {study.best_params}")
    print(f"Najlepsze AUC: {study.best_value:.4f}")
```

### Wizualizacja przestrzeni hiperparametrów

```python
import optuna.visualization as vis

# Wykres ważności hiperparametrów
fig = vis.plot_param_importances(study)
fig.write_html("/tmp/param_importances.html")
mlflow.log_artifact("/tmp/param_importances.html")

# Historia optymalizacji
fig = vis.plot_optimization_history(study)
fig.write_html("/tmp/optimization_history.html")
mlflow.log_artifact("/tmp/optimization_history.html")
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_03/03_hyperparameter_tuning.py --trials 15`  
> **Kiedy:** po omówieniu integracji Optuna + MLflow.  
> **Co pokazać:** w eksperymencie `churn-hpo` rozwiń run-rodzica z próbami jako runami zagnieżdżonymi, a następnie użyj **Compare → Parallel Coordinates**. Bez zainstalowanej Optuny skrypt użyje random search. Warto zestawić oba podejścia.


### Strategie przeszukiwania hiperparametrów – teoria

| Metoda | Idea | Zalety | Wady |
|---|---|---|---|
| Grid search | wszystkie kombinacje siatki | prosta, pełna | koszt rośnie wykładniczo z liczbą parametrów |
| Random search | losowe punkty | lepiej pokrywa ważne wymiary (Bergstra & Bengio, 2012) | nie uczy się z poprzednich prób |
| Bayesowska (TPE, GP) | model zastępczy przewiduje, gdzie szukać | mniej prób do dobrego wyniku | sekwencyjna, trudniej zrównoleglić |
| Hyperband / ASHA | wczesne przerywanie słabych prób (pruning) | ogromna oszczędność czasu | wymaga metryki pośredniej (np. per epoka) |

Dlaczego random search bije grid search? Gdy tylko jeden z dwóch parametrów jest istotny, siatka 3×3 testuje tylko **3** różne wartości ważnego parametru, a 9 losowych prób – **9** różnych wartości.

```mermaid
flowchart LR
    S[Start] --> Q[Sampler proponuje\nhiperparametry]
    Q --> T[Trening\n+ metryki pośrednie]
    T --> PR{Pruner:\nczy próba rokuje?}
    PR -->|nie| X[Przerwij próbę]
    PR -->|tak| F[Dokończ i zapisz wynik]
    X & F --> U[Aktualizacja modelu\nzastępczego TPE]
    U --> Q
```

**TPE (Tree-structured Parzen Estimator)** – domyślny sampler Optuny – dzieli dotychczasowe próby na „dobre” (górne γ%, np. 25%) i „złe”, modeluje gęstości $l(x)$ i $g(x)$ hiperparametrów w obu grupach i proponuje punkty maksymalizujące $l(x)/g(x)$.

> **Pułapka:** strojenie na zbiorze testowym. Hiperparametry wybieramy na walidacji (lub CV), a test używamy **raz**, na końcu – inaczej wynik testowy jest optymistycznie obciążony.

---

## 7. Śledzenie eksperymentów głębokiego uczenia

### Integracja z PyTorch

```python
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_neural_network(
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.001,
    epochs: int = 50,
    batch_size: int = 256
):
    """Trenuje sieć neuronową z logowaniem do MLflow."""
    
    with mlflow.start_run():
        # Logowanie hiperparametrów
        params = {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
        }
        mlflow.log_params(params)
        
        # Przygotowanie danych
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train.values)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Model i optymalizator
        model = SimpleNN(X_train.shape[1], hidden_dim, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Trening z logowaniem metryk per epoka
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Logowanie metryki per krok (epoka)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Logowanie modelu PyTorch
        mlflow.pytorch.log_model(model, "neural_network")
        
        return model
```

---

## 8. Reproducibility – gwarancja powtarzalności

```python
import os
import sys
import platform
import subprocess
import mlflow

def log_environment_info():
    """Loguje informacje o środowisku do MLflow."""
    
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    
    # Wersje bibliotek
    import sklearn, numpy, pandas
    lib_versions = {
        "sklearn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
    }
    
    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
        lib_versions["git_commit"] = git_hash
    except Exception:
        lib_versions["git_commit"] = "unknown"
    
    mlflow.log_params({**env_info, **lib_versions})
    
    # Zapisz requirements.txt jako artefakt
    subprocess.run(["pip", "freeze"], 
                   stdout=open("/tmp/requirements_snapshot.txt", "w"))
    mlflow.log_artifact("/tmp/requirements_snapshot.txt")

# Użycie w eksperymencie
with mlflow.start_run():
    log_environment_info()
    # ... reszta eksperymentu
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_03/04_reproducibility.py`  
> **Kiedy:** po sekcji 8. Uzupełnia też sekcję 10 (logowanie nieudanych eksperymentów).  
> **Co pokazać:** dwa runy z `seed=42` dają identyczne `roc_auc`. Pokaż w UI tagi `git_commit`, `data_hash` i wersje bibliotek oraz run `failing-run` ze statusem FAILED.


### Źródła niedeterminizmu

| Źródło | Przykład | Jak kontrolować |
|---|---|---|
| Generatory losowe | inicjalizacja wag, podział danych, bootstrap | `random`, `numpy`, `torch` seed, `random_state` |
| Dane | ktoś nadpisał plik CSV | DVC, hash danych w tagu runu |
| Kolejność danych | `SELECT` bez `ORDER BY`, wielowątkowe ładowanie | sortowanie, seed `DataLoader` |
| Biblioteki | inna wersja sklearn zmienia domyślne parametry | `requirements.txt` z wersjami, Docker |
| Sprzęt | niedeterministyczne operacje cuDNN na GPU | `torch.use_deterministic_algorithms(True)` |
| Arytmetyka | inna kolejność sumowania float w obliczeniach równoległych | akceptacja tolerancji (np. ±1e-6) |

### Lineage – pochodzenie modelu

Pełny rodowód pozwala odpowiedzieć na pytanie audytora: *„Na jakich danych i jakim kodem powstał model, który odrzucił ten wniosek kredytowy?”*

```mermaid
flowchart LR
    D[(data v3\nmd5: a1b2…)] --> R[Run 7f3c\ncommit 9e1d]
    C[train.py\ncommit 9e1d] --> R
    R --> M[Model churn v5]
    M --> P[Produkcja\nalias champion]
    P --> PR[Predykcja #123]
```

---

## 9. Weights & Biases (W&B) – alternatywa dla MLflow

**W&B** to chmurowa platforma do śledzenia eksperymentów, popularna w środowiskach badawczych.

```python
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Inicjalizacja projektu W&B
wandb.init(
    project="churn-prediction",
    name="RF-experiment-01",
    config={
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": 42
    }
)

config = wandb.config

# Trening
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    random_state=config.random_state
)
model.fit(X_train, y_train)

# Logowanie metryk
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
wandb.log({"auc_roc": auc, "n_features": X_train.shape[1]})

# Logowanie modelu
wandb.save("model.pkl")

wandb.finish()
```

### Porównanie MLflow vs W&B

| Cecha | MLflow | Weights & Biases |
|-------|--------|-----------------|
| Hosting | Self-hosted / Managed | Cloud (SaaS) |
| Koszt | Open Source (darmowy) | Freemium |
| UI | Podstawowy | Zaawansowany |
| Integracje | Szerokie | Bardzo szerokie |
| Kolaboracja | Ograniczona | Wbudowana |
| Model Registry | Tak | Tak |


### Model Registry – cykl życia wersji modelu

```mermaid
stateDiagram-v2
    [*] --> Zarejestrowany: register_model()
    Zarejestrowany --> Challenger: walidacja OK
    Challenger --> Champion: lepszy od obecnego\n(test A/B, shadow)
    Champion --> Zarchiwizowany: nowy champion
    Challenger --> Zarchiwizowany: przegrał porównanie
    Zarchiwizowany --> Champion: rollback
```

W MLflow ≥ 2.9 zamiast etapów (`Staging`/`Production`, obecnie przestarzałe) używa się **aliasów** (`champion`, `challenger`) i tagów. Kod serwujący ładuje model po aliasie, więc wdrożenie i rollback to tylko przestawienie aliasu:

```python
from mlflow import MlflowClient
import mlflow

client = MlflowClient()
client.set_registered_model_alias("churn-classifier", "champion", version=5)
model = mlflow.pyfunc.load_model("models:/churn-classifier@champion")
# rollback:
client.set_registered_model_alias("churn-classifier", "champion", version=4)
```

---

## 10. Typowe pułapki w śledzeniu eksperymentów

> ⚠️ **Pułapka 1: Porównywanie jabłek z gruszkami**
> Porównywanie modeli trenowanych na różnych wersjach danych lub z różnym preprocessingiem. Zawsze loguj `data_version` i `preprocessing_version` jako parametry.

> ⚠️ **Pułapka 2: Overfitting do zbioru testowego**
> Wielokrotne strojenie hiperparametrów na tym samym zbiorze testowym prowadzi do overfittingu. Rozwiązanie: używaj trzech zbiorów (train/validation/test) lub cross-validation, a zbiór testowy traktuj jako „ostateczny egzamin".

> ⚠️ **Pułapka 3: Brak logowania negatywnych wyników**
> Zespoły często logują tylko udane eksperymenty. Negatywne wyniki są równie wartościowe — pokazują, czego nie warto próbować. Loguj **każdy** eksperyment, nawet nieudany.

```python
# ✅ Dobra praktyka: loguj też nieudane eksperymenty z odpowiednim tagiem
with mlflow.start_run(run_name="failed-experiment", tags={"status": "failed"}):
    mlflow.log_param("reason", "OOM - model zbyt duży dla dostępnej pamięci")
    mlflow.log_param("model_class", "TransformerXL")
    mlflow.log_param("memory_required_gb", 32)
    mlflow.log_param("memory_available_gb", 16)
```

> ⚠️ **Pułapka 4: Metryki ML vs metryki biznesowe**
> Wysoki AUC-ROC nie zawsze oznacza wartość biznesową. Model z AUC=0.85 może generować więcej zysku niż model z AUC=0.92, jeśli lepiej identyfikuje klientów o wysokiej wartości. Zawsze loguj metryki biznesowe obok metryk ML.

### Case Study: Netflix — eksperymenty na skalę

**Netflix** prowadzi tysiące eksperymentów ML jednocześnie (rekomendacje, thumbnails, personalizacja UI):
- Każdy eksperyment jest śledzony z pełnym kontekstem: dane, parametry, metryki ML i metryki biznesowe (czas oglądania, retencja).
- Używają wewnętrznej platformy do A/B testowania modeli na żywym ruchu.
- Kluczowa lekcja: **metryka offline (AUC) nie zawsze koreluje z metryką online (czas oglądania)**. Dlatego każdy model przechodzi testy A/B przed pełnym wdrożeniem.

---

## Pytania kontrolne i do dyskusji

1. Dlaczego systematyczne śledzenie eksperymentów jest ważniejsze niż sam wybór algorytmu?
2. Jakie informacje powinien zawierać każdy zalogowany eksperyment w MLflow?
3. Czym różni się `mlflow.log_param` od `mlflow.log_metric`? Kiedy użyjesz każdego?
4. Wyjaśnij, jak Optuna wybiera kolejne hiperparametry do przetestowania (TPE sampler).
5. Porównaj MLflow i Weights & Biases — kiedy wybierzesz jedno, a kiedy drugie?
6. Co to jest overfitting do zbioru testowego i jak go uniknąć?
7. **Dyskusja:** Czy warto logować eksperymenty, które zakończyły się niepowodzeniem? Uzasadnij.


### Proponowane odpowiedzi

1. Bez trackingu nie da się rzetelnie porównać wyników ani odtworzyć najlepszego modelu, więc nawet „lepszy algorytm” niewiele daje. Największą wartość daje kontrola procesu uczenia, nie pojedynczy wybór modelu.
2. Minimum: parametry, metryki, artefakty modelu, wersja danych, wersja kodu (commit), środowisko (biblioteki), czas uruchomienia i tagi kontekstowe. To zapewnia pełną reprodukowalność i audyt.
3. `mlflow.log_param` służy do zapisu konfiguracji (np. `max_depth=8`), a `mlflow.log_metric` do wyników liczbowych (np. `auc=0.89`) zwykle także po epokach/krokach. Parametry opisują wejście eksperymentu, metryki jego jakość.
4. TPE buduje probabilistyczny model przestrzeni hiperparametrów i wybiera kolejne konfiguracje, które mają najwyższą oczekiwaną poprawę względem dotychczasowych wyników. Dzięki temu szuka mądrzej niż brute force.
5. MLflow wybiorę, gdy potrzebuję open-source i self-hostingu; W&B, gdy priorytetem jest współpraca zespołowa i bogate UI SaaS. Oba narzędzia wspierają tracking, ale różnią się stylem pracy i kosztem.
6. Overfitting do testu to strojenie decyzji pod jeden zbiór testowy, przez co wynik testowy przestaje być obiektywny. Zapobiegamy temu przez osobny validation set/CV i użycie testu dopiero na końcu.
7. Tak, warto logować także nieudane eksperymenty, bo oszczędza to czas i zapobiega powtarzaniu tych samych błędów. To też buduje wiedzę zespołu o ograniczeniach danych i modeli.


---

## Podsumowanie

- Systematyczne śledzenie eksperymentów jest kluczowe dla reprodukowalności i porównywania modeli.
- **MLflow** oferuje kompletne rozwiązanie: tracking, registry, serving.
- Loguj **wszystko**: parametry, metryki, artefakty, środowisko, wersję kodu — także nieudane eksperymenty.
- **Optuna** automatyzuje poszukiwanie optymalnych hiperparametrów (TPE, pruning).
- Dobra organizacja eksperymentów (nazewnictwo, tagi, hierarchia) oszczędza czas w przyszłości.
- Pamiętaj o różnicy między metrykami ML a metrykami biznesowymi.

## Literatura i zasoby

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764)
- [Netflix Technology Blog – ML Experimentation](https://netflixtechblog.com/)
- [How to Track ML Experiments (Neptune.ai Blog)](https://neptune.ai/blog/ml-experiment-tracking)
