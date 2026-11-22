# Starter kit MLOps — laboratoria 1–3

Gotowy, mały projekt do ćwiczeń. Skopiuj katalog `mlops-project` do własnego repozytorium i pracuj w jego katalogu głównym. Pliki można czytać i uruchamiać od razu; podczas laboratoriów skup się na przepływie danych, artefaktach i wynikach, a nie na przepisywaniu kodu.

## Szybki start (przed laboratorium 1)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
python scripts/generate_data.py --n-samples 10000
git init
dvc init
dvc add data/raw/customers.parquet
dvc repro
```

Na VM kursu (Ubuntu 24.04) używajcie wspólnego venv z repozytorium kursu (`source ~/mlops/materials-about-MLOps/.venv/bin/activate`) — szczegóły w `labs-PL/BLOK_1_PRZED_ZAJECIAMI.md`. `scripts/check_env.py` sprawdza Pythona, venv, wersje pakietów z `requirements.txt`, Git, DVC i wolny port MLflow; każde `[BŁĄD]` ma podpowiedź naprawy.

Jeśli Git wymaga konfiguracji autora, ustaw lokalne `user.name` i `user.email`. Dodaj do Git kod, konfigurację, pliki `*.dvc`, `dvc.yaml`, `dvc.lock` i `.gitignore`; nie dodawaj danych ani plików modelu bezpośrednio. DVC przechowuje ich zawartość w cache lokalnym.

## Co jest w pakiecie

| Obszar | Pliki | Rezultat |
|---|---|---|
| Dane i pipeline | `scripts/generate_data.py`, `src/data/preprocess.py`, `dvc.yaml`, `params.yaml` | deterministyczne dane i powtarzalny podział train/test |
| Baseline | `src/models/train.py` | model Random Forest i metryki accuracy, F1, ROC-AUC na zbiorze treningowym |
| Tracking | `src/models/experiment.py`, `scripts/run_experiments.py`, `scripts/compare_experiments.py`, `scripts/registry_aliases.py` | porównywalne runy MLflow i alias modelu |
| Jakość danych | `src/data/validation.py`, `src/data/anomaly_detection.py`, `scripts/validate_data.py`, `scripts/validate_gx.py` | reguły jakości, IQR i opcjonalne ćwiczenie Great Expectations |
| Cechy | `src/features/feature_store.py` | offline feature store Parquet z rejestrem JSON |
| Sprawdzenie postępu | `scripts/check_progress.py` | lista brakujących artefaktów Labs 1–3 |
| Kontrola środowiska | `scripts/check_env.py` | gotowość venv, pakietów, Git/DVC i portu MLflow przed zajęciami |

## Laboratorium 1

- `python scripts/generate_data.py` przygotowuje `data/raw/customers.parquet` ze schematem z Lab 1.
- `dvc add data/raw/customers.parquet` wersjonuje dane; `dvc repro` wykonuje preprocessing i trening.
- `dvc dag`, `dvc status` i `dvc metrics show` pomagają sprawdzić przepływ i rezultat.
- `pytest tests/unit -q` uruchamia testy podziału danych.

## Laboratorium 2: MLflow

W osobnym terminalu uruchom lokalny serwer (na komputerze studenta dostępny tylko lokalnie):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

W drugim terminalu, z katalogu projektu i aktywnym venv:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m scripts.run_experiments --quick
python -m scripts.compare_experiments
```

Metryki runów są średnimi z 3-fold cross-validation na zbiorze treningowym; zbiór testowy pozostaje do jednorazowej, końcowej oceny pipeline'u.

Wybierz `run_id` z wyniku i zarejestruj wersję z aliasem:

```bash
python -m scripts.registry_aliases RUN_ID --name churn-random-forest --alias candidate
```

UI: `http://127.0.0.1:5000`. Nie wystawiaj niezabezpieczonego serwera MLflow do publicznej sieci.

## Laboratorium 3

```bash
pytest tests/data -q
python -m scripts.validate_data
python scripts/validate_gx.py  # opcjonalna demonstracja Great Expectations
python scripts/check_progress.py
```

Walidację można dodać do DVC przez ręczne przeniesienie etapu z `dvc_lab03_validate.yaml` pod `stages:` w `dvc.yaml`, a następnie dodanie `reports/validation.json` jako zależności etapu `train`. (Plik z ćwiczeniem pokazuje sam etap, nie jest samodzielnym plikiem pipeline'u.)

Przykład offline feature store do uruchomienia z katalogu projektu:

```python
import pandas as pd
from src.features.feature_store import FileFeatureStore

train = pd.read_parquet("data/processed/train.parquet")
features = train[["customer_id", "age", "income", "balance", "num_products"]].copy()
features["income_per_age"] = features["income"] / features["age"]
store = FileFeatureStore()
store.write("customer_features_v1", features)
print(store.read("customer_features_v1", entity_ids=[1, 2]))
```

## Ścieżki / uwagi

- Bazowe zależności przypięte są zakresami zgodnymi z `requirements.txt` kursu; jeśli środowisko ma już venv, nie twórz drugiego.
- `reports/metrics/train_metrics.json` jest metryką DVC bez cache (`cache: false`), dzięki czemu można porównywać wyniki w Git.
- Zbiór jest syntetyczny i służy wyłącznie do nauki; nie interpretuje rzeczywistej retencji klientów.