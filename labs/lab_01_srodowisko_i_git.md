# Laboratorium 1: Środowisko MLOps i Wersjonowanie Kodu

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Podstawowy
- **Wymagania wstępne:** Podstawy Pythona, znajomość terminala

## Cele laboratorium
Po tym laboratorium student:
- skonfiguruje lokalne środowisko MLOps (Python, Git, DVC),
- zorganizuje projekt ML zgodnie z dobrymi praktykami,
- opanuje podstawy wersjonowania kodu i danych,
- skonfiguruje pre-commit hooks.

---

## Część 1: Konfiguracja środowiska (30 min)

### Krok 1.1: Instalacja narzędzi

```bash
# Sprawdź wersję Pythona (wymagane >= 3.10)
python3 --version

# Utwórz wirtualne środowisko
python3 -m venv mlops-env
source mlops-env/bin/activate  # Linux/Mac
# mlops-env\Scripts\activate   # Windows

# Zainstaluj podstawowe narzędzia
pip install --upgrade pip
pip install \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    numpy==1.26.0 \
    matplotlib==3.8.0 \
    mlflow==2.10.0 \
    dvc==3.40.0 \
    great-expectations==0.18.0 \
    pytest==8.0.0 \
    ruff==0.3.0 \
    pre-commit==3.6.0

# Zapisz zależności
pip freeze > requirements.txt
```

### Krok 1.2: Struktura projektu ML

Utwórz strukturę katalogów zgodną z Cookiecutter Data Science:

```bash
mkdir -p mlops-project/{data/{raw,processed,features},\
models,\
notebooks,\
src/{data,features,models,serving},\
tests/{unit,data,model},\
scripts,\
reports/{figures,metrics},\
configs}

cd mlops-project

# Utwórz pliki inicjalizacyjne
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/serving/__init__.py
touch tests/__init__.py
```

Struktura powinna wyglądać tak:

```
mlops-project/
├── data/
│   ├── raw/          # surowe dane (śledzone przez DVC)
│   ├── processed/    # przetworzone dane
│   └── features/     # cechy ML
├── models/           # wytrenowane modele
├── notebooks/        # eksploracja (Jupyter)
├── src/
│   ├── data/         # skrypty przetwarzania danych
│   ├── features/     # feature engineering
│   ├── models/       # trening i ewaluacja
│   └── serving/      # serwowanie modelu
├── tests/            # testy
├── scripts/          # skrypty pomocnicze
├── reports/          # raporty i wykresy
├── configs/          # konfiguracje
├── params.yaml       # parametry modelu
├── dvc.yaml          # pipeline DVC
├── requirements.txt
└── README.md
```

### Krok 1.3: Inicjalizacja Git i DVC

```bash
# Inicjalizacja Git
git init
git config user.name "Twoje Imię"
git config user.email "twoj@email.com"

# Utwórz .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
mlops-env/
*.pyc

# Dane (śledzone przez DVC, nie Git)
data/raw/
data/processed/
data/features/
models/

# MLflow
mlruns/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.idea/
.vscode/
*.swp

# Raporty
reports/figures/
reports/metrics/

# Środowisko
.env
*.env
EOF

# Inicjalizacja DVC
dvc init

# Pierwszy commit
git add .
git commit -m "feat: initial project structure"
```

---

## Część 2: Wersjonowanie danych z DVC (30 min)

### Krok 2.1: Generowanie danych i śledzenie przez DVC

Utwórz plik `scripts/generate_data.py`:

```python
# scripts/generate_data.py
"""Generuje przykładowe dane dla projektu churn prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_churn_dataset(
    n_samples: int = 10_000,
    seed: int = 42,
    output_path: str = "data/raw/customers.parquet"
) -> pd.DataFrame:
    """
    Generuje syntetyczny zbiór danych o churnie klientów.
    
    Args:
        n_samples: liczba próbek
        seed: ziarno losowości
        output_path: ścieżka do zapisu
    
    Returns:
        DataFrame z danymi klientów
    """
    np.random.seed(seed)
    
    # Cechy demograficzne
    age = np.random.normal(40, 12, n_samples).clip(18, 90)
    income = np.random.lognormal(10.8, 0.5, n_samples).clip(15000, 500000)
    
    # Cechy produktowe
    tenure_months = np.random.exponential(24, n_samples).clip(1, 240).astype(int)
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.15, 0.1])
    has_credit_card = np.random.binomial(1, 0.7, n_samples)
    is_active = np.random.binomial(1, 0.8, n_samples)
    
    # Cechy transakcyjne
    balance = np.random.lognormal(9, 1.5, n_samples) * is_active
    num_transactions_30d = np.random.poisson(15, n_samples) * is_active
    
    # Etykieta (churn) - zależna od cech
    churn_prob = (
        0.05
        + 0.15 * (age > 60)
        + 0.20 * (tenure_months < 6)
        + 0.10 * (num_products == 1)
        - 0.10 * (is_active)
        - 0.05 * (has_credit_card)
        + 0.15 * (balance < 1000)
    ).clip(0.02, 0.95)
    
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    df = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': age.round(1),
        'income': income.round(2),
        'tenure_months': tenure_months,
        'num_products': num_products,
        'has_credit_card': has_credit_card,
        'is_active': is_active,
        'balance': balance.round(2),
        'num_transactions_30d': num_transactions_30d,
        'churn': churn
    })
    
    # Zapisz dane
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Wygenerowano {n_samples:,} rekordów")
    print(f"   Churn rate: {churn.mean():.2%}")
    print(f"   Zapisano do: {output_path}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/customers.parquet")
    args = parser.parse_args()
    
    generate_churn_dataset(args.n_samples, args.seed, args.output)
```

```bash
# Wygeneruj dane
python scripts/generate_data.py --n-samples 10000

# Dodaj dane do śledzenia DVC
dvc add data/raw/customers.parquet

# Sprawdź co DVC utworzył
cat data/raw/customers.parquet.dvc

# Commituj plik .dvc (nie same dane!)
git add data/raw/customers.parquet.dvc data/.gitignore
git commit -m "data: add raw customer dataset v1 (10k samples)"
```

### Krok 2.2: Aktualizacja danych i wersjonowanie

```bash
# Wygeneruj nową wersję danych (więcej próbek)
python scripts/generate_data.py --n-samples 50000 --output data/raw/customers.parquet

# Zaktualizuj śledzenie DVC
dvc add data/raw/customers.parquet

# Commituj nową wersję
git add data/raw/customers.parquet.dvc
git commit -m "data: update customer dataset to 50k samples"

# Sprawdź historię
git log --oneline

# Wróć do poprzedniej wersji danych
git checkout HEAD~1 -- data/raw/customers.parquet.dvc
dvc checkout
python -c "import pandas as pd; df = pd.read_parquet('data/raw/customers.parquet'); print(f'Wierszy: {len(df):,}')"

# Wróć do najnowszej wersji
git checkout HEAD -- data/raw/customers.parquet.dvc
dvc checkout
```

---

## Część 3: Pipeline DVC (30 min)

### Krok 3.1: Skrypt przetwarzania danych

Utwórz `src/data/preprocess.py`:

```python
# src/data/preprocess.py
"""Przetwarzanie danych surowych."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def preprocess_data(
    input_path: str,
    output_train: str,
    output_test: str,
    test_size: float = 0.2,
    seed: int = 42
) -> dict:
    """Przetwarza surowe dane i dzieli na train/test."""
    from sklearn.model_selection import train_test_split
    
    print(f"Wczytywanie danych z {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Wczytano {len(df):,} wierszy")
    
    # Czyszczenie
    initial_size = len(df)
    df = df.drop_duplicates(subset=['customer_id'])
    df = df[df['age'].between(18, 90)]
    df = df[df['income'] > 0]
    removed = initial_size - len(df)
    print(f"  Usunięto {removed} błędnych wierszy")
    
    # Feature engineering
    df['income_per_age'] = df['income'] / df['age']
    df['balance_per_product'] = df['balance'] / df['num_products']
    df['is_long_tenure'] = (df['tenure_months'] > 24).astype(int)
    
    # Podział train/test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['churn']
    )
    
    # Zapis
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_train, index=False)
    test_df.to_parquet(output_test, index=False)
    
    stats = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "churn_rate_train": float(train_df['churn'].mean()),
        "churn_rate_test": float(test_df['churn'].mean()),
        "n_features": df.shape[1] - 2  # bez customer_id i churn
    }
    
    print(f"  Train: {stats['train_rows']:,}, Test: {stats['test_rows']:,}")
    print(f"  Churn rate: {stats['churn_rate_train']:.2%}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers.parquet")
    parser.add_argument("--output-train", default="data/processed/train.parquet")
    parser.add_argument("--output-test", default="data/processed/test.parquet")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    
    stats = preprocess_data(args.input, args.output_train, args.output_test, args.test_size)
    
    # Zapisz statystyki
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statystyki zapisane do reports/metrics/data_stats.json")
```

### Krok 3.2: Skrypt treningu

Utwórz `src/models/train.py`:

```python
# src/models/train.py
"""Trening modelu predykcji churnu."""

import pandas as pd
import numpy as np
import joblib
import json
import yaml
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def train_model(
    train_path: str,
    model_path: str,
    params_path: str = "params.yaml"
) -> dict:
    """Trenuje model i zapisuje artefakty."""
    
    # Wczytaj parametry
    with open(params_path) as f:
        params = yaml.safe_load(f)
    
    model_params = params.get('model', {})
    feature_cols = params.get('features', [
        'age', 'income', 'tenure_months', 'num_products',
        'has_credit_card', 'is_active', 'balance',
        'num_transactions_30d', 'income_per_age',
        'balance_per_product', 'is_long_tenure'
    ])
    
    # Wczytaj dane
    train_df = pd.read_parquet(train_path)
    X_train = train_df[feature_cols]
    y_train = train_df['churn']
    
    print(f"Trening na {len(X_train):,} próbkach, {len(feature_cols)} cechach")
    
    # Buduj pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 8),
            min_samples_split=model_params.get('min_samples_split', 10),
            random_state=model_params.get('random_state', 42),
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Ewaluacja na zbiorze treningowym
    y_pred = pipeline.predict(X_train)
    y_prob = pipeline.predict_proba(X_train)[:, 1]
    
    metrics = {
        "train_auc": float(roc_auc_score(y_train, y_prob)),
        "train_f1": float(f1_score(y_train, y_pred)),
        "train_accuracy": float(accuracy_score(y_train, y_pred)),
        "n_estimators": model_params.get('n_estimators', 100),
        "max_depth": model_params.get('max_depth', 8)
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
```

### Krok 3.3: Plik parametrów i pipeline DVC

```yaml
# params.yaml
model:
  n_estimators: 100
  max_depth: 8
  min_samples_split: 10
  random_state: 42

data:
  test_size: 0.2
  seed: 42

features:
  - age
  - income
  - tenure_months
  - num_products
  - has_credit_card
  - is_active
  - balance
  - num_transactions_30d
  - income_per_age
  - balance_per_product
  - is_long_tenure
```

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - src/data/preprocess.py
      - data/raw/customers.parquet
    params:
      - params.yaml:
          - data.test_size
          - data.seed
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
    metrics:
      - reports/metrics/data_stats.json:
          cache: false

  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py
      - data/processed/train.parquet
    params:
      - params.yaml:
          - model
          - features
    outs:
      - models/churn_model.pkl
    metrics:
      - reports/metrics/train_metrics.json:
          cache: false
```

```bash
# Uruchom cały pipeline
dvc repro

# Sprawdź status pipeline'u
dvc status

# Porównaj metryki między wersjami
dvc metrics show
dvc metrics diff HEAD~1
```

---

## Część 4: Pre-commit hooks (30 min)

### Krok 4.1: Konfiguracja pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=5000']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

```bash
# Zainstaluj hooks
pre-commit install

# Przetestuj na wszystkich plikach
pre-commit run --all-files

# Commituj konfigurację
git add .pre-commit-config.yaml params.yaml dvc.yaml src/ scripts/
git commit -m "feat: add DVC pipeline and pre-commit hooks"
```

### Krok 4.2: Testy jednostkowe

Utwórz `tests/unit/test_preprocess.py`:

```python
# tests/unit/test_preprocess.py
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestPreprocessing:
    
    @pytest.fixture
    def raw_data(self, tmp_path):
        """Tworzy przykładowe surowe dane."""
        df = pd.DataFrame({
            'customer_id': range(1, 101),
            'age': np.random.uniform(18, 80, 100),
            'income': np.random.uniform(20000, 100000, 100),
            'tenure_months': np.random.randint(1, 120, 100),
            'num_products': np.random.randint(1, 5, 100),
            'has_credit_card': np.random.randint(0, 2, 100),
            'is_active': np.random.randint(0, 2, 100),
            'balance': np.random.uniform(0, 50000, 100),
            'num_transactions_30d': np.random.randint(0, 50, 100),
            'churn': np.random.randint(0, 2, 100)
        })
        path = tmp_path / "raw.parquet"
        df.to_parquet(path)
        return str(path)
    
    def test_preprocess_creates_output_files(self, raw_data, tmp_path):
        """Test: preprocess tworzy pliki train i test."""
        from src.data.preprocess import preprocess_data
        
        train_path = str(tmp_path / "train.parquet")
        test_path = str(tmp_path / "test.parquet")
        
        preprocess_data(raw_data, train_path, test_path)
        
        assert Path(train_path).exists()
        assert Path(test_path).exists()
    
    def test_preprocess_correct_split_ratio(self, raw_data, tmp_path):
        """Test: podział train/test jest poprawny."""
        from src.data.preprocess import preprocess_data
        
        train_path = str(tmp_path / "train.parquet")
        test_path = str(tmp_path / "test.parquet")
        
        preprocess_data(raw_data, train_path, test_path, test_size=0.2)
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        
        assert abs(test_ratio - 0.2) < 0.05  # tolerancja 5%
    
    def test_preprocess_adds_engineered_features(self, raw_data, tmp_path):
        """Test: preprocess dodaje nowe cechy."""
        from src.data.preprocess import preprocess_data
        
        train_path = str(tmp_path / "train.parquet")
        test_path = str(tmp_path / "test.parquet")
        
        preprocess_data(raw_data, train_path, test_path)
        
        train_df = pd.read_parquet(train_path)
        
        assert 'income_per_age' in train_df.columns
        assert 'balance_per_product' in train_df.columns
        assert 'is_long_tenure' in train_df.columns

# Uruchomienie testów
# pytest tests/unit/test_preprocess.py -v
```

```bash
# Uruchom testy
pytest tests/ -v

# Commituj testy
git add tests/
git commit -m "test: add unit tests for preprocessing"
```

---

## Zadania do samodzielnego wykonania

1. **Rozszerz pipeline DVC** o krok ewaluacji modelu na zbiorze testowym.
2. **Dodaj parametr** `min_samples_leaf` do `params.yaml` i użyj go w treningu.
3. **Napisz test** sprawdzający, że `income_per_age` jest obliczane poprawnie.
4. **Porównaj metryki** między dwoma wersjami modelu (zmień `n_estimators` i uruchom `dvc repro`).

## Pytania kontrolne

1. Jaka jest różnica między `git add` a `dvc add`?
2. Co przechowuje plik `.dvc` i dlaczego jest ważny?
3. Dlaczego nie commitujemy plików danych bezpośrednio do Git?
4. Co robi `dvc repro` i kiedy pomija kroki?

## Podsumowanie

W tym laboratorium:
- ✅ Skonfigurowałeś środowisko MLOps z Git i DVC
- ✅ Zorganizowałeś projekt zgodnie z dobrymi praktykami
- ✅ Wersjonujesz dane z DVC
- ✅ Zdefiniowałeś reprodukowalny pipeline DVC
- ✅ Skonfigurowałeś pre-commit hooks i testy
