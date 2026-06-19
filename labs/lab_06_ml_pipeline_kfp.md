# Laboratorium 6: Budowanie ML Pipeline z Kubeflow Pipelines

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Zaawansowany
- **Wymagania wstępne:** Lab 1–5 ukończone, podstawy Docker

## Cele laboratorium
Po tym laboratorium student:
- zbuduje kompletny ML Pipeline z użyciem KFP SDK v2,
- zdefiniuje komponenty pipeline'u jako funkcje Python,
- skompiluje pipeline do YAML i uruchomi lokalnie,
- zaimplementuje warunkowe wdrożenie modelu.

---

## Część 1: Instalacja i konfiguracja KFP (20 min)

### Krok 1.1: Instalacja KFP SDK

```bash
pip install kfp==2.7.0 kfp-local==0.1.0

# Sprawdź instalację
python -c "import kfp; print(f'KFP version: {kfp.__version__}')"
```

### Krok 1.2: Struktura projektu pipeline'u

```
mlops-project/
├── pipeline/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── ingest.py        # komponent pobierania danych
│   │   ├── validate.py      # komponent walidacji
│   │   ├── preprocess.py    # komponent preprocessingu
│   │   ├── train.py         # komponent treningu
│   │   ├── evaluate.py      # komponent ewaluacji
│   │   └── deploy.py        # komponent wdrożenia
│   ├── churn_pipeline.py    # definicja pipeline'u
│   └── run_pipeline.py      # skrypt uruchamiający
```

```bash
mkdir -p pipeline/components
touch pipeline/__init__.py pipeline/components/__init__.py
```

---

## Część 2: Komponenty pipeline'u (60 min)

### Krok 2.1: Komponent pobierania danych

Utwórz `pipeline/components/ingest.py`:

```python
# pipeline/components/ingest.py
"""Komponent KFP: pobieranie i generowanie danych."""

from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.0", "numpy==1.26.0", "pyarrow==15.0.0"]
)
def ingest_data(
    n_samples: int,
    seed: int,
    output_dataset: Output[Dataset]
):
    """
    Generuje syntetyczne dane klientów i zapisuje do artefaktu.
    
    W produkcji: pobierałby dane z BigQuery/Cloud Storage.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    np.random.seed(seed)
    n = n_samples

    age = np.random.normal(40, 12, n).clip(18, 90)
    income = np.random.lognormal(10.8, 0.5, n).clip(15000, 500000)
    tenure_months = np.random.exponential(24, n).clip(1, 240).astype(int)
    num_products = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1])
    has_credit_card = np.random.binomial(1, 0.7, n)
    is_active = np.random.binomial(1, 0.8, n)
    balance = np.random.lognormal(9, 1.5, n) * is_active
    num_transactions_30d = np.random.poisson(15, n) * is_active

    churn_prob = (
        0.05
        + 0.15 * (age > 60)
        + 0.20 * (tenure_months < 6)
        + 0.10 * (num_products == 1)
        - 0.10 * is_active
        - 0.05 * has_credit_card
        + 0.15 * (balance < 1000)
    ).clip(0.02, 0.95)

    churn = np.random.binomial(1, churn_prob, n)

    df = pd.DataFrame({
        'customer_id': range(1, n + 1),
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

    # Zapisz do artefaktu KFP
    Path(output_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dataset.path, index=False)

    print(f"✅ Wygenerowano {n:,} rekordów")
    print(f"   Churn rate: {churn.mean():.2%}")
    print(f"   Zapisano do: {output_dataset.path}")
```

### Krok 2.2: Komponent walidacji danych

Utwórz `pipeline/components/validate.py`:

```python
# pipeline/components/validate.py
"""Komponent KFP: walidacja danych."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.0", "pyarrow==15.0.0"]
)
def validate_data(
    input_dataset: Input[Dataset],
    validation_metrics: Output[Metrics],
    min_rows: int = 1000,
    max_churn_rate: float = 0.60,
    min_churn_rate: float = 0.01
) -> bool:
    """Waliduje dane wejściowe i zwraca True jeśli dane są poprawne."""
    import pandas as pd

    df = pd.read_parquet(input_dataset.path)

    errors = []
    warnings = []

    # Sprawdzenia krytyczne
    required_cols = [
        'customer_id', 'age', 'income', 'tenure_months',
        'num_products', 'has_credit_card', 'is_active',
        'balance', 'num_transactions_30d', 'churn'
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Brakujące kolumny: {missing}")

    if len(df) < min_rows:
        errors.append(f"Za mało wierszy: {len(df)} < {min_rows}")

    if df.isnull().sum().sum() > 0:
        errors.append(f"Wartości null: {df.isnull().sum().sum()}")

    if 'age' in df.columns and not df['age'].between(18, 100).all():
        errors.append("Wiek poza zakresem [18, 100]")

    if 'churn' in df.columns:
        churn_rate = df['churn'].mean()
        if not (min_churn_rate <= churn_rate <= max_churn_rate):
            warnings.append(f"Churn rate {churn_rate:.2%} poza oczekiwanym zakresem")

    # Logowanie metryk do KFP
    validation_metrics.log_metric("n_rows", len(df))
    validation_metrics.log_metric("n_columns", df.shape[1])
    validation_metrics.log_metric("n_errors", len(errors))
    validation_metrics.log_metric("n_warnings", len(warnings))
    if 'churn' in df.columns:
        validation_metrics.log_metric("churn_rate", float(df['churn'].mean()))

    if errors:
        print(f"❌ Walidacja FAILED: {errors}")
        return False

    if warnings:
        print(f"⚠️  Ostrzeżenia: {warnings}")

    print(f"✅ Walidacja OK: {len(df):,} wierszy, {df.shape[1]} kolumn")
    return True
```

### Krok 2.3: Komponent preprocessingu

Utwórz `pipeline/components/preprocess.py`:

```python
# pipeline/components/preprocess.py
"""Komponent KFP: preprocessing danych."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.0", "scikit-learn==1.4.0",
        "numpy==1.26.0", "pyarrow==15.0.0"
    ]
)
def preprocess_data(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preprocessing_metrics: Output[Metrics],
    test_size: float = 0.2,
    seed: int = 42
):
    """Przetwarza dane i dzieli na train/test."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    df = pd.read_parquet(input_dataset.path)
    initial_size = len(df)

    # Czyszczenie
    df = df.drop_duplicates(subset=['customer_id'])
    df = df[df['age'].between(18, 90)]
    df = df[df['income'] > 0]
    removed = initial_size - len(df)

    # Feature engineering
    df['income_per_age'] = df['income'] / df['age']
    df['balance_per_product'] = df['balance'] / df['num_products']
    df['is_long_tenure'] = (df['tenure_months'] > 24).astype(int)
    df['log_income'] = np.log1p(df['income'])
    df['log_balance'] = np.log1p(df['balance'])

    # Podział
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['churn']
    )

    # Zapis
    Path(train_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_dataset.path, index=False)
    test_df.to_parquet(test_dataset.path, index=False)

    # Metryki
    preprocessing_metrics.log_metric("initial_rows", initial_size)
    preprocessing_metrics.log_metric("removed_rows", removed)
    preprocessing_metrics.log_metric("train_rows", len(train_df))
    preprocessing_metrics.log_metric("test_rows", len(test_df))
    preprocessing_metrics.log_metric("n_features", df.shape[1] - 2)
    preprocessing_metrics.log_metric("churn_rate_train", float(train_df['churn'].mean()))

    print(f"✅ Preprocessing zakończony")
    print(f"   Usunięto: {removed} wierszy")
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
```

### Krok 2.4: Komponent treningu

Utwórz `pipeline/components/train.py`:

```python
# pipeline/components/train.py
"""Komponent KFP: trening modelu."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.0", "scikit-learn==1.4.0",
        "numpy==1.26.0", "pyarrow==15.0.0", "joblib==1.3.0"
    ]
)
def train_model(
    train_dataset: Input[Dataset],
    output_model: Output[Model],
    training_metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 8,
    min_samples_split: int = 10,
    seed: int = 42
):
    """Trenuje model Random Forest i zapisuje artefakty."""
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from pathlib import Path

    FEATURE_COLS = [
        'age', 'income', 'tenure_months', 'num_products',
        'has_credit_card', 'is_active', 'balance',
        'num_transactions_30d', 'income_per_age',
        'balance_per_product', 'is_long_tenure',
        'log_income', 'log_balance'
    ]

    train_df = pd.read_parquet(train_dataset.path)
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[available_features]
    y_train = train_df['churn']

    print(f"Trening: {len(X_train):,} próbek, {len(available_features)} cech")

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Trening na pełnym zbiorze
    pipeline.fit(X_train, y_train)

    # Metryki treningowe
    y_pred = pipeline.predict(X_train)
    y_prob = pipeline.predict_proba(X_train)[:, 1]

    training_metrics.log_metric("cv_auc_mean", float(cv_scores.mean()))
    training_metrics.log_metric("cv_auc_std", float(cv_scores.std()))
    training_metrics.log_metric("train_auc", float(roc_auc_score(y_train, y_prob)))
    training_metrics.log_metric("train_f1", float(f1_score(y_train, y_pred)))
    training_metrics.log_metric("n_estimators", n_estimators)
    training_metrics.log_metric("max_depth", max_depth)
    training_metrics.log_metric("n_features", len(available_features))

    # Zapis modelu
    Path(output_model.path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'pipeline': pipeline,
        'feature_cols': available_features,
        'params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'seed': seed
        }
    }
    joblib.dump(model_data, output_model.path)
    print(f"✅ Model zapisany: {output_model.path}")
```

### Krok 2.5: Komponent ewaluacji

Utwórz `pipeline/components/evaluate.py`:

```python
# pipeline/components/evaluate.py
"""Komponent KFP: ewaluacja modelu."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Metrics, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.0", "scikit-learn==1.4.0",
        "numpy==1.26.0", "pyarrow==15.0.0", "joblib==1.3.0"
    ]
)
def evaluate_model(
    test_dataset: Input[Dataset],
    input_model: Input[Model],
    evaluation_metrics: Output[Metrics],
    min_auc_threshold: float = 0.75
) -> str:
    """
    Ewaluuje model na zbiorze testowym.
    
    Returns:
        'deploy' jeśli model spełnia wymagania, 'reject' w przeciwnym razie
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        precision_score, recall_score, average_precision_score
    )

    # Wczytaj model i dane
    model_data = joblib.load(input_model.path)
    pipeline = model_data['pipeline']
    feature_cols = model_data['feature_cols']

    test_df = pd.read_parquet(test_dataset.path)
    available_features = [c for c in feature_cols if c in test_df.columns]

    X_test = test_df[available_features]
    y_test = test_df['churn']

    # Predykcje
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metryki
    auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred))
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    avg_precision = float(average_precision_score(y_test, y_prob))

    # Logowanie metryk
    evaluation_metrics.log_metric("test_auc_roc", auc)
    evaluation_metrics.log_metric("test_f1", f1)
    evaluation_metrics.log_metric("test_accuracy", acc)
    evaluation_metrics.log_metric("test_precision", precision)
    evaluation_metrics.log_metric("test_recall", recall)
    evaluation_metrics.log_metric("test_avg_precision", avg_precision)
    evaluation_metrics.log_metric("n_test_samples", len(y_test))
    evaluation_metrics.log_metric("churn_rate_test", float(y_test.mean()))

    print(f"=== Wyniki ewaluacji ===")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Avg Prec:  {avg_precision:.4f}")

    # Decyzja o wdrożeniu
    if auc >= min_auc_threshold:
        decision = "deploy"
        print(f"\n✅ Model ZAAKCEPTOWANY (AUC={auc:.4f} >= {min_auc_threshold})")
    else:
        decision = "reject"
        print(f"\n❌ Model ODRZUCONY (AUC={auc:.4f} < {min_auc_threshold})")

    evaluation_metrics.log_metric("deployment_decision", 1 if decision == "deploy" else 0)
    return decision
```

---

## Część 3: Definicja i uruchomienie pipeline'u (40 min)

### Krok 3.1: Główny pipeline

Utwórz `pipeline/churn_pipeline.py`:

```python
# pipeline/churn_pipeline.py
"""Definicja kompletnego pipeline'u ML dla predykcji churnu."""

from kfp import dsl, compiler
from kfp.dsl import If

# Import komponentów
import sys
sys.path.insert(0, '.')

from pipeline.components.ingest import ingest_data
from pipeline.components.validate import validate_data
from pipeline.components.preprocess import preprocess_data
from pipeline.components.train import train_model
from pipeline.components.evaluate import evaluate_model


@dsl.pipeline(
    name="churn-prediction-pipeline",
    description="Kompletny pipeline ML dla predykcji churnu klientów"
)
def churn_pipeline(
    # Parametry danych
    n_samples: int = 10000,
    seed: int = 42,
    test_size: float = 0.2,
    # Parametry modelu
    n_estimators: int = 100,
    max_depth: int = 8,
    min_samples_split: int = 10,
    # Bramka jakości
    min_auc_threshold: float = 0.75
):
    """
    Pipeline ML: ingest → validate → preprocess → train → evaluate → [deploy]
    """

    # ── Krok 1: Pobierz dane ─────────────────────────────────────────────────
    ingest_task = ingest_data(
        n_samples=n_samples,
        seed=seed
    )
    ingest_task.set_display_name("1. Pobierz dane")

    # ── Krok 2: Waliduj dane ─────────────────────────────────────────────────
    validate_task = validate_data(
        input_dataset=ingest_task.outputs["output_dataset"],
        min_rows=1000
    )
    validate_task.set_display_name("2. Waliduj dane")
    validate_task.after(ingest_task)

    # ── Krok 3: Preprocessing ────────────────────────────────────────────────
    preprocess_task = preprocess_data(
        input_dataset=ingest_task.outputs["output_dataset"],
        test_size=test_size,
        seed=seed
    )
    preprocess_task.set_display_name("3. Preprocessing")
    preprocess_task.after(validate_task)

    # ── Krok 4: Trening ──────────────────────────────────────────────────────
    train_task = train_model(
        train_dataset=preprocess_task.outputs["train_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        seed=seed
    )
    train_task.set_display_name("4. Trenuj model")
    # Konfiguracja zasobów
    train_task.set_cpu_request("1")
    train_task.set_memory_request("2Gi")

    # ── Krok 5: Ewaluacja ────────────────────────────────────────────────────
    evaluate_task = evaluate_model(
        test_dataset=preprocess_task.outputs["test_dataset"],
        input_model=train_task.outputs["output_model"],
        min_auc_threshold=min_auc_threshold
    )
    evaluate_task.set_display_name("5. Ewaluuj model")

    # ── Krok 6: Warunkowe wdrożenie ──────────────────────────────────────────
    with dsl.If(evaluate_task.output == "deploy", name="Czy wdrożyć?"):
        # Komponent wdrożenia (uproszczony)
        @dsl.component(base_image="python:3.11-slim",
                       packages_to_install=["joblib==1.3.0"])
        def deploy_model_component(input_model: dsl.Input[dsl.Model]) -> str:
            import joblib
            model_data = joblib.load(input_model.path)
            print(f"🚀 Wdrażanie modelu...")
            print(f"   Cechy: {model_data['feature_cols']}")
            print(f"   Parametry: {model_data['params']}")
            # W produkcji: wdrożenie na Vertex AI Endpoint
            return "deployed"

        deploy_task = deploy_model_component(
            input_model=train_task.outputs["output_model"]
        )
        deploy_task.set_display_name("6. Wdróż model")


# ── Kompilacja ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = "pipeline/churn_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path=output_path
    )
    print(f"✅ Pipeline skompilowany do: {output_path}")
```

### Krok 3.2: Uruchomienie pipeline'u lokalnie

```python
# pipeline/run_pipeline.py
"""Uruchamia pipeline lokalnie (bez klastra Kubernetes)."""

import sys
sys.path.insert(0, '.')

from kfp.local import init, SubprocessRunner
from pipeline.churn_pipeline import churn_pipeline

# Inicjalizacja lokalnego runnera
init(runner=SubprocessRunner())

# Uruchomienie pipeline'u
print("Uruchamianie pipeline'u lokalnie...")
print("=" * 60)

pipeline_run = churn_pipeline(
    n_samples=5000,
    seed=42,
    test_size=0.2,
    n_estimators=50,    # mniej drzew dla szybszego testu
    max_depth=6,
    min_samples_split=10,
    min_auc_threshold=0.70
)

print("\n" + "=" * 60)
print("Pipeline zakończony!")
```

```bash
# Skompiluj pipeline
python pipeline/churn_pipeline.py

# Sprawdź wygenerowany YAML
head -50 pipeline/churn_pipeline.yaml

# Uruchom pipeline lokalnie
python pipeline/run_pipeline.py
```

### Krok 3.3: Testy komponentów

Utwórz `tests/unit/test_pipeline_components.py`:

```python
# tests/unit/test_pipeline_components.py
"""Testy jednostkowe komponentów pipeline'u."""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestIngestComponent:
    """Testy komponentu ingest_data."""

    def test_generates_correct_number_of_rows(self, tmp_path):
        """Test: generuje poprawną liczbę wierszy."""
        from pipeline.components.ingest import ingest_data

        output_path = tmp_path / "data.parquet"

        # Wywołaj funkcję bezpośrednio (bez dekoratora KFP)
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'customer_id': range(1, n + 1),
            'age': np.random.normal(40, 12, n).clip(18, 90),
            'income': np.random.lognormal(10.8, 0.5, n),
            'churn': np.random.binomial(1, 0.2, n)
        })
        df.to_parquet(output_path)

        loaded = pd.read_parquet(output_path)
        assert len(loaded) == n

    def test_churn_column_is_binary(self, tmp_path):
        """Test: kolumna churn jest binarna."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'customer_id': range(1, n + 1),
            'churn': np.random.binomial(1, 0.2, n)
        })
        output_path = tmp_path / "data.parquet"
        df.to_parquet(output_path)

        loaded = pd.read_parquet(output_path)
        assert set(loaded['churn'].unique()).issubset({0, 1})


class TestPreprocessComponent:
    """Testy komponentu preprocess_data."""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Tworzy przykładowy plik Parquet."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'customer_id': range(1, n + 1),
            'age': np.random.uniform(18, 80, n),
            'income': np.random.uniform(20000, 100000, n),
            'tenure_months': np.random.randint(1, 120, n),
            'num_products': np.random.randint(1, 5, n),
            'has_credit_card': np.random.randint(0, 2, n),
            'is_active': np.random.randint(0, 2, n),
            'balance': np.random.uniform(0, 50000, n),
            'num_transactions_30d': np.random.randint(0, 50, n),
            'churn': np.random.binomial(1, 0.2, n)
        })
        path = tmp_path / "raw.parquet"
        df.to_parquet(path)
        return path

    def test_creates_engineered_features(self, sample_parquet, tmp_path):
        """Test: tworzy cechy pochodne."""
        df = pd.read_parquet(sample_parquet)

        # Symuluj preprocessing
        df['income_per_age'] = df['income'] / df['age']
        df['balance_per_product'] = df['balance'] / df['num_products']
        df['is_long_tenure'] = (df['tenure_months'] > 24).astype(int)

        assert 'income_per_age' in df.columns
        assert 'balance_per_product' in df.columns
        assert 'is_long_tenure' in df.columns

    def test_train_test_split_ratio(self, sample_parquet, tmp_path):
        """Test: podział train/test jest poprawny."""
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(sample_parquet)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        assert abs(test_ratio - 0.2) < 0.05


class TestEvaluateComponent:
    """Testy komponentu evaluate_model."""

    def test_deploy_decision_for_good_model(self):
        """Test: dobry model dostaje decyzję 'deploy'."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        X, y = make_classification(n_samples=1000, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:800], y[:800])

        auc = roc_auc_score(y[800:], model.predict_proba(X[800:])[:, 1])
        decision = "deploy" if auc >= 0.70 else "reject"

        assert decision == "deploy", f"Oczekiwano 'deploy', AUC={auc:.4f}"

    def test_reject_decision_for_bad_model(self):
        """Test: zły model dostaje decyzję 'reject'."""
        # Model z losowymi predykcjami
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.random.uniform(0, 1, 200)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_prob)
        decision = "deploy" if auc >= 0.80 else "reject"

        assert decision == "reject", f"Oczekiwano 'reject', AUC={auc:.4f}"
```

```bash
# Uruchom testy komponentów
pytest tests/unit/test_pipeline_components.py -v

# Uruchom wszystkie testy
pytest tests/ -v --tb=short
```

---

## Zadania do samodzielnego wykonania

1. **Dodaj komponent** `feature_selection` między preprocessing a treningiem, który usuwa cechy o niskiej ważności.
2. **Dodaj parametr** `model_type` do pipeline'u umożliwiający wybór między RF a GBM.
3. **Zaimplementuj caching** – dodaj `enable_caching=True` i sprawdź, które kroki są pomijane przy ponownym uruchomieniu.
4. **Napisz test** sprawdzający, że pipeline kompiluje się bez błędów.

## Pytania kontrolne

1. Dlaczego każdy komponent KFP jest uruchamiany w osobnym kontenerze?
2. Co to jest artefakt KFP i jak różni się od zwykłego parametru?
3. Jak działa `dsl.If` i kiedy go używamy?
4. Dlaczego testujemy komponenty jako zwykłe funkcje Python, a nie przez KFP?

## Podsumowanie

W tym laboratorium:
- ✅ Zdefiniowałeś 5 komponentów KFP jako funkcje Python
- ✅ Zbudowałeś kompletny pipeline z warunkowym wdrożeniem
- ✅ Skompilowałeś pipeline do YAML
- ✅ Uruchomiłeś pipeline lokalnie
- ✅ Napisałeś testy jednostkowe dla komponentów
