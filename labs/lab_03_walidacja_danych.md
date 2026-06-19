# Laboratorium 3: Walidacja Danych i Feature Store

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Średni
- **Wymagania wstępne:** Lab 1 i 2 ukończone

## Cele laboratorium
Po tym laboratorium student:
- napisze testy jakości danych z użyciem pytest i własnych reguł,
- zbuduje prosty Feature Store oparty na plikach Parquet,
- wykryje anomalie w danych wejściowych,
- zintegruje walidację danych z pipeline'em DVC.

---

## Część 1: Testy jakości danych (40 min)

### Krok 1.1: Framework walidacji danych

Utwórz `src/data/validation.py`:

```python
# src/data/validation.py
"""Framework walidacji danych dla projektu churn prediction."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime


@dataclass
class ValidationRule:
    """Pojedyncza reguła walidacji."""
    name: str
    description: str
    check_fn: Callable[[pd.DataFrame], bool]
    severity: str = "error"   # error | warning
    column: str = None        # None = reguła dla całego DataFrame


@dataclass
class ValidationResult:
    """Wynik walidacji jednej reguły."""
    rule_name: str
    passed: bool
    severity: str
    message: str = ""
    affected_rows: int = 0


@dataclass
class ValidationReport:
    """Pełny raport walidacji."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def errors(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines = [
            f"=== Raport Walidacji [{self.timestamp}] ===",
            f"Wynik: {'✅ PASS' if self.passed else '❌ FAIL'}",
            f"Reguły: {passed}/{total} przeszły",
        ]
        if self.errors:
            lines.append("\nBłędy krytyczne:")
            for e in self.errors:
                lines.append(f"  ❌ {e.rule_name}: {e.message}")
        if self.warnings:
            lines.append("\nOstrzeżenia:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w.rule_name}: {w.message}")
        return "\n".join(lines)


class DataValidator:
    """Walidator danych z zestawem reguł."""

    def __init__(self):
        self.rules: list[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> "DataValidator":
        self.rules.append(rule)
        return self  # umożliwia chaining

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Uruchamia wszystkie reguły i zwraca raport."""
        report = ValidationReport()

        for rule in self.rules:
            try:
                passed = rule.check_fn(df)
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=passed,
                    severity=rule.severity,
                    message="" if passed else rule.description,
                )
            except Exception as e:
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=rule.severity,
                    message=f"Błąd wykonania reguły: {e}",
                )
            report.results.append(result)

        return report


def build_churn_validator() -> DataValidator:
    """Buduje walidator dla danych churn prediction."""
    validator = DataValidator()

    # --- Reguły schematu ---
    required_columns = [
        'customer_id', 'age', 'income', 'tenure_months',
        'num_products', 'has_credit_card', 'is_active',
        'balance', 'num_transactions_30d', 'churn'
    ]

    validator.add_rule(ValidationRule(
        name="required_columns",
        description=f"Brakujące kolumny: {required_columns}",
        check_fn=lambda df: all(c in df.columns for c in required_columns),
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="no_missing_values",
        description="Dane zawierają wartości null",
        check_fn=lambda df: df.isnull().sum().sum() == 0,
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="no_duplicate_customers",
        description="Zduplikowane customer_id",
        check_fn=lambda df: not df['customer_id'].duplicated().any()
            if 'customer_id' in df.columns else True,
        severity="error"
    ))

    # --- Reguły zakresów ---
    validator.add_rule(ValidationRule(
        name="age_range",
        description="Wiek poza zakresem [18, 100]",
        check_fn=lambda df: df['age'].between(18, 100).all()
            if 'age' in df.columns else True,
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="income_positive",
        description="Ujemny dochód",
        check_fn=lambda df: (df['income'] >= 0).all()
            if 'income' in df.columns else True,
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="churn_binary",
        description="Etykieta churn nie jest binarna (0/1)",
        check_fn=lambda df: set(df['churn'].unique()).issubset({0, 1})
            if 'churn' in df.columns else True,
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="num_products_range",
        description="Liczba produktów poza zakresem [1, 10]",
        check_fn=lambda df: df['num_products'].between(1, 10).all()
            if 'num_products' in df.columns else True,
        severity="warning"
    ))

    # --- Reguły statystyczne ---
    validator.add_rule(ValidationRule(
        name="churn_rate_reasonable",
        description="Churn rate poza zakresem [1%, 60%]",
        check_fn=lambda df: 0.01 <= df['churn'].mean() <= 0.60
            if 'churn' in df.columns else True,
        severity="warning"
    ))

    validator.add_rule(ValidationRule(
        name="min_sample_size",
        description="Za mało próbek (< 100)",
        check_fn=lambda df: len(df) >= 100,
        severity="error"
    ))

    validator.add_rule(ValidationRule(
        name="balance_non_negative",
        description="Ujemne saldo konta",
        check_fn=lambda df: (df['balance'] >= 0).all()
            if 'balance' in df.columns else True,
        severity="warning"
    ))

    return validator
```

### Krok 1.2: Testy walidacji

Utwórz `tests/data/test_validation.py`:

```python
# tests/data/test_validation.py
"""Testy walidacji danych."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.validation import build_churn_validator, DataValidator, ValidationRule


@pytest.fixture
def valid_df():
    """Poprawny DataFrame spełniający wszystkie reguły."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
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


class TestDataValidator:

    def test_valid_data_passes_all_rules(self, valid_df):
        """Test: poprawne dane przechodzą wszystkie reguły."""
        validator = build_churn_validator()
        report = validator.validate(valid_df)
        assert report.passed, f"Błędy: {[e.rule_name for e in report.errors]}"

    def test_missing_column_fails(self, valid_df):
        """Test: brakująca kolumna powoduje błąd."""
        df_missing = valid_df.drop(columns=['age'])
        validator = build_churn_validator()
        report = validator.validate(df_missing)
        assert not report.passed
        assert any(r.rule_name == "required_columns" for r in report.errors)

    def test_null_values_fail(self, valid_df):
        """Test: wartości null powodują błąd."""
        df_with_nulls = valid_df.copy()
        df_with_nulls.loc[0, 'age'] = None
        validator = build_churn_validator()
        report = validator.validate(df_with_nulls)
        assert not report.passed

    def test_invalid_age_fails(self, valid_df):
        """Test: wiek poza zakresem powoduje błąd."""
        df_bad_age = valid_df.copy()
        df_bad_age.loc[0, 'age'] = -5
        validator = build_churn_validator()
        report = validator.validate(df_bad_age)
        assert not report.passed
        assert any(r.rule_name == "age_range" for r in report.errors)

    def test_non_binary_churn_fails(self, valid_df):
        """Test: niebinarna etykieta powoduje błąd."""
        df_bad_churn = valid_df.copy()
        df_bad_churn.loc[0, 'churn'] = 2
        validator = build_churn_validator()
        report = validator.validate(df_bad_churn)
        assert not report.passed

    def test_duplicate_customers_fail(self, valid_df):
        """Test: zduplikowane customer_id powodują błąd."""
        df_dupes = pd.concat([valid_df, valid_df.head(5)], ignore_index=True)
        validator = build_churn_validator()
        report = validator.validate(df_dupes)
        assert not report.passed

    def test_report_summary_format(self, valid_df):
        """Test: raport ma poprawny format."""
        validator = build_churn_validator()
        report = validator.validate(valid_df)
        summary = report.summary()
        assert "Raport Walidacji" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_custom_rule_chaining(self):
        """Test: dodawanie reguł przez chaining."""
        validator = (DataValidator()
            .add_rule(ValidationRule(
                name="min_rows",
                description="Za mało wierszy",
                check_fn=lambda df: len(df) >= 10
            ))
            .add_rule(ValidationRule(
                name="has_id",
                description="Brak kolumny id",
                check_fn=lambda df: 'id' in df.columns
            )))

        df = pd.DataFrame({'id': range(20), 'value': range(20)})
        report = validator.validate(df)
        assert report.passed
```

```bash
# Uruchom testy walidacji
pytest tests/data/test_validation.py -v

# Uruchom walidację na danych
python -c "
import pandas as pd
import sys
sys.path.insert(0, '.')
from src.data.validation import build_churn_validator

df = pd.read_parquet('data/processed/train.parquet')
validator = build_churn_validator()
report = validator.validate(df)
print(report.summary())
"
```

---

## Część 2: Wykrywanie anomalii (30 min)

### Krok 2.1: Detektor anomalii statystycznych

Utwórz `src/data/anomaly_detection.py`:

```python
# src/data/anomaly_detection.py
"""Wykrywanie anomalii w danych wejściowych."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class AnomalyReport:
    """Raport anomalii dla jednej cechy."""
    feature: str
    n_anomalies: int
    anomaly_pct: float
    method: str
    threshold: float
    anomaly_indices: list[int]

    def __str__(self):
        return (f"{self.feature}: {self.n_anomalies} anomalii "
                f"({self.anomaly_pct:.1%}) metodą {self.method}")


class StatisticalAnomalyDetector:
    """Wykrywa anomalie metodami statystycznymi."""

    def __init__(self, z_score_threshold: float = 3.5, iqr_multiplier: float = 3.0):
        self.z_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self._reference_stats: dict = {}

    def fit(self, df: pd.DataFrame, numeric_cols: list[str]):
        """Uczy się statystyk referencyjnych z danych treningowych."""
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                self._reference_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'lower_iqr': float(Q1 - self.iqr_multiplier * IQR),
                    'upper_iqr': float(Q3 + self.iqr_multiplier * IQR),
                }
        return self

    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        """Wykrywa anomalie w nowych danych."""
        reports = []

        for col, stats_ref in self._reference_stats.items():
            if col not in df.columns:
                continue

            values = df[col].dropna()

            # Metoda IQR
            iqr_mask = ~values.between(stats_ref['lower_iqr'], stats_ref['upper_iqr'])
            n_iqr = iqr_mask.sum()

            if n_iqr > 0:
                reports.append(AnomalyReport(
                    feature=col,
                    n_anomalies=int(n_iqr),
                    anomaly_pct=float(n_iqr / len(values)),
                    method="IQR",
                    threshold=self.iqr_multiplier,
                    anomaly_indices=values[iqr_mask].index.tolist()
                ))

            # Metoda Z-score (na podstawie statystyk referencyjnych)
            z_scores = np.abs((values - stats_ref['mean']) / max(stats_ref['std'], 1e-6))
            z_mask = z_scores > self.z_threshold
            n_z = z_mask.sum()

            if n_z > 0:
                reports.append(AnomalyReport(
                    feature=col,
                    n_anomalies=int(n_z),
                    anomaly_pct=float(n_z / len(values)),
                    method="Z-score",
                    threshold=self.z_threshold,
                    anomaly_indices=values[z_mask].index.tolist()
                ))

        return reports

    def get_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zwraca dane bez anomalii (IQR method)."""
        mask = pd.Series(True, index=df.index)

        for col, stats_ref in self._reference_stats.items():
            if col in df.columns:
                col_mask = df[col].between(stats_ref['lower_iqr'], stats_ref['upper_iqr'])
                mask = mask & col_mask

        removed = (~mask).sum()
        print(f"Usunięto {removed} wierszy z anomaliami ({removed/len(df):.1%})")
        return df[mask].copy()


# Przykład użycia
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    # Wczytaj dane treningowe
    train_df = pd.read_parquet("data/processed/train.parquet")

    numeric_cols = ['age', 'income', 'balance', 'num_transactions_30d']

    # Naucz detektor na danych treningowych
    detector = StatisticalAnomalyDetector(z_score_threshold=3.5, iqr_multiplier=3.0)
    detector.fit(train_df, numeric_cols)

    # Symuluj dane produkcyjne z anomaliami
    import numpy as np
    np.random.seed(42)
    prod_df = train_df.sample(1000).copy()
    # Wprowadź sztuczne anomalie
    prod_df.loc[prod_df.index[:10], 'age'] = 999
    prod_df.loc[prod_df.index[10:20], 'income'] = -1000

    # Wykryj anomalie
    reports = detector.detect(prod_df)

    print("=== Raport Anomalii ===")
    for report in reports:
        print(f"  {report}")

    # Oczyść dane
    clean_df = detector.get_clean_data(prod_df)
    print(f"\nDane po czyszczeniu: {len(clean_df)} wierszy (było {len(prod_df)})")
```

---

## Część 3: Prosty Feature Store (30 min)

### Krok 3.1: Implementacja Feature Store

Utwórz `src/features/feature_store.py`:

```python
# src/features/feature_store.py
"""Prosty Feature Store oparty na plikach Parquet."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class FeatureDefinition:
    """Definicja cechy w Feature Store."""
    name: str
    dtype: str
    description: str
    source_column: str
    transformation: str = "identity"  # identity | log | normalize | bin
    version: str = "1.0"


class SimpleFeatureStore:
    """
    Prosty Feature Store oparty na plikach Parquet.
    
    Struktura katalogów:
        feature_store/
            registry.json          # rejestr cech
            features/
                user_stats/
                    v1/data.parquet
                    v2/data.parquet
    """

    def __init__(self, store_path: str = "feature_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        (self.store_path / "features").mkdir(exist_ok=True)
        self._registry_path = self.store_path / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {"feature_groups": {}}

    def _save_registry(self):
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_feature_group(
        self,
        group_name: str,
        features: list[FeatureDefinition],
        entity_key: str = "customer_id"
    ):
        """Rejestruje grupę cech."""
        self._registry["feature_groups"][group_name] = {
            "entity_key": entity_key,
            "features": [asdict(f) for f in features],
            "created_at": datetime.now().isoformat(),
            "versions": []
        }
        self._save_registry()
        print(f"✅ Zarejestrowano grupę cech: {group_name} ({len(features)} cech)")

    def write_features(
        self,
        group_name: str,
        df: pd.DataFrame,
        version: str = None
    ) -> str:
        """Zapisuje cechy do Feature Store."""
        if group_name not in self._registry["feature_groups"]:
            raise ValueError(f"Nieznana grupa cech: {group_name}")

        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # Zapisz dane
        group_path = self.store_path / "features" / group_name / version
        group_path.mkdir(parents=True, exist_ok=True)
        data_path = group_path / "data.parquet"
        df.to_parquet(data_path, index=False)

        # Aktualizuj rejestr
        self._registry["feature_groups"][group_name]["versions"].append({
            "version": version,
            "path": str(data_path),
            "n_rows": len(df),
            "created_at": datetime.now().isoformat()
        })
        self._save_registry()

        print(f"✅ Zapisano {len(df):,} wierszy do {group_name}/{version}")
        return version

    def read_features(
        self,
        group_name: str,
        entity_ids: list,
        feature_names: list[str] = None,
        version: str = "latest"
    ) -> pd.DataFrame:
        """Pobiera cechy dla podanych encji."""
        if group_name not in self._registry["feature_groups"]:
            raise ValueError(f"Nieznana grupa cech: {group_name}")

        group_info = self._registry["feature_groups"][group_name]
        versions = group_info["versions"]

        if not versions:
            raise ValueError(f"Brak danych dla grupy: {group_name}")

        # Wybierz wersję
        if version == "latest":
            data_path = versions[-1]["path"]
        else:
            matching = [v for v in versions if v["version"] == version]
            if not matching:
                raise ValueError(f"Wersja {version} nie istnieje")
            data_path = matching[0]["path"]

        # Wczytaj dane
        df = pd.read_parquet(data_path)
        entity_key = group_info["entity_key"]

        # Filtruj po encjach
        df = df[df[entity_key].isin(entity_ids)]

        # Wybierz kolumny
        if feature_names:
            cols = [entity_key] + feature_names
            df = df[[c for c in cols if c in df.columns]]

        return df

    def list_feature_groups(self):
        """Wyświetla wszystkie grupy cech."""
        print("\n=== Feature Store Registry ===")
        for name, info in self._registry["feature_groups"].items():
            n_versions = len(info["versions"])
            n_features = len(info["features"])
            print(f"\n📦 {name}")
            print(f"   Cechy: {n_features}, Wersje: {n_versions}")
            print(f"   Entity key: {info['entity_key']}")
            if info["versions"]:
                latest = info["versions"][-1]
                print(f"   Ostatnia wersja: {latest['version']} ({latest['n_rows']:,} wierszy)")


# Skrypt tworzenia Feature Store
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    # Wczytaj dane
    df = pd.read_parquet("data/processed/train.parquet")

    # Inicjalizuj Feature Store
    fs = SimpleFeatureStore("feature_store")

    # Zdefiniuj cechy
    user_features = [
        FeatureDefinition("age", "float", "Wiek klienta", "age"),
        FeatureDefinition("income", "float", "Dochód miesięczny", "income"),
        FeatureDefinition("tenure_months", "int", "Staż klienta", "tenure_months"),
        FeatureDefinition("num_products", "int", "Liczba produktów", "num_products"),
        FeatureDefinition("income_per_age", "float", "Dochód na rok życia", "income_per_age"),
    ]

    behavioral_features = [
        FeatureDefinition("balance", "float", "Saldo konta", "balance"),
        FeatureDefinition("num_transactions_30d", "int", "Transakcje 30d", "num_transactions_30d"),
        FeatureDefinition("is_active", "int", "Czy aktywny", "is_active"),
    ]

    # Zarejestruj grupy cech
    fs.register_feature_group("user_demographics", user_features)
    fs.register_feature_group("user_behavior", behavioral_features)

    # Zapisz cechy
    demo_cols = ['customer_id', 'age', 'income', 'tenure_months', 'num_products', 'income_per_age']
    behav_cols = ['customer_id', 'balance', 'num_transactions_30d', 'is_active']

    fs.write_features("user_demographics", df[demo_cols])
    fs.write_features("user_behavior", df[behav_cols])

    # Wyświetl rejestr
    fs.list_feature_groups()

    # Pobierz cechy dla konkretnych klientów
    sample_ids = df['customer_id'].head(5).tolist()
    features = fs.read_features(
        "user_demographics",
        entity_ids=sample_ids,
        feature_names=['age', 'income', 'income_per_age']
    )
    print(f"\nPobrane cechy dla {len(sample_ids)} klientów:")
    print(features)
```

---

## Część 4: Integracja z pipeline DVC (20 min)

### Krok 4.1: Skrypt walidacji w pipeline

Utwórz `scripts/validate_data.py`:

```python
# scripts/validate_data.py
"""Skrypt walidacji danych – używany w pipeline DVC."""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, '.')
from src.data.validation import build_churn_validator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ścieżka do danych")
    parser.add_argument("--output-report", default="reports/metrics/validation_report.json")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    print(f"Walidacja danych: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Wczytano {len(df):,} wierszy")

    validator = build_churn_validator()
    report = validator.validate(df)

    print(report.summary())

    # Zapisz raport
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "timestamp": report.timestamp,
        "passed": report.passed,
        "n_rules": len(report.results),
        "n_passed": sum(1 for r in report.results if r.passed),
        "errors": [{"rule": e.rule_name, "message": e.message} for e in report.errors],
        "warnings": [{"rule": w.rule_name, "message": w.message} for w in report.warnings],
    }
    with open(args.output_report, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Zakończ z błędem jeśli walidacja nie przeszła
    if not report.passed:
        print("\n❌ Walidacja FAILED – przerywam pipeline")
        sys.exit(1)

    if args.fail_on_warning and report.warnings:
        print("\n⚠️  Ostrzeżenia wykryte – przerywam pipeline (--fail-on-warning)")
        sys.exit(1)

    print("\n✅ Walidacja PASSED")

if __name__ == "__main__":
    main()
```

Dodaj krok walidacji do `dvc.yaml`:

```yaml
# Dodaj do dvc.yaml po etapie preprocess:
  validate:
    cmd: python scripts/validate_data.py --input data/processed/train.parquet
    deps:
      - scripts/validate_data.py
      - src/data/validation.py
      - data/processed/train.parquet
    metrics:
      - reports/metrics/validation_report.json:
          cache: false
```

```bash
# Uruchom pipeline z walidacją
dvc repro

# Sprawdź raport walidacji
cat reports/metrics/validation_report.json

# Uruchom testy
pytest tests/data/ -v --tb=short
```

---

## Zadania do samodzielnego wykonania

1. **Dodaj regułę walidacji** sprawdzającą, że `balance` jest skorelowane z `is_active` (klienci nieaktywni powinni mieć saldo = 0).
2. **Rozszerz Feature Store** o metodę `get_training_dataset()` łączącą cechy z wielu grup.
3. **Napisz test** dla `StatisticalAnomalyDetector` sprawdzający, że wykrywa wartości odstające.
4. **Dodaj wersjonowanie** do Feature Store – zapisz dwie wersje cech i pobierz starszą.

## Pytania kontrolne

1. Jaka jest różnica między błędem (`error`) a ostrzeżeniem (`warning`) w walidacji?
2. Dlaczego Feature Store rozwiązuje problem „training-serving skew"?
3. Kiedy używać metody IQR, a kiedy Z-score do wykrywania anomalii?
4. Jak zintegrować walidację danych z pipeline'em CI/CD?

## Podsumowanie

W tym laboratorium:
- ✅ Zbudowałeś framework walidacji danych z regułami i raportami
- ✅ Napisałeś testy jednostkowe dla walidatora
- ✅ Zaimplementowałeś detektor anomalii statystycznych
- ✅ Zbudowałeś prosty Feature Store z wersjonowaniem
- ✅ Zintegrowałeś walidację z pipeline'em DVC
