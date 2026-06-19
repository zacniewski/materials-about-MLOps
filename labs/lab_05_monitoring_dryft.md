# Laboratorium 5: Monitoring Modelu i Wykrywanie Dryfu

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Średni–Zaawansowany
- **Wymagania wstępne:** Lab 1–4 ukończone

## Cele laboratorium
Po tym laboratorium student:
- zaimplementuje monitoring dryfu danych (KS test, PSI),
- zbuduje dashboard monitoringu z Prometheus i Grafana,
- skonfiguruje alerty przy wykryciu dryfu,
- zasymuluje data drift i zweryfikuje działanie systemu.

---

## Część 1: Implementacja monitora dryfu (50 min)

### Krok 1.1: Monitor dryfu danych

Utwórz `src/monitoring/drift_monitor.py`:

```python
# src/monitoring/drift_monitor.py
"""Monitor dryfu danych dla modelu ML."""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi_value: float
    drift_detected: bool
    severity: str   # none | low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "psi_value": round(self.psi_value, 4),
            "drift_detected": self.drift_detected,
            "severity": self.severity
        }


@dataclass
class DriftReport:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drift: bool = False
    drifted_features: list[str] = field(default_factory=list)
    n_reference_samples: int = 0
    n_current_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_drift": self.overall_drift,
            "drifted_features": self.drifted_features,
            "n_reference_samples": self.n_reference_samples,
            "n_current_samples": self.n_current_samples,
            "features": [r.to_dict() for r in self.feature_results]
        }

    def summary(self) -> str:
        status = "🔴 DRIFT" if self.overall_drift else "🟢 OK"
        lines = [
            f"=== Raport Dryfu [{self.timestamp}] ===",
            f"Status: {status}",
            f"Próbki: ref={self.n_reference_samples}, current={self.n_current_samples}",
        ]
        for r in self.feature_results:
            icon = {"none": "✅", "low": "🟡", "medium": "🟠",
                    "high": "🔴", "critical": "💀"}.get(r.severity, "❓")
            lines.append(
                f"  {icon} {r.feature}: PSI={r.psi_value:.3f}, "
                f"KS p={r.ks_pvalue:.4f}, severity={r.severity}"
            )
        return "\n".join(lines)


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Oblicza Population Stability Index."""
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_severity(psi: float, ks_pvalue: float) -> tuple[bool, str]:
    """Klasyfikuje nasilenie dryfu."""
    if psi > 0.5 or ks_pvalue < 0.001:
        return True, "critical"
    elif psi > 0.2 or ks_pvalue < 0.01:
        return True, "high"
    elif psi > 0.1 or ks_pvalue < 0.05:
        return True, "medium"
    elif psi > 0.05:
        return False, "low"
    return False, "none"


class DataDriftMonitor:
    """Monitor dryfu danych."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_columns: list[str],
        alert_callback: Optional[Callable] = None,
        ks_threshold: float = 0.05
    ):
        self.reference_df = reference_df
        self.feature_columns = [c for c in feature_columns if c in reference_df.columns]
        self.alert_callback = alert_callback
        self.ks_threshold = ks_threshold
        self._history: list[DriftReport] = []

    def check_drift(self, current_df: pd.DataFrame) -> DriftReport:
        """Sprawdza dryft dla wszystkich cech."""
        report = DriftReport(
            n_reference_samples=len(self.reference_df),
            n_current_samples=len(current_df)
        )

        for feature in self.feature_columns:
            if feature not in current_df.columns:
                continue

            ref_vals = self.reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values

            if len(cur_vals) < 30:
                logger.warning(f"Za mało próbek dla {feature}: {len(cur_vals)}")
                continue

            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
            psi = calculate_psi(ref_vals, cur_vals)
            drift_detected, severity = classify_severity(psi, ks_pvalue)

            result = FeatureDriftResult(
                feature=feature,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                psi_value=psi,
                drift_detected=drift_detected,
                severity=severity
            )
            report.feature_results.append(result)

            if drift_detected:
                report.drifted_features.append(feature)

        report.overall_drift = len(report.drifted_features) > 0
        self._history.append(report)

        if report.overall_drift and self.alert_callback:
            self.alert_callback(report)

        return report

    def get_drift_trend(self) -> pd.DataFrame:
        """Zwraca historię dryfu jako DataFrame."""
        rows = []
        for rep in self._history:
            for feat in rep.feature_results:
                rows.append({
                    "timestamp": rep.timestamp,
                    "feature": feat.feature,
                    "psi": feat.psi_value,
                    "ks_pvalue": feat.ks_pvalue,
                    "severity": feat.severity,
                    "drift": feat.drift_detected
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_report(self, report: DriftReport, path: str):
        """Zapisuje raport do pliku JSON."""
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Raport zapisany: {path}")
```

### Krok 1.2: Symulacja dryfu i testy

Utwórz `scripts/simulate_drift.py`:

```python
# scripts/simulate_drift.py
"""Symuluje data drift i testuje monitor."""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.monitoring.drift_monitor import DataDriftMonitor

np.random.seed(42)

# ── Dane referencyjne (z treningu) ───────────────────────────────────────────
n_ref = 5000
reference_df = pd.DataFrame({
    'age':                np.random.normal(40, 10, n_ref).clip(18, 90),
    'income':             np.random.normal(50000, 15000, n_ref).clip(0),
    'tenure_months':      np.random.exponential(24, n_ref).clip(1, 240),
    'balance':            np.random.lognormal(9, 1, n_ref),
    'num_transactions_30d': np.random.poisson(15, n_ref),
})

feature_cols = list(reference_df.columns)

# ── Scenariusze dryfu ────────────────────────────────────────────────────────

def generate_no_drift(n=500) -> pd.DataFrame:
    """Dane bez dryfu – taki sam rozkład jak referencyjne."""
    return pd.DataFrame({
        'age':                np.random.normal(40, 10, n).clip(18, 90),
        'income':             np.random.normal(50000, 15000, n).clip(0),
        'tenure_months':      np.random.exponential(24, n).clip(1, 240),
        'balance':            np.random.lognormal(9, 1, n),
        'num_transactions_30d': np.random.poisson(15, n),
    })

def generate_mild_drift(n=500) -> pd.DataFrame:
    """Łagodny dryft – małe przesunięcie średniej."""
    return pd.DataFrame({
        'age':                np.random.normal(43, 11, n).clip(18, 90),
        'income':             np.random.normal(52000, 15000, n).clip(0),
        'tenure_months':      np.random.exponential(22, n).clip(1, 240),
        'balance':            np.random.lognormal(9, 1, n),
        'num_transactions_30d': np.random.poisson(14, n),
    })

def generate_severe_drift(n=500) -> pd.DataFrame:
    """Silny dryft – znaczące zmiany rozkładu."""
    return pd.DataFrame({
        'age':                np.random.normal(55, 15, n).clip(18, 90),  # starsi klienci
        'income':             np.random.normal(35000, 20000, n).clip(0), # niższe dochody
        'tenure_months':      np.random.exponential(8, n).clip(1, 240),  # krótszy staż
        'balance':            np.random.lognormal(7, 1.5, n),            # niższe salda
        'num_transactions_30d': np.random.poisson(5, n),                 # mniej transakcji
    })

# ── Uruchomienie monitoringu ─────────────────────────────────────────────────

def alert_handler(report):
    print(f"\n🚨 ALERT: Wykryto drift w cechach: {report.drifted_features}")

monitor = DataDriftMonitor(
    reference_df=reference_df,
    feature_columns=feature_cols,
    alert_callback=alert_handler
)

print("=" * 60)
print("SCENARIUSZ 1: Brak dryfu")
print("=" * 60)
report1 = monitor.check_drift(generate_no_drift())
print(report1.summary())

print("\n" + "=" * 60)
print("SCENARIUSZ 2: Łagodny dryft")
print("=" * 60)
report2 = monitor.check_drift(generate_mild_drift())
print(report2.summary())

print("\n" + "=" * 60)
print("SCENARIUSZ 3: Silny dryft")
print("=" * 60)
report3 = monitor.check_drift(generate_severe_drift())
print(report3.summary())

# ── Wizualizacja ─────────────────────────────────────────────────────────────

Path("reports/figures").mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feature in enumerate(feature_cols):
    ax = axes[i]
    ref_vals = reference_df[feature].values
    severe_vals = generate_severe_drift()[feature].values

    ax.hist(ref_vals, bins=30, alpha=0.6, label='Referencyjne', color='blue', density=True)
    ax.hist(severe_vals, bins=30, alpha=0.6, label='Produkcyjne (drift)', color='red', density=True)
    ax.set_title(f'{feature}')
    ax.legend(fontsize=8)
    ax.set_xlabel('Wartość')
    ax.set_ylabel('Gęstość')

plt.suptitle('Porównanie rozkładów: Referencyjne vs Produkcyjne (silny drift)', fontsize=12)
plt.tight_layout()
plt.savefig("reports/figures/drift_comparison.png", dpi=100, bbox_inches='tight')
print("\n📊 Wykres zapisany: reports/figures/drift_comparison.png")

# ── Trend dryfu ──────────────────────────────────────────────────────────────

trend_df = monitor.get_drift_trend()
if not trend_df.empty:
    print(f"\nHistoria dryfu ({len(trend_df)} wpisów):")
    print(trend_df.groupby(['feature', 'severity']).size().unstack(fill_value=0))
```

```bash
python scripts/simulate_drift.py
```

### Krok 1.3: Testy monitora dryfu

Utwórz `tests/model/test_drift_monitor.py`:

```python
# tests/model/test_drift_monitor.py
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.drift_monitor import (
    DataDriftMonitor, calculate_psi, classify_severity
)

@pytest.fixture
def reference_df():
    np.random.seed(42)
    n = 2000
    return pd.DataFrame({
        'age': np.random.normal(40, 10, n),
        'income': np.random.normal(50000, 15000, n),
        'balance': np.random.lognormal(9, 1, n),
    })

class TestCalculatePSI:

    def test_psi_same_distribution_near_zero(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 5000)
        cur = np.random.normal(0, 1, 1000)
        psi = calculate_psi(ref, cur)
        assert psi < 0.1, f"PSI dla tego samego rozkładu powinno być < 0.1, got {psi}"

    def test_psi_different_distribution_high(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 5000)
        cur = np.random.normal(5, 1, 1000)  # bardzo różny rozkład
        psi = calculate_psi(ref, cur)
        assert psi > 0.2, f"PSI dla różnych rozkładów powinno być > 0.2, got {psi}"

    def test_psi_non_negative(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(1, 1, 500)
        psi = calculate_psi(ref, cur)
        assert psi >= 0

class TestClassifySeverity:

    def test_no_drift_when_psi_low_pvalue_high(self):
        drift, severity = classify_severity(psi=0.02, ks_pvalue=0.5)
        assert not drift
        assert severity == "none"

    def test_high_severity_when_psi_very_high(self):
        drift, severity = classify_severity(psi=0.6, ks_pvalue=0.001)
        assert drift
        assert severity == "critical"

    def test_medium_severity(self):
        drift, severity = classify_severity(psi=0.15, ks_pvalue=0.03)
        assert drift
        assert severity == "medium"

class TestDataDriftMonitor:

    def test_no_drift_detected_same_distribution(self, reference_df):
        np.random.seed(99)
        current_df = pd.DataFrame({
            'age': np.random.normal(40, 10, 500),
            'income': np.random.normal(50000, 15000, 500),
            'balance': np.random.lognormal(9, 1, 500),
        })
        monitor = DataDriftMonitor(reference_df, list(reference_df.columns))
        report = monitor.check_drift(current_df)
        assert not report.overall_drift

    def test_drift_detected_different_distribution(self, reference_df):
        np.random.seed(99)
        current_df = pd.DataFrame({
            'age': np.random.normal(65, 5, 500),      # bardzo różny
            'income': np.random.normal(20000, 5000, 500),
            'balance': np.random.lognormal(6, 0.5, 500),
        })
        monitor = DataDriftMonitor(reference_df, list(reference_df.columns))
        report = monitor.check_drift(current_df)
        assert report.overall_drift
        assert len(report.drifted_features) > 0

    def test_alert_callback_called_on_drift(self, reference_df):
        alerts = []
        def callback(report):
            alerts.append(report)

        np.random.seed(99)
        current_df = pd.DataFrame({
            'age': np.random.normal(70, 5, 500),
            'income': np.random.normal(15000, 3000, 500),
            'balance': np.random.lognormal(5, 0.5, 500),
        })
        monitor = DataDriftMonitor(
            reference_df, list(reference_df.columns), alert_callback=callback
        )
        monitor.check_drift(current_df)
        assert len(alerts) > 0

    def test_report_contains_all_features(self, reference_df):
        current_df = reference_df.sample(500)
        monitor = DataDriftMonitor(reference_df, list(reference_df.columns))
        report = monitor.check_drift(current_df)
        reported_features = {r.feature for r in report.feature_results}
        assert reported_features == set(reference_df.columns)

    def test_drift_history_accumulates(self, reference_df):
        monitor = DataDriftMonitor(reference_df, list(reference_df.columns))
        for _ in range(3):
            monitor.check_drift(reference_df.sample(200))
        trend = monitor.get_drift_trend()
        assert len(trend) > 0
```

```bash
pytest tests/model/test_drift_monitor.py -v
```

---

## Część 2: Metryki Prometheus dla monitoringu ML (30 min)

### Krok 2.1: Eksporter metryk

Utwórz `src/monitoring/metrics_exporter.py`:

```python
# src/monitoring/metrics_exporter.py
"""Eksporter metryk ML do Prometheus."""

from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server, CollectorRegistry, REGISTRY
)
import time
import numpy as np
import threading

# ── Definicje metryk ─────────────────────────────────────────────────────────

PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Łączna liczba predykcji',
    ['model_version', 'risk_level']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Czas predykcji',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

CHURN_PROBABILITY_AVG = Gauge(
    'ml_churn_probability_avg',
    'Średnie prawdopodobieństwo churnu (ostatnie 1000)'
)

DATA_DRIFT_PSI = Gauge(
    'ml_data_drift_psi',
    'PSI dla cechy',
    ['feature_name']
)

DATA_DRIFT_DETECTED = Gauge(
    'ml_data_drift_detected',
    'Czy wykryto drift (1=tak, 0=nie)',
    ['feature_name']
)

MODEL_AUC = Gauge(
    'ml_model_auc_roc',
    'AUC-ROC modelu',
    ['model_version']
)

ERRORS_TOTAL = Counter(
    'ml_prediction_errors_total',
    'Liczba błędów predykcji',
    ['error_type']
)


class MLMetricsCollector:
    """Zbiera i eksportuje metryki ML."""

    def __init__(self, model_version: str = "v1.0"):
        self.model_version = model_version
        self._recent_probs: list[float] = []
        self._lock = threading.Lock()

    def record_prediction(
        self,
        probability: float,
        latency_seconds: float,
        risk_level: str
    ):
        """Rejestruje predykcję."""
        PREDICTIONS_TOTAL.labels(
            model_version=self.model_version,
            risk_level=risk_level
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=self.model_version
        ).observe(latency_seconds)

        with self._lock:
            self._recent_probs.append(probability)
            if len(self._recent_probs) > 1000:
                self._recent_probs.pop(0)
            CHURN_PROBABILITY_AVG.set(np.mean(self._recent_probs))

    def record_error(self, error_type: str):
        """Rejestruje błąd predykcji."""
        ERRORS_TOTAL.labels(error_type=error_type).inc()

    def update_drift_metrics(self, drift_report):
        """Aktualizuje metryki dryfu."""
        for feat_result in drift_report.feature_results:
            DATA_DRIFT_PSI.labels(
                feature_name=feat_result.feature
            ).set(feat_result.psi_value)

            DATA_DRIFT_DETECTED.labels(
                feature_name=feat_result.feature
            ).set(1 if feat_result.drift_detected else 0)

    def update_model_auc(self, auc: float):
        """Aktualizuje metrykę AUC modelu."""
        MODEL_AUC.labels(model_version=self.model_version).set(auc)


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    collector = MLMetricsCollector(model_version="v1.0")

    # Uruchom serwer metryk
    start_http_server(8000)
    print("Metryki dostępne na http://localhost:8000/metrics")
    print("Symulacja predykcji... (Ctrl+C aby zatrzymać)")

    collector.update_model_auc(0.87)

    while True:
        # Symuluj predykcję
        prob = random.betavariate(2, 5)  # większość niskich prawdopodobieństw
        latency = random.expovariate(50)  # ~20ms średnio
        risk = "high" if prob > 0.7 else "medium" if prob > 0.3 else "low"

        collector.record_prediction(prob, latency, risk)

        # Co 100 predykcji – symuluj drift check
        if random.random() < 0.01:
            print(f"Predykcja: prob={prob:.3f}, latency={latency*1000:.1f}ms, risk={risk}")

        time.sleep(0.05)  # 20 predykcji/s
```

### Krok 2.2: Konfiguracja Prometheus i Grafana

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-service'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'
```

```yaml
# monitoring/docker-compose-monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=7d'

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=mlops123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-data:
```

```bash
# Uruchom monitoring stack
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d

# Uruchom eksporter metryk (w osobnym terminalu)
python src/monitoring/metrics_exporter.py

# Sprawdź metryki
curl http://localhost:8000/metrics | grep ml_

# Otwórz Grafana: http://localhost:3000 (admin/mlops123)
# Otwórz Prometheus: http://localhost:9090
```

---

## Część 3: Automatyczny retraining (30 min)

### Krok 3.1: Scheduler monitoringu

Utwórz `scripts/monitoring_scheduler.py`:

```python
# scripts/monitoring_scheduler.py
"""Harmonogram monitoringu z automatycznym retreaningiem."""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from pathlib import Path

from src.monitoring.drift_monitor import DataDriftMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Harmonogram monitoringu modelu ML."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_columns: list[str],
        check_interval_seconds: int = 60,
        drift_threshold_psi: float = 0.2
    ):
        self.monitor = DataDriftMonitor(
            reference_df=reference_df,
            feature_columns=feature_columns,
            alert_callback=self._on_drift_detected
        )
        self.check_interval = check_interval_seconds
        self.drift_threshold = drift_threshold_psi
        self._retraining_triggered = False
        self._check_count = 0

    def _on_drift_detected(self, report):
        """Callback wywoływany przy wykryciu dryfu."""
        logger.warning(f"🚨 DRIFT WYKRYTY: {report.drifted_features}")

        # Sprawdź nasilenie
        critical_features = [
            r.feature for r in report.feature_results
            if r.severity in ("high", "critical")
        ]

        if critical_features and not self._retraining_triggered:
            logger.warning(f"Krytyczny drift w: {critical_features}")
            self._trigger_retraining(report)

    def _trigger_retraining(self, report):
        """Wyzwala retraining modelu."""
        logger.info("🔄 Wyzwalanie retreningu...")
        self._retraining_triggered = True

        # W produkcji: uruchom pipeline ML
        # subprocess.run(["python", "src/models/train.py"])
        # lub: kfp_client.create_run(pipeline_func=training_pipeline)

        logger.info("✅ Retraining wyzwolony (symulacja)")

        # Zapisz raport
        Path("reports/metrics").mkdir(parents=True, exist_ok=True)
        self.monitor.save_report(
            report,
            f"reports/metrics/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    def run_once(self, current_df: pd.DataFrame):
        """Wykonuje jedno sprawdzenie dryfu."""
        self._check_count += 1
        logger.info(f"Sprawdzenie #{self._check_count} ({len(current_df)} próbek)")

        report = self.monitor.check_drift(current_df)

        if not report.overall_drift:
            logger.info("✅ Brak dryfu")
        
        return report

    def simulate_monitoring(self, n_checks: int = 5):
        """Symuluje monitoring przez N sprawdzeń."""
        logger.info(f"Rozpoczynam symulację monitoringu ({n_checks} sprawdzeń)")

        for i in range(n_checks):
            # Symuluj stopniowy drift
            drift_factor = i / n_checks

            current_df = pd.DataFrame({
                'age': np.random.normal(40 + drift_factor * 20, 10, 300),
                'income': np.random.normal(50000 - drift_factor * 20000, 15000, 300),
                'tenure_months': np.random.exponential(24 - drift_factor * 15, 300).clip(1),
                'balance': np.random.lognormal(9 - drift_factor * 2, 1, 300),
                'num_transactions_30d': np.random.poisson(max(1, 15 - drift_factor * 12), 300),
            })

            report = self.run_once(current_df)
            logger.info(f"  Dryft: {report.overall_drift}, cechy: {report.drifted_features}")

            if i < n_checks - 1:
                time.sleep(1)  # krótka przerwa między sprawdzeniami

        # Podsumowanie
        trend = self.monitor.get_drift_trend()
        if not trend.empty:
            logger.info("\n=== Trend dryfu ===")
            summary = trend.groupby('feature')['psi'].agg(['mean', 'max'])
            logger.info(f"\n{summary}")


# Uruchomienie
if __name__ == "__main__":
    np.random.seed(42)

    # Dane referencyjne
    reference = pd.DataFrame({
        'age': np.random.normal(40, 10, 5000).clip(18, 90),
        'income': np.random.normal(50000, 15000, 5000).clip(0),
        'tenure_months': np.random.exponential(24, 5000).clip(1, 240),
        'balance': np.random.lognormal(9, 1, 5000),
        'num_transactions_30d': np.random.poisson(15, 5000),
    })

    scheduler = MonitoringScheduler(
        reference_df=reference,
        feature_columns=list(reference.columns),
        check_interval_seconds=60
    )

    scheduler.simulate_monitoring(n_checks=6)
```

```bash
python scripts/monitoring_scheduler.py
```

---

## Zadania do samodzielnego wykonania

1. **Dodaj monitoring predykcji** – śledź rozkład predykcji modelu (nie tylko cech wejściowych).
2. **Zaimplementuj Chi-square test** dla zmiennych kategorycznych (np. `num_products`).
3. **Stwórz dashboard Grafana** z panelami: PSI per cecha, latencja p95, error rate.
4. **Napisz test** sprawdzający, że `MonitoringScheduler` wyzwala retraining przy silnym dryfcie.

## Pytania kontrolne

1. Jaka jest różnica między data drift a concept drift?
2. Kiedy PSI > 0.2 jest poważnym problemem, a kiedy można je zignorować?
3. Dlaczego monitorujemy rozkład predykcji nawet gdy nie mamy etykiet?
4. Jak uniknąć fałszywych alarmów w systemie monitoringu?

## Podsumowanie

W tym laboratorium:
- ✅ Zaimplementowałeś monitor dryfu z KS test i PSI
- ✅ Zasymulowałeś różne scenariusze dryfu danych
- ✅ Skonfigurowałeś eksport metryk do Prometheus
- ✅ Uruchomiłeś Grafana do wizualizacji metryk
- ✅ Zbudowałeś scheduler monitoringu z automatycznym retreaningiem
