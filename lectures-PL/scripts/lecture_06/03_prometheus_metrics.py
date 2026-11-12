"""Wykład 6, sekcja 5 – Instrumentacja serwisu ML metrykami Prometheus.

Symuluje 2 000 predykcji z instrumentacją (Counter, Histogram, Gauge) na
własnym CollectorRegistry, a następnie:
  * wypisuje fragment formatu ekspozycji tekstowej (to, co widzi Prometheus
    pod /metrics),
  * odtwarza p95 latencji z bucketów histogramu (jak histogram_quantile),
  * zapisuje pełną ekspozycję do outputs/metrics.txt.
Nie uruchamia serwera HTTP – flaga --serve włącza start_http_server(8000).

Uruchomienie:
    python lectures/scripts/lecture_06/03_prometheus_metrics.py [--serve]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

OUT_DIR = Path(__file__).parent / "outputs"
BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]


def histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float:
    """Interpolacja liniowa w buckecie – ta sama idea co w PromQL."""
    total = buckets[-1][1]
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * (rank - prev_count) / max(count - prev_count, 1)
        prev_le, prev_count = le, count
    return prev_le


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    registry = CollectorRegistry()
    predictions = Counter("ml_predictions_total", "Liczba predykcji",
                          ["model_version", "risk_level"], registry=registry)
    errors = Counter("ml_prediction_errors_total", "Liczba błędów",
                     ["error_type"], registry=registry)
    latency = Histogram("ml_prediction_latency_seconds", "Latencja predykcji",
                        ["model_version"], buckets=BUCKETS, registry=registry)
    drift = Gauge("ml_data_drift_psi", "PSI cechy", ["feature_name"], registry=registry)

    rng = np.random.default_rng(42)
    for _ in range(2_000):
        lat = rng.lognormal(mean=np.log(0.02), sigma=0.6)
        latency.labels(model_version="v2").observe(lat)
        if rng.random() < 0.005:
            errors.labels(error_type="ValueError").inc()
            continue
        p = rng.beta(2, 5)
        risk = "high" if p > 0.7 else "medium" if p > 0.3 else "low"
        predictions.labels(model_version="v2", risk_level=risk).inc()
    for feature, value in {"age": 0.04, "income": 0.13, "tenure": 0.27}.items():
        drift.labels(feature_name=feature).set(value)

    exposition = generate_latest(registry).decode()
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "metrics.txt").write_text(exposition, encoding="utf-8")

    print("=== Fragment /metrics ===")
    for line in exposition.splitlines():
        if line.startswith(("ml_predictions_total{", "ml_data_drift_psi{",
                            "ml_prediction_errors_total{", "# TYPE")):
            print(line)

    buckets = []
    for metric in registry.collect():
        for s in metric.samples:
            if s.name == "ml_prediction_latency_seconds_bucket":
                buckets.append((float(s.labels["le"]), s.value))
    print("\n=== Latencja z histogramu ===")
    for q in (0.5, 0.95, 0.99):
        print(f"  p{int(q * 100)} ≈ {histogram_quantile(q, buckets) * 1000:.1f} ms")
    print("\nAlert DataDriftDetected (PSI > 0.2) odpaliłby dla: tenure")
    print(f"Pełna ekspozycja: {OUT_DIR / 'metrics.txt'}")

    if args.serve:
        start_http_server(8000, registry=registry)
        print("Metryki na http://localhost:8000/metrics (Ctrl+C aby zakończyć)")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
