"""Wykład 8, sekcja 2.3 – Batching: latencja vs przepustowość.

Mierzy, jak rozmiar batcha wpływa na czas obsługi jednego wywołania modelu
(latencja) i liczbę predykcji na sekundę (przepustowość). Następnie symuluje
dynamic batching: żądania czekają w kolejce max `--max-wait-ms`, a im dłużej
czekają, tym większy batch i tym wyższa latencja ogona (p95/p99).

Uruchomienie:
    python lectures/scripts/lecture_08/01_batching_latency_throughput.py [--requests 2000]
"""

import argparse
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def measure_batch(model, X: np.ndarray, batch_size: int, repeats: int = 20) -> tuple[float, float]:
    batch = X[:batch_size]
    model.predict_proba(batch)  # warmup
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(batch)
    per_call = (time.perf_counter() - start) / repeats
    return per_call * 1000, batch_size / per_call


def simulate_dynamic_batching(call_ms: dict, rps: float, max_batch: int, max_wait_ms: float,
                              n_requests: int, rng) -> np.ndarray:
    """Symulacja zdarzeń: kolejka + jeden worker. Zwraca latencje end-to-end [ms]."""
    arrivals = np.cumsum(rng.exponential(1000 / rps, n_requests))
    latencies = np.empty(n_requests)
    sizes = sorted(call_ms)
    worker_free = 0.0
    i = 0
    while i < n_requests:
        start = max(worker_free, arrivals[i])
        deadline = start + max_wait_ms
        j = i
        while j < n_requests and j - i < max_batch and arrivals[j] <= deadline:
            j += 1
        j = max(j, i + 1)
        bs = j - i
        # pełny batch → start od razu po przyjściu ostatniego żądania; niepełny → po upływie okna
        dispatch = max(start, arrivals[j - 1]) if bs == max_batch else deadline
        cost = np.interp(bs, sizes, [call_ms[s] for s in sizes])
        done = dispatch + cost
        latencies[i:j] = done - arrivals[i:j]
        worker_free = done
        i = j
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--rps", type=float, default=800)
    args = parser.parse_args()
    rng = np.random.default_rng(42)

    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1).fit(X, y)

    print("=== 1. Batching statyczny (jedno wywołanie modelu) ===")
    print(f"{'batch':>6} | {'latencja wywołania [ms]':>24} | {'przepustowość [pred/s]':>22}")
    call_ms = {}
    for bs in [1, 8, 32, 128, 512]:
        lat, thr = measure_batch(model, X, bs)
        call_ms[bs] = lat
        print(f"{bs:>6} | {lat:>24.2f} | {thr:>22.0f}")
    print("→ Większy batch = koszt stały (overhead) rozkłada się na więcej predykcji.\n")

    print(f"=== 2. Dynamic batching (ruch Poissona {args.rps:.0f} RPS, 1 worker) ===")
    print(f"{'max_batch':>9} | {'max_wait':>8} | {'p50 [ms]':>8} | {'p95 [ms]':>8} | {'p99 [ms]':>8}")
    for max_batch, max_wait in [(1, 0), (8, 2), (32, 5), (128, 20)]:
        lat = simulate_dynamic_batching(call_ms, args.rps, max_batch, max_wait, args.requests, rng)
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        print(f"{max_batch:>9} | {max_wait:>6}ms | {p50:>8.1f} | {p95:>8.1f} | {p99:>8.1f}")
    print("→ Bez batchingu kolejka może rosnąć bez końca (worker nie nadąża).")
    print("→ Zbyt długie okno czekania podnosi latencję – trzeba dobrać kompromis pod SLO.")


if __name__ == "__main__":
    main()
