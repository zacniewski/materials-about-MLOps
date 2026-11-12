"""Lecture 5, section 9 – Latency, percentiles and (micro)batching.

Measures prediction time of single requests vs. batches of various sizes,
reports p50/p95/p99 and throughput, and shows why the mean
hides the latency "tail". Saves a plot to outputs/latency_batching.png.

Run:
    python lectures/scripts/lecture_05/02_latency_batching.py [--requests 2000]
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

OUT_DIR = Path(__file__).parent / "outputs"


def percentiles(lat_ms: np.ndarray) -> dict:
    return {f"p{q}": float(np.percentile(lat_ms, q)) for q in (50, 95, 99)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2000)
    args = parser.parse_args()

    X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42).fit(X, y)
    rng = np.random.default_rng(0)
    data = X[rng.integers(0, len(X), args.requests)]

    print(f"{'batch':>6} | {'p50 ms/req':>10} | {'p95':>8} | {'p99':>8} | {'mean':>8} | {'req/s':>8}")
    print("-" * 64)
    results = {}
    for bs in (1, 8, 32, 128, 512):
        per_req = []
        t_start = time.perf_counter()
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            t0 = time.perf_counter()
            model.predict_proba(chunk)
            dt = (time.perf_counter() - t0) * 1000
            # every request in the batch "waits" for the whole batch
            per_req.extend([dt] * len(chunk))
            if bs == 1 and i >= 500:  # single requests are slow – a sample is enough
                break
        total = time.perf_counter() - t_start
        lat = np.array(per_req)
        pc = percentiles(lat)
        thr = len(lat) / total
        results[bs] = (pc, thr)
        print(f"{bs:>6} | {pc['p50']:>10.2f} | {pc['p95']:>8.2f} | {pc['p99']:>8.2f} | "
              f"{lat.mean():>8.2f} | {thr:>8.0f}")

    print("\nConclusions:")
    print(" * Batching drastically increases throughput (call overhead is amortized),")
    print("   but single-request latency grows – the latency vs throughput trade-off.")
    print(" * SLAs are defined on p95/p99, not on the mean.")

    # Tail simulation: 1% of requests hit GC / a cold cache (+50 ms)
    base = rng.normal(20, 3, 10_000).clip(5)
    tail = base + (rng.random(10_000) < 0.01) * 50
    print(f"\nLatency tail: mean={tail.mean():.1f} ms, p50={np.percentile(tail, 50):.1f}, "
          f"p99={np.percentile(tail, 99):.1f} ms  <- the mean barely changes!")

    OUT_DIR.mkdir(exist_ok=True)
    bss = list(results)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(bss, [results[b][0]["p99"] for b in bss], "o-r", label="p99 latency [ms]")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("p99 [ms]", color="r")
    ax2 = ax1.twinx()
    ax2.plot(bss, [results[b][1] for b in bss], "s-b", label="throughput")
    ax2.set_ylabel("req/s", color="b")
    plt.title("Batching: latency vs throughput")
    fig.tight_layout()
    path = OUT_DIR / "latency_batching.png"
    fig.savefig(path, dpi=100)
    print(f"\nPlot saved: {path}")


if __name__ == "__main__":
    main()
