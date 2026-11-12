"""Lecture 1, sections 4.2–4.3 – Data drift and concept drift.

1. Data drift: Kolmogorov-Smirnov test for several distribution shift scenarios.
2. Concept drift: a model trained on the "old" X→y relationship loses quality
   when the relationship changes, even though the distribution of X stays the same.

Usage:
    python lectures-EN/scripts/lecture_01/02_data_drift.py [--plot]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

OUTPUT_DIR = Path(__file__).parent / "outputs"


def detect_data_drift(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05) -> dict:
    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < threshold,
        "severity": "high" if p_value < 0.01 else "medium" if p_value < threshold else "none",
    }


def data_drift_demo(rng: np.random.Generator, plot: bool) -> None:
    print("=== DATA DRIFT (KS test) ===")
    reference = rng.normal(0, 1, 1000)
    scenarios = {
        "no drift N(0, 1)": rng.normal(0, 1, 500),
        "small drift N(0.1, 1)": rng.normal(0.1, 1, 500),
        "large drift N(0.5, 1.2)": rng.normal(0.5, 1.2, 500),
    }
    for name, current in scenarios.items():
        r = detect_data_drift(reference, current)
        print(f"{name:<25} KS={r['statistic']:.3f}  p={r['p_value']:.4f}  "
              f"drift={r['drift_detected']!s:<5}  severity={r['severity']}")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.hist(reference, bins=40, alpha=0.5, density=True, label="reference (training)")
        plt.hist(scenarios["large drift N(0.5, 1.2)"], bins=40, alpha=0.5, density=True,
                 label="production (drift)")
        plt.legend()
        plt.title("Data drift – distribution comparison")
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "data_drift.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        print(f"📊 Plot saved to: {path}")


def concept_drift_demo(rng: np.random.Generator) -> None:
    print("\n=== CONCEPT DRIFT ===")
    X_old = rng.normal(0, 1, (2000, 2))
    y_old = (X_old[:, 0] + X_old[:, 1] > 0).astype(int)  # old rule
    model = LogisticRegression().fit(X_old, y_old)

    X_new = rng.permutation(X_old)                        # same distribution of X!
    y_new = (X_new[:, 0] - X_new[:, 1] > 0).astype(int)   # new rule

    ks = detect_data_drift(X_old[:, 0], X_new[:, 0])
    print(f"KS test on feature x0: drift={ks['drift_detected']} (p={ks['p_value']:.3f})")
    print(f"Accuracy on old data: {accuracy_score(y_old, model.predict(X_old)):.3f}")
    print(f"Accuracy on new data: {accuracy_score(y_new, model.predict(X_new)):.3f}")
    print("➡️  The distribution of X did not change, but the model stopped working – "
          "concept drift can only be detected by monitoring quality metrics (requires labels).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="save the histogram plot to PNG")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    data_drift_demo(rng, args.plot)
    concept_drift_demo(rng)


if __name__ == "__main__":
    main()
