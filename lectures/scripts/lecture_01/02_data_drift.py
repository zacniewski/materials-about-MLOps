"""Wykład 1, sekcje 4.2–4.3 – Data drift i concept drift.

1. Data drift: test Kolmogorova-Smirnova dla kilku scenariuszy przesunięcia rozkładu.
2. Concept drift: model wytrenowany na "starej" zależności X→y traci jakość,
   gdy zależność się zmienia, mimo że rozkład X jest taki sam.

Uruchomienie:
    python lectures/scripts/lecture_01/02_data_drift.py [--plot]
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
    print("=== DATA DRIFT (test KS) ===")
    reference = rng.normal(0, 1, 1000)
    scenarios = {
        "brak dryftu N(0, 1)": rng.normal(0, 1, 500),
        "mały dryft N(0.1, 1)": rng.normal(0.1, 1, 500),
        "duży dryft N(0.5, 1.2)": rng.normal(0.5, 1.2, 500),
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
        plt.hist(reference, bins=40, alpha=0.5, density=True, label="referencja (trening)")
        plt.hist(scenarios["duży dryft N(0.5, 1.2)"], bins=40, alpha=0.5, density=True,
                 label="produkcja (dryft)")
        plt.legend()
        plt.title("Data drift – porównanie rozkładów")
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "data_drift.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        print(f"📊 Wykres zapisany do: {path}")


def concept_drift_demo(rng: np.random.Generator) -> None:
    print("\n=== CONCEPT DRIFT ===")
    X_old = rng.normal(0, 1, (2000, 2))
    y_old = (X_old[:, 0] + X_old[:, 1] > 0).astype(int)  # stara reguła
    model = LogisticRegression().fit(X_old, y_old)

    X_new = rng.permutation(X_old)                        # ten sam rozkład X!
    y_new = (X_new[:, 0] - X_new[:, 1] > 0).astype(int)   # nowa reguła

    ks = detect_data_drift(X_old[:, 0], X_new[:, 0])
    print(f"Test KS na cesze x0: drift={ks['drift_detected']} (p={ks['p_value']:.3f})")
    print(f"Accuracy na starych danych: {accuracy_score(y_old, model.predict(X_old)):.3f}")
    print(f"Accuracy na nowych danych:  {accuracy_score(y_new, model.predict(X_new)):.3f}")
    print("➡️  Rozkład X się nie zmienił, ale model przestał działać – "
          "concept drift wykryjemy tylko monitorując metryki jakości (wymaga etykiet).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="zapisz wykres histogramów do PNG")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    data_drift_demo(rng, args.plot)
    concept_drift_demo(rng)


if __name__ == "__main__":
    main()
