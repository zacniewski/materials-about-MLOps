"""Lecture 6, section 3 – Drift metrics: PSI, KS, KL, JS and Wasserstein.

Compares drift metrics for a growing shift of a feature distribution and
shows the large-sample trap: at n=100 000 the KS test "detects" drift
even for changes with no business significance, while PSI stays stable.

Run:
    python lectures/scripts/lecture_06/01_drift_metrics.py
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPS = 1e-4


def binned(reference: np.ndarray, current: np.ndarray, n_bins: int = 10):
    """Proportions in bins defined by quantiles of the reference data."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    return np.clip(ref, EPS, None), np.clip(cur, EPS, None)


def psi(reference, current, n_bins: int = 10) -> float:
    ref, cur = binned(reference, current, n_bins)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def kl(reference, current, n_bins: int = 10) -> float:
    ref, cur = binned(reference, current, n_bins)
    return float(np.sum(cur * np.log(cur / ref)))  # KL(cur || ref)


def js(reference, current, n_bins: int = 10) -> float:
    ref, cur = binned(reference, current, n_bins)
    return float(jensenshannon(ref, cur, base=2))  # JS distance in [0, 1]


def all_metrics(reference, current) -> dict:
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return {
        "PSI": psi(reference, current),
        "KL": kl(reference, current),
        "JS": js(reference, current),
        "Wass/σ": stats.wasserstein_distance(reference, current) / reference.std(),
        "KS D": ks_stat,
        "KS p": p_value,
    }


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n=== {title} ===")
    header = f"{'scenario':<22}" + "".join(f"{k:>10}" for k in rows[0][1])
    print(header)
    print("-" * len(header))
    for name, m in rows:
        cells = "".join(
            f"{v:>10.2e}" if k == "KS p" else f"{v:>10.4f}" for k, v in m.items()
        )
        print(f"{name:<22}{cells}")


def main() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(50, 10, 10_000)

    rows = []
    for shift in [0.0, 1.0, 2.5, 5.0, 10.0]:
        current = rng.normal(50 + shift, 10, 2_000)
        rows.append((f"shift μ +{shift}", all_metrics(reference, current)))
    current = rng.normal(50, 15, 2_000)
    rows.append(("increase σ 10→15", all_metrics(reference, current)))
    print_table("Growing drift (n_ref=10 000, n_cur=2 000)", rows)

    print("\nPSI interpretation: <0.1 none | 0.1–0.2 moderate | >0.2 significant drift")

    rows = []
    for n in [200, 2_000, 20_000, 100_000]:
        big_ref = rng.normal(50, 10, n)
        cur = rng.normal(50.5, 10, n)  # shift of 0.05σ – insignificant
        rows.append((f"n={n}", all_metrics(big_ref, cur)))
    print_table("Large-sample trap: the same small change (+0.05σ)", rows)
    print(
        "\nConclusion: KS p-value drops with n (alarm under heavy traffic), while PSI/JS/"
        "Wasserstein\nmeasure the SIZE of the change – use effect size for alerts, not the p-value alone."
    )


if __name__ == "__main__":
    main()
