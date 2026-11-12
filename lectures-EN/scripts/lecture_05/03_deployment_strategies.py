"""Lecture 5, section 4 – Deployment strategy simulation: shadow and canary with auto-rollback.

Two models: stable v1 and candidate v2. Scenario A: v2 is better,
scenario B: v2 has a hidden bug (swapped feature columns – a typical
training-serving skew). Shows:
  * shadow – comparing predictions without affecting the user,
  * canary – deterministic routing (hash of user_id, "sticky"),
    gradually increasing traffic 5% -> 25% -> 50% -> 100% and automatic
    rollback when the canary metric drops below the threshold.

Run:
    python lectures/scripts/lecture_05/03_deployment_strategies.py
"""

import hashlib

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

STEPS = [0.05, 0.25, 0.50, 1.00]
MAX_DEGRADATION = 0.02  # canary may be at most 2 pp worse than stable


def in_canary(user_id: int, pct: float) -> bool:
    """Sticky routing: the same user always hits the same version."""
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 10_000
    return h < pct * 10_000


class BuggyModel:
    """Wrapper simulating a bug in serving code (wrong feature order)."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X[:, ::-1])


def shadow(stable, candidate, X) -> None:
    p1, p2 = stable.predict(X), candidate.predict(X)
    print(f"  [shadow] prediction agreement v1 vs v2: {np.mean(p1 == p2):.1%} "
          f"(the user always gets v1)")


def canary(stable, candidate, X, y, user_ids) -> str:
    for pct in STEPS:
        mask = np.array([in_canary(u, pct) for u in user_ids])
        acc_c = accuracy_score(y[mask], candidate.predict(X[mask]))
        acc_s = accuracy_score(y[~mask], stable.predict(X[~mask])) if (~mask).any() else acc_c
        ok = acc_c >= acc_s - MAX_DEGRADATION
        print(f"  [canary {pct:>4.0%}] n={mask.sum():>5}  acc_canary={acc_c:.3f}  "
              f"acc_stable={acc_s:.3f}  -> {'OK' if ok else 'ROLLBACK'}")
        if not ok:
            return "ROLLBACK – 100% of traffic returns to v1"
    return "PROMOTED – v2 serves 100% of traffic"


def main() -> None:
    X, y = make_classification(n_samples=12_000, n_features=8, n_informative=5,
                               class_sep=0.8, random_state=7)
    X_tr, y_tr, X_live, y_live = X[:6000], y[:6000], X[6000:], y[6000:]
    user_ids = np.arange(100_000, 100_000 + len(X_live))

    v1 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    v2 = GradientBoostingClassifier(random_state=7).fit(X_tr, y_tr)

    print("Sticky routing (5%): user 100123 ->",
          "canary" if in_canary(100123, 0.05) else "stable", "(the same every time)")

    for name, cand in [("A: v2 better", v2), ("B: v2 with feature bug", BuggyModel(v2))]:
        print(f"\nScenario {name}")
        shadow(v1, cand, X_live)
        print("  Result:", canary(v1, cand, X_live, y_live, user_ids))


if __name__ == "__main__":
    main()
