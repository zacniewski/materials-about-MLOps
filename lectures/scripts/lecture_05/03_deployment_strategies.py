"""Wykład 5, sekcja 4 – Symulacja strategii wdrożeń: shadow i canary z auto-rollbackiem.

Dwa modele: stabilny v1 i kandydat v2. Scenariusz A: v2 jest lepszy,
scenariusz B: v2 ma ukryty błąd (zamienione kolumny cech – typowy
training-serving skew). Pokazuje:
  * shadow – porównanie predykcji bez wpływu na użytkownika,
  * canary – deterministyczny routing (hash user_id, "sticky"),
    stopniowe zwiększanie ruchu 5% -> 25% -> 50% -> 100% i automatyczny
    rollback, gdy metryka canary spada poniżej progu.

Uruchomienie:
    python lectures/scripts/lecture_05/03_deployment_strategies.py
"""

import hashlib

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

STEPS = [0.05, 0.25, 0.50, 1.00]
MAX_DEGRADATION = 0.02  # canary może być co najwyżej 2 pp gorszy od stable


def in_canary(user_id: int, pct: float) -> bool:
    """Sticky routing: ten sam użytkownik zawsze trafia do tej samej wersji."""
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 10_000
    return h < pct * 10_000


class BuggyModel:
    """Opakowanie symulujące błąd w kodzie serwującym (zła kolejność cech)."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X[:, ::-1])


def shadow(stable, candidate, X) -> None:
    p1, p2 = stable.predict(X), candidate.predict(X)
    print(f"  [shadow] zgodność predykcji v1 vs v2: {np.mean(p1 == p2):.1%} "
          f"(użytkownik zawsze dostaje v1)")


def canary(stable, candidate, X, y, user_ids) -> str:
    for pct in STEPS:
        mask = np.array([in_canary(u, pct) for u in user_ids])
        acc_c = accuracy_score(y[mask], candidate.predict(X[mask]))
        acc_s = accuracy_score(y[~mask], stable.predict(X[~mask])) if (~mask).any() else acc_c
        ok = acc_c >= acc_s - MAX_DEGRADATION
        print(f"  [canary {pct:>4.0%}] n={mask.sum():>5}  acc_canary={acc_c:.3f}  "
              f"acc_stable={acc_s:.3f}  -> {'OK' if ok else 'ROLLBACK'}")
        if not ok:
            return "ROLLBACK – ruch wraca w 100% do v1"
    return "PROMOTED – v2 obsługuje 100% ruchu"


def main() -> None:
    X, y = make_classification(n_samples=12_000, n_features=8, n_informative=5,
                               class_sep=0.8, random_state=7)
    X_tr, y_tr, X_live, y_live = X[:6000], y[:6000], X[6000:], y[6000:]
    user_ids = np.arange(100_000, 100_000 + len(X_live))

    v1 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    v2 = GradientBoostingClassifier(random_state=7).fit(X_tr, y_tr)

    print("Sticky routing (5%): użytkownik 100123 ->",
          "canary" if in_canary(100123, 0.05) else "stable", "(za każdym razem tak samo)")

    for name, cand in [("A: v2 lepszy", v2), ("B: v2 z błędem cech", BuggyModel(v2))]:
        print(f"\nScenariusz {name}")
        shadow(v1, cand, X_live)
        print("  Wynik:", canary(v1, cand, X_live, y_live, user_ids))


if __name__ == "__main__":
    main()
