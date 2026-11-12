"""Wykład 6, sekcja 1 i 6 – Data drift vs concept drift na strumieniu danych.

Symuluje 12 tygodni produkcji modelu klasyfikacyjnego:
  * tygodnie 1–4:  stabilnie,
  * tygodnie 5–8:  data drift (zmienia się P(X), relacja X→Y bez zmian),
  * tygodnie 9–12: concept drift (P(X) jak w treningu, ale zmienia się P(Y|X)).
Dla każdego tygodnia liczy PSI cech, PSI predykcji i AUC (etykiety
„opóźnione" – dostępne dopiero po tygodniu). Pokazuje, że PSI cech nie wykryje
concept driftu, a spadek AUC nie zawsze towarzyszy data driftowi.
Na końcu stosuje prostą politykę retrainingu z quality gate.

Uruchomienie:
    python lectures/scripts/lecture_06/02_concept_drift_stream.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-4
PSI_ALERT = 0.2
AUC_DROP_ALERT = 0.05


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), EPS, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def make_batch(rng, n: int, x_shift: float = 0.0, w=(1.5, -1.0)):
    X = rng.normal(0, 1, (n, 2)) + np.array([x_shift, 0.0])
    logits = w[0] * X[:, 0] + w[1] * X[:, 1]
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return X, y


def main() -> None:
    rng = np.random.default_rng(42)
    X_train, y_train = make_batch(rng, 5_000)
    model = LogisticRegression().fit(X_train, y_train)
    ref_pred = model.predict_proba(X_train)[:, 1]
    baseline_auc = roc_auc_score(y_train, ref_pred)
    print(f"Baseline AUC (trening): {baseline_auc:.3f}\n")

    print(f"{'tydz.':>5} {'scenariusz':<14} {'PSI x0':>7} {'PSI x1':>7} "
          f"{'PSI ŷ':>7} {'AUC':>6}  alerty")
    for week in range(1, 13):
        if week <= 4:
            scenario, X, y = "stabilnie", *make_batch(rng, 1_000)
        elif week <= 8:
            scenario, X, y = "data drift", *make_batch(rng, 1_000, x_shift=1.0)
        else:
            scenario, X, y = "concept drift", *make_batch(rng, 1_000, w=(-0.5, -1.0))

        pred = model.predict_proba(X)[:, 1]
        psi_x = [psi(X_train[:, i], X[:, i]) for i in range(2)]
        psi_pred = psi(ref_pred, pred)
        auc = roc_auc_score(y, pred)  # w praktyce znane z opóźnieniem

        alerts = []
        if max(psi_x) > PSI_ALERT:
            alerts.append("DATA_DRIFT")
        if psi_pred > PSI_ALERT:
            alerts.append("PRED_DRIFT")
        if baseline_auc - auc > AUC_DROP_ALERT:
            alerts.append("PERF_DROP→RETRAIN?")
        print(f"{week:>5} {scenario:<14} {psi_x[0]:>7.3f} {psi_x[1]:>7.3f} "
              f"{psi_pred:>7.3f} {auc:>6.3f}  {', '.join(alerts) or '-'}")

    # --- Retraining z quality gate (na danych z nowym konceptem) ---
    print("\nRetraining na świeżych danych (tydzień 13) + quality gate:")
    X_new, y_new = make_batch(rng, 3_000, w=(-0.5, -1.0))
    X_hold, y_hold = make_batch(rng, 1_000, w=(-0.5, -1.0))
    challenger = LogisticRegression().fit(X_new, y_new)
    auc_champion = roc_auc_score(y_hold, model.predict_proba(X_hold)[:, 1])
    auc_challenger = roc_auc_score(y_hold, challenger.predict_proba(X_hold)[:, 1])
    promote = auc_challenger >= auc_champion + 0.01 and auc_challenger >= 0.70
    print(f"  champion AUC={auc_champion:.3f}  challenger AUC={auc_challenger:.3f}"
          f"  → {'PROMUJ challengera' if promote else 'zostaw championa'}")


if __name__ == "__main__":
    main()
