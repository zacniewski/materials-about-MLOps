"""Wykład 7, sekcja 4 – Quality Gates: bramka jakości champion vs challenger.

Trenuje model produkcyjny (champion) i kandydata (challenger), liczy metryki
(AUC, F1, latencję, różnicę AUC między grupami) i sprawdza:
  * progi bezwzględne (np. AUC >= 0.75),
  * progi względne (kandydat nie może być gorszy od championa o > 0.01 AUC),
  * fairness (różnica AUC między grupami <= 0.10).
Raport zapisywany jest do outputs/quality_gate_report.json, a kod wyjścia
(0 = PASSED, 1 = FAILED) pozwala zatrzymać pipeline CI/CD.

Uruchomienie:
    python lectures/scripts/lecture_07/02_quality_gate.py            # dobry kandydat
    python lectures/scripts/lecture_07/02_quality_gate.py --bad      # kandydat z błędem
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).parent / "outputs"
SEED = 42


@dataclass
class GateConfig:
    min_auc: float = 0.75
    min_f1: float = 0.60
    max_latency_ms: float = 50.0  # na batch 1000 rekordów
    max_auc_drop_vs_champion: float = 0.01
    max_group_auc_diff: float = 0.10


def evaluate(model, X, y, group) -> dict:
    start = time.perf_counter()
    model.predict_proba(X[:1000])
    latency_ms = (time.perf_counter() - start) * 1000
    proba = model.predict_proba(X)[:, 1]
    group_aucs = [roc_auc_score(y[group == g], proba[group == g]) for g in np.unique(group)]
    return {
        "auc_roc": round(float(roc_auc_score(y, proba)), 4),
        "f1_score": round(float(f1_score(y, proba >= 0.5)), 4),
        "latency_ms": round(latency_ms, 2),
        "group_auc_diff": round(float(max(group_aucs) - min(group_aucs)), 4),
    }


def check_gate(cand: dict, champ: dict, cfg: GateConfig) -> list[str]:
    failures = []
    if cand["auc_roc"] < cfg.min_auc:
        failures.append(f"AUC {cand['auc_roc']} < {cfg.min_auc}")
    if cand["f1_score"] < cfg.min_f1:
        failures.append(f"F1 {cand['f1_score']} < {cfg.min_f1}")
    if cand["latency_ms"] > cfg.max_latency_ms:
        failures.append(f"latencja {cand['latency_ms']} ms > {cfg.max_latency_ms} ms")
    drop = champ["auc_roc"] - cand["auc_roc"]
    if drop > cfg.max_auc_drop_vs_champion:
        failures.append(f"regresja vs champion: AUC spadło o {drop:.4f}")
    if cand["group_auc_diff"] > cfg.max_group_auc_diff:
        failures.append(f"fairness: różnica AUC między grupami {cand['group_auc_diff']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad", action="store_true",
                        help="symuluj zepsuty kandydat (przetasowane etykiety)")
    args = parser.parse_args()

    X, y = make_classification(n_samples=6000, n_features=12, n_informative=6,
                               weights=[0.6], random_state=SEED)
    group = (X[:, 0] > 0).astype(int)  # np. dwie grupy klientów
    X_tr, X_te, y_tr, y_te, _, g_te = train_test_split(X, y, group, test_size=0.3,
                                                       random_state=SEED)

    champion = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    y_cand = y_tr.copy()
    if args.bad:
        # typowy błąd w pipeline: etykiety rozjechały się z cechami (np. zły join)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y_cand), size=int(0.9 * len(y_cand)), replace=False)
        y_cand[idx] = rng.permutation(y_cand[idx])
    challenger = GradientBoostingClassifier(random_state=SEED).fit(X_tr, y_cand)

    cfg = GateConfig()
    champ_m = evaluate(champion, X_te, y_te, g_te)
    cand_m = evaluate(challenger, X_te, y_te, g_te)
    failures = check_gate(cand_m, champ_m, cfg)

    print(f"{'metryka':<16}{'champion':>10}{'challenger':>12}")
    for k in champ_m:
        print(f"{k:<16}{champ_m[k]:>10}{cand_m[k]:>12}")

    OUT_DIR.mkdir(exist_ok=True)
    report = {"config": asdict(cfg), "champion": champ_m, "challenger": cand_m,
              "passed": not failures, "failures": failures}
    (OUT_DIR / "quality_gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if failures:
        print("\n❌ Quality Gate: FAILED")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("\n✅ Quality Gate: PASSED – kandydat może zostać promowany do Staging")


if __name__ == "__main__":
    main()
