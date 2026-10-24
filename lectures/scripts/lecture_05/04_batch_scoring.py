"""Wykład 5, sekcja 6 – Batch scoring: chunki, partycje, idempotencja.

Generuje syntetyczną bazę klientów (Parquet), trenuje model, a następnie
uruchamia nocny scoring w chunkach. Wyniki trafiają do partycji
`outputs/scores/score_date=YYYY-MM-DD/` – ponowne uruchomienie dla tej
samej daty NADPISUJE partycję (idempotencja), zamiast dublować wiersze.
Dodatkowo zapisuje metadane (wersja modelu, liczba wierszy, rozkład ryzyka).

Uruchomienie:
    python lectures/scripts/lecture_05/04_batch_scoring.py [--rows 200000] [--date 2026-10-24]
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = Path(__file__).parent / "outputs"
FEATURES = ["age", "income", "tenure_months", "num_products"]
MODEL_VERSION = "churn-hgb-v1"


def make_customers(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user_id": np.arange(n),
        "age": rng.integers(18, 80, n),
        "income": rng.normal(6000, 2000, n).clip(0).round(2),
        "tenure_months": rng.integers(0, 120, n),
        "num_products": rng.integers(1, 6, n),
    })
    logit = 1.0 - 0.03 * df.tenure_months - 0.4 * df.num_products
    df["churn"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return df


def score(input_path: Path, model, score_date: str, chunk: int) -> dict:
    part_dir = OUT / "scores" / f"score_date={score_date}"
    if part_dir.exists():
        shutil.rmtree(part_dir)  # idempotencja: nadpisz całą partycję
    part_dir.mkdir(parents=True)

    df = pd.read_parquet(input_path, columns=["user_id", *FEATURES])  # column pruning
    counts = {"low": 0, "medium": 0, "high": 0}
    for i, start in enumerate(range(0, len(df), chunk)):
        b = df.iloc[start:start + chunk]
        p = model.predict_proba(b[FEATURES])[:, 1]
        risk = pd.cut(p, bins=[0, 0.3, 0.7, 1.0], labels=["low", "medium", "high"],
                      include_lowest=True)  # bez tego p == 0.0 dałoby NaN!
        out = pd.DataFrame({"user_id": b.user_id.values, "churn_probability": p.round(4),
                            "risk_level": risk.astype(str), "model_version": MODEL_VERSION})
        out.to_parquet(part_dir / f"part-{i:04d}.parquet", index=False)
        for k, v in out.risk_level.value_counts().items():
            counts[k] += int(v)
    meta = {"score_date": score_date, "model_version": MODEL_VERSION,
            "rows": len(df), "risk_counts": counts}
    (part_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--date", default="2026-10-24")
    parser.add_argument("--chunk", type=int, default=50_000)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    customers = make_customers(args.rows)
    inp = OUT / "customers.parquet"
    customers.drop(columns="churn").to_parquet(inp, index=False)

    model = HistGradientBoostingClassifier(random_state=42).fit(
        customers[FEATURES].iloc[:50_000], customers.churn.iloc[:50_000])
    joblib.dump(model, OUT / "churn_model.joblib")
    model = joblib.load(OUT / "churn_model.joblib")  # jak w prawdziwym jobie

    for run in (1, 2):
        t0 = time.perf_counter()
        meta = score(inp, model, args.date, args.chunk)
        total = len(pd.read_parquet(OUT / "scores"))
        print(f"Run {run}: {meta['rows']:,} wierszy w {time.perf_counter() - t0:.2f}s, "
              f"ryzyko={meta['risk_counts']}, wierszy w tabeli wyników={total:,}")
    print("\nPo dwóch uruchomieniach liczba wierszy się nie podwoiła -> job jest idempotentny.")
    print(f"Wyniki: {OUT / 'scores'}")


if __name__ == "__main__":
    main()
