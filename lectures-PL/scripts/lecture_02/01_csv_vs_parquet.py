"""Wykład 2, sekcja 3 – Formaty danych: CSV vs Parquet.

Porównuje rozmiar pliku, czas zapisu/odczytu, zachowanie typów oraz
column pruning (odczyt tylko wybranych kolumn) dla CSV i Parquet.

Uruchomienie:
    python lectures/scripts/lecture_02/01_csv_vs_parquet.py [--rows 1000000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "outputs"


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    n = args.rows
    df = pd.DataFrame({
        "user_id": np.arange(n),
        "age": rng.integers(18, 80, n),
        "income": rng.normal(5000, 1500, n).round(2),
        "city": pd.Categorical(rng.choice(["Warszawa", "Kraków", "Gdańsk", "Wrocław"], n)),
        "signup_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
        "churn": rng.integers(0, 2, n).astype(bool),
    })
    print(f"Wygenerowano DataFrame: {n:,} wierszy × {df.shape[1]} kolumn\n")

    DATA_DIR.mkdir(exist_ok=True)
    csv_path, pq_path = DATA_DIR / "data.csv", DATA_DIR / "data.parquet"

    _, t_csv_w = timed(lambda: df.to_csv(csv_path, index=False))
    _, t_pq_w = timed(lambda: df.to_parquet(pq_path, compression="snappy", index=False))
    df_csv, t_csv_r = timed(lambda: pd.read_csv(csv_path))
    df_pq, t_pq_r = timed(lambda: pd.read_parquet(pq_path))
    _, t_csv_cols = timed(lambda: pd.read_csv(csv_path, usecols=["age", "income"]))
    _, t_pq_cols = timed(lambda: pd.read_parquet(pq_path, columns=["age", "income"]))

    mb = lambda p: p.stat().st_size / 1024**2  # noqa: E731
    report = pd.DataFrame({
        "CSV": [mb(csv_path), t_csv_w, t_csv_r, t_csv_cols],
        "Parquet": [mb(pq_path), t_pq_w, t_pq_r, t_pq_cols],
    }, index=["rozmiar [MB]", "zapis [s]", "odczyt całości [s]", "odczyt 2 kolumn [s]"])
    report["CSV/Parquet"] = report["CSV"] / report["Parquet"]
    print(report.round(3).to_string())

    print("\nTypy kolumn po odczycie:")
    print(pd.DataFrame({"oryginał": df.dtypes, "CSV": df_csv.dtypes, "Parquet": df_pq.dtypes}).to_string())
    print("\n➡️  CSV gubi typy (data → object, category → object); Parquet je zachowuje.")


if __name__ == "__main__":
    main()
