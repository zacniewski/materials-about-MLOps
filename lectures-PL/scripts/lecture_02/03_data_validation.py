"""Wykład 2, sekcja 6 – Walidacja danych.

Lekka implementacja idei Great Expectations w czystym pandas (bez dodatkowych
zależności): definiujemy "oczekiwania" wobec danych i uruchamiamy je jako
bramkę jakości (quality gate) w pipeline. Skrypt kończy się kodem 1, jeśli
walidacja nie przejdzie – tak jak krok w CI/CD.

Uruchomienie:
    python lectures/scripts/lecture_02/03_data_validation.py           # dane poprawne
    python lectures/scripts/lecture_02/03_data_validation.py --corrupt # dane z błędami
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Expectation:
    name: str
    check: Callable[[pd.DataFrame], pd.Series | bool]
    mostly: float = 1.0  # odsetek wierszy, które muszą spełnić warunek


EXPECTATIONS = [
    Expectation("kolumny zgodne ze schematem",
                lambda df: set(df.columns) == {"age", "income", "city", "churn"}),
    Expectation("age nie jest null", lambda df: df["age"].notna()),
    Expectation("age w zakresie [18, 100]", lambda df: df["age"].between(18, 100), mostly=0.99),
    Expectation("income > 0", lambda df: df["income"] > 0, mostly=0.95),
    Expectation("city ze słownika", lambda df: df["city"].isin(["Warszawa", "Kraków", "Gdańsk"])),
    Expectation("churn ∈ {0, 1}", lambda df: df["churn"].isin([0, 1])),
    Expectation("min. 1000 wierszy", lambda df: len(df) >= 1000),
]


def validate(df: pd.DataFrame) -> bool:
    all_ok = True
    for exp in EXPECTATIONS:
        try:
            result = exp.check(df)
            ratio = float(result.mean()) if isinstance(result, pd.Series) else float(bool(result))
        except KeyError as e:
            ratio = 0.0
            print(f"   brak kolumny {e}")
        ok = ratio >= exp.mostly
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {exp.name:<30} spełnione: {ratio:7.2%} (wymagane {exp.mostly:.0%})")
    return all_ok


def make_data(corrupt: bool, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "income": rng.normal(5000, 1000, n).clip(100),
        "city": rng.choice(["Warszawa", "Kraków", "Gdańsk"], n),
        "churn": rng.integers(0, 2, n),
    })
    if corrupt:
        df.loc[:49, "age"] = np.nan              # braki
        df.loc[50:99, "age"] = 150               # wartości spoza zakresu
        df.loc[100:299, "income"] = -10          # 10% błędnych dochodów
        df.loc[300, "city"] = "Berlin"           # wartość spoza słownika
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt", action="store_true", help="wstrzyknij błędy do danych")
    args = parser.parse_args()

    df = make_data(args.corrupt)
    print(f"Walidacja {len(df)} wierszy (corrupt={args.corrupt})\n")
    if validate(df):
        print("\n🟢 Walidacja OK – dane mogą trafić do treningu.")
    else:
        print("\n🔴 Walidacja NIE przeszła – pipeline zatrzymany.")
        sys.exit(1)


if __name__ == "__main__":
    main()
