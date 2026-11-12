"""Lecture 2, section 6 – Data validation.

A lightweight implementation of the Great Expectations idea in plain pandas (no extra
dependencies): we define "expectations" for the data and run them as a
quality gate in the pipeline. The script exits with code 1 if
validation fails – just like a CI/CD step.

Usage:
    python lectures-EN/scripts/lecture_02/03_data_validation.py           # valid data
    python lectures-EN/scripts/lecture_02/03_data_validation.py --corrupt # data with errors
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
    mostly: float = 1.0  # fraction of rows that must satisfy the condition


EXPECTATIONS = [
    Expectation("columns match the schema",
                lambda df: set(df.columns) == {"age", "income", "city", "churn"}),
    Expectation("age is not null", lambda df: df["age"].notna()),
    Expectation("age in range [18, 100]", lambda df: df["age"].between(18, 100), mostly=0.99),
    Expectation("income > 0", lambda df: df["income"] > 0, mostly=0.95),
    Expectation("city from the dictionary", lambda df: df["city"].isin(["Warszawa", "Kraków", "Gdańsk"])),
    Expectation("churn ∈ {0, 1}", lambda df: df["churn"].isin([0, 1])),
    Expectation("at least 1000 rows", lambda df: len(df) >= 1000),
]


def validate(df: pd.DataFrame) -> bool:
    all_ok = True
    for exp in EXPECTATIONS:
        try:
            result = exp.check(df)
            ratio = float(result.mean()) if isinstance(result, pd.Series) else float(bool(result))
        except KeyError as e:
            ratio = 0.0
            print(f"   missing column {e}")
        ok = ratio >= exp.mostly
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {exp.name:<30} satisfied: {ratio:7.2%} (required {exp.mostly:.0%})")
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
        df.loc[:49, "age"] = np.nan              # missing values
        df.loc[50:99, "age"] = 150               # out-of-range values
        df.loc[100:299, "income"] = -10          # 10% invalid incomes
        df.loc[300, "city"] = "Berlin"           # value outside the dictionary
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt", action="store_true", help="inject errors into the data")
    args = parser.parse_args()

    df = make_data(args.corrupt)
    print(f"Validating {len(df)} rows (corrupt={args.corrupt})\n")
    if validate(df):
        print("\n🟢 Validation OK – the data can go to training.")
    else:
        print("\n🔴 Validation FAILED – pipeline stopped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
