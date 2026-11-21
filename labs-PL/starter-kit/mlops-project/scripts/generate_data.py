"""Generuje przykładowe dane dla projektu churn prediction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_dataset(
    n_samples: int = 10_000,
    seed: int = 42,
    output_path: str = "data/raw/customers.parquet",
) -> pd.DataFrame:
    """
    Generuje syntetyczny zbiór danych o churnie klientów.

    Args:
        n_samples: liczba próbek
        seed: ziarno losowości
        output_path: ścieżka do zapisu

    Returns:
        DataFrame z danymi klientów
    """
    np.random.seed(seed)

    # Cechy demograficzne
    age = np.random.normal(40, 12, n_samples).clip(18, 90)
    income = np.random.lognormal(10.8, 0.5, n_samples).clip(15000, 500000)

    # Cechy produktowe
    tenure_months = np.random.exponential(24, n_samples).clip(1, 240).astype(int)
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.15, 0.1])
    has_credit_card = np.random.binomial(1, 0.7, n_samples)
    is_active = np.random.binomial(1, 0.8, n_samples)

    # Cechy transakcyjne
    balance = np.random.lognormal(9, 1.5, n_samples) * is_active
    num_transactions_30d = np.random.poisson(15, n_samples) * is_active

    # Etykieta (churn) - zależna od cech
    churn_prob = (
        0.05
        + 0.15 * (age > 60)
        + 0.20 * (tenure_months < 6)
        + 0.10 * (num_products == 1)
        - 0.10 * (is_active)
        - 0.05 * (has_credit_card)
        + 0.15 * (balance < 1000)
    ).clip(0.02, 0.95)

    churn = np.random.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame(
        {
            "customer_id": range(1, n_samples + 1),
            "age": age.round(1),
            "income": income.round(2),
            "tenure_months": tenure_months,
            "num_products": num_products,
            "has_credit_card": has_credit_card,
            "is_active": is_active,
            "balance": balance.round(2),
            "num_transactions_30d": num_transactions_30d,
            "churn": churn,
        }
    )

    # Zapisz dane
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"✅ Wygenerowano {n_samples:,} rekordów")
    print(f"   Churn rate: {churn.mean():.2%}")
    print(f"   Zapisano do: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/customers.parquet")
    args = parser.parse_args()

    generate_churn_dataset(args.n_samples, args.seed, args.output)
