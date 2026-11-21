"""Przetwarzanie danych surowych."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def preprocess_data(
    input_path: str,
    output_train: str,
    output_test: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Przetwarza surowe dane i dzieli na train/test."""
    from sklearn.model_selection import train_test_split

    print(f"Wczytywanie danych z {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Wczytano {len(df):,} wierszy")

    # Czyszczenie
    initial_size = len(df)
    df = df.drop_duplicates(subset=["customer_id"])
    df = df[df["age"].between(18, 90)]
    df = df[df["income"] > 0]
    removed = initial_size - len(df)
    print(f"  Usunięto {removed} błędnych wierszy")

    # Feature engineering
    df["income_per_age"] = df["income"] / df["age"]
    df["balance_per_product"] = df["balance"] / df["num_products"]
    df["is_long_tenure"] = (df["tenure_months"] > 24).astype(int)

    # Podział train/test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["churn"]
    )

    # Zapis
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_train, index=False)
    test_df.to_parquet(output_test, index=False)

    stats = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "churn_rate_train": float(train_df["churn"].mean()),
        "churn_rate_test": float(test_df["churn"].mean()),
        "n_features": df.shape[1] - 2,  # bez customer_id i churn
    }

    print(f"  Train: {stats['train_rows']:,}, Test: {stats['test_rows']:,}")
    print(f"  Churn rate: {stats['churn_rate_train']:.2%}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/customers.parquet")
    parser.add_argument("--output-train", default="data/processed/train.parquet")
    parser.add_argument("--output-test", default="data/processed/test.parquet")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    # Parametry czytamy z params.yaml — DVC śledzi data.test_size i data.seed,
    # więc ich zmiana musi faktycznie wpływać na wynik etapu
    with open(args.params) as f:
        data_params = yaml.safe_load(f).get("data", {})

    stats = preprocess_data(
        args.input,
        args.output_train,
        args.output_test,
        test_size=data_params.get("test_size", 0.2),
        seed=data_params.get("seed", 42),
    )

    # Zapisz statystyki
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Statystyki zapisane do reports/metrics/data_stats.json")
