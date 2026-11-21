"""Unit tests for the Lab 1 data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_data


@pytest.fixture
def raw_data(tmp_path):
    """Create reproducible raw customer data in Parquet format."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "age": rng.uniform(18, 80, 100),
            "income": rng.uniform(20_000, 100_000, 100),
            "tenure_months": rng.integers(1, 120, 100),
            "num_products": rng.integers(1, 5, 100),
            "has_credit_card": rng.integers(0, 2, 100),
            "is_active": rng.integers(0, 2, 100),
            "balance": rng.uniform(0, 50_000, 100),
            "num_transactions_30d": rng.integers(0, 50, 100),
            "churn": rng.integers(0, 2, 100),
        }
    )
    path = tmp_path / "raw.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_preprocess_creates_train_and_test_files(raw_data, tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    preprocess_data(str(raw_data), str(train_path), str(test_path))

    assert train_path.exists()
    assert test_path.exists()


def test_preprocess_preserves_split_ratio(raw_data, tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    preprocess_data(str(raw_data), str(train_path), str(test_path), test_size=0.2)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.05


def test_preprocess_adds_engineered_features(raw_data, tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    preprocess_data(str(raw_data), str(train_path), str(test_path))

    train = pd.read_parquet(train_path)
    assert {
        "income_per_age",
        "balance_per_product",
        "is_long_tenure",
    }.issubset(train.columns)


def test_preprocess_removes_invalid_rows(raw_data, tmp_path):
    frame = pd.read_parquet(raw_data)
    frame.loc[0, "age"] = 100
    frame.loc[1, "income"] = 0
    frame.loc[2, "customer_id"] = frame.loc[3, "customer_id"]
    frame.to_parquet(raw_data, index=False)
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    stats = preprocess_data(str(raw_data), str(train_path), str(test_path))

    assert stats["total_rows"] == 97
    assert stats["total_rows"] == stats["train_rows"] + stats["test_rows"]


@pytest.mark.parametrize("seed", [21, 42])
def test_preprocess_split_is_reproducible(raw_data, tmp_path, seed):
    first_train = tmp_path / f"train-{seed}.parquet"
    first_test = tmp_path / f"test-{seed}.parquet"
    second_train = tmp_path / f"train-again-{seed}.parquet"
    second_test = tmp_path / f"test-again-{seed}.parquet"

    preprocess_data(str(raw_data), str(first_train), str(first_test), seed=seed)
    preprocess_data(str(raw_data), str(second_train), str(second_test), seed=seed)

    pd.testing.assert_frame_equal(
        pd.read_parquet(first_train), pd.read_parquet(second_train)
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first_test), pd.read_parquet(second_test)
    )
