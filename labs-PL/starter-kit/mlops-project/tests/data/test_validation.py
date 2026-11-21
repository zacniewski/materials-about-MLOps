"""Tests for Lab 1 customer data validation rules."""

import pandas as pd

from src.data.validation import validate_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(100),
            "age": [35.0] * 100,
            "income": [50_000.0] * 100,
            "tenure_months": [12] * 100,
            "num_products": [2] * 100,
            "has_credit_card": [1] * 100,
            "is_active": [1] * 100,
            "balance": [1_000.0] * 100,
            "num_transactions_30d": [15] * 100,
            "income_per_age": [50_000.0 / 35] * 100,
            "balance_per_product": [500.0] * 100,
            "is_long_tenure": [0] * 100,
            "churn": [0, 1] * 50,
        }
    )


def test_valid_frame_passes_error_rules():
    report = validate_data(valid_frame())
    assert report.passed
    assert all(result.passed for result in report.results)


def test_invalid_age_fails_report():
    frame = valid_frame()
    frame.loc[0, "age"] = 91
    report = validate_data(frame)
    assert not report.passed
    assert not next(item for item in report.results if item.name == "age_range").passed


def test_nonpositive_income_fails_report():
    frame = valid_frame()
    frame.loc[0, "income"] = 0
    report = validate_data(frame)
    assert not report.passed
    assert not next(
        item for item in report.results if item.name == "positive_income"
    ).passed


def test_missing_lab1_column_fails_report():
    frame = valid_frame().drop(columns="income")
    report = validate_data(frame)
    assert not report.passed
    assert not next(
        item for item in report.results if item.name == "required_columns"
    ).passed
