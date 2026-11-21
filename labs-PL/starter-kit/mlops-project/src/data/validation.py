"""Small, dependency-light data quality checks used in the lab pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class ValidationResult:
    name: str
    passed: bool
    severity: str
    message: str


@dataclass
class ValidationReport:
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        return all(
            result.passed for result in self.results if result.severity == "error"
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "results": [asdict(item) for item in self.results],
        }


def validate_data(frame: pd.DataFrame, target: str = "churn") -> ValidationReport:
    """Run schema, null, identifier, target-domain and size checks."""
    required = {
        "customer_id",
        "age",
        "income",
        "tenure_months",
        "num_products",
        "has_credit_card",
        "is_active",
        "balance",
        "num_transactions_30d",
        "income_per_age",
        "balance_per_product",
        "is_long_tenure",
        target,
    }
    rules: list[tuple[str, Callable[[], bool], str, str]] = [
        (
            "required_columns",
            lambda: required.issubset(frame.columns),
            "error",
            "Required columns are present",
        ),
        (
            "non_empty",
            lambda: not frame.empty,
            "error",
            "Dataset contains at least one row",
        ),
        (
            "no_missing_values",
            lambda: not frame[sorted(required & set(frame.columns))].isna().any().any(),
            "error",
            "Required fields have no missing values",
        ),
        (
            "unique_customer_id",
            lambda: "customer_id" in frame and frame["customer_id"].is_unique,
            "error",
            "Customer identifiers are unique",
        ),
        (
            "churn_binary",
            lambda: (
                target in frame
                and set(frame[target].dropna().unique()).issubset({0, 1})
            ),
            "error",
            "Target contains only 0 and 1",
        ),
        (
            "age_range",
            lambda: "age" in frame and frame["age"].between(18, 90).all(),
            "error",
            "Age is in the expected range [18, 90]",
        ),
        (
            "positive_income",
            lambda: "income" in frame and (frame["income"] > 0).all(),
            "error",
            "Income is positive",
        ),
        (
            "positive_product_count",
            lambda: "num_products" in frame and (frame["num_products"] > 0).all(),
            "error",
            "Product count is positive",
        ),
        (
            "minimum_sample_size",
            lambda: len(frame) >= 20,
            "warning",
            "At least 20 rows are recommended",
        ),
    ]
    results = [
        ValidationResult(name, bool(check()), severity, message)
        for name, check, severity, message in rules
    ]
    return ValidationReport(results)
