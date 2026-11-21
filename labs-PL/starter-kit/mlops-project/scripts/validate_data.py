"""Validate a Parquet data batch and save a machine-readable report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.validation import validate_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/train.parquet")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/validation.json"))
    args = parser.parse_args()
    report = validate_data(pd.read_parquet(args.input))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    for result in report.results:
        print(
            f"{'PASS' if result.passed else result.severity.upper()}: {result.name} — {result.message}"
        )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
