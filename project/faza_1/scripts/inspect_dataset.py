"""Print a compact schema and quality summary for a CSV or Parquet dataset."""

import argparse
from pathlib import Path

import pandas as pd


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a supported tabular file based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Supported formats are CSV (.csv) and Parquet (.parquet, .pq).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or Parquet file")
    parser.add_argument("--target", help="Target column to summarize")
    parser.add_argument("--id", dest="id_column", help="Identifier column to check")
    args = parser.parse_args()

    frame = read_dataset(args.input)
    for label, column in (("target", args.target), ("ID", args.id_column)):
        if column and column not in frame.columns:
            parser.error(f"{label} column {column!r} is not present in {args.input}.")

    print(f"Dataset: {args.input}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns ({len(frame.columns)}): {', '.join(map(str, frame.columns))}")

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    print(f"Missing values: {int(frame.isna().sum().sum()):,}")
    if missing.empty:
        print("  None")
    else:
        for column, count in missing.items():
            print(f"  {column}: {count:,}")

    if args.id_column:
        duplicate_count = int(frame[args.id_column].duplicated().sum())
        missing_id_count = int(frame[args.id_column].isna().sum())
        print(
            f"ID {args.id_column}: {duplicate_count:,} duplicate row(s) after first occurrence; "
            f"{missing_id_count:,} missing value(s)"
        )
    else:
        print("ID duplicates: not checked (pass --id COLUMN to check)")

    if args.target:
        counts = frame[args.target].value_counts(dropna=False)
        print(f"Target distribution ({args.target}):")
        for value, count in counts.items():
            percentage = count / len(frame) * 100 if len(frame) else 0.0
            print(f"  {value}: {count:,} ({percentage:.2f}%)")
    else:
        print("Target distribution: not checked (pass --target COLUMN to check)")


if __name__ == "__main__":
    main()