"""Optional demonstration of the same null check with Great Expectations."""

import argparse
from pathlib import Path

import great_expectations as gx
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/train.parquet")
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name="customer_data")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})
    expectation = gx.expectations.ExpectColumnValuesToNotBeNull(column="churn")
    result = batch.validate(expectation)
    print(result)
    if not result.success:
        raise SystemExit("Great Expectations reported a failed expectation")


if __name__ == "__main__":
    main()
