"""Launch a small, comparable MLflow parameter sweep."""

import argparse
import os

from src.models.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Run only two configurations"
    )
    parser.add_argument("--experiment", default="churn-baseline")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    configurations = (
        [(50, 5), (100, 8)] if args.quick else [(50, 4), (100, 6), (150, 8), (200, 12)]
    )
    for n_estimators, max_depth in configurations:
        run_id = run_experiment(
            n_estimators, max_depth, args.experiment, args.tracking_uri
        )
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: run_id={run_id}")


if __name__ == "__main__":
    main()
