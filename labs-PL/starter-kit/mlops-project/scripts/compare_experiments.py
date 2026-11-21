"""Print the best MLflow runs by ROC-AUC."""

import argparse
import os

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="churn-baseline")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"Experiment {args.experiment!r} does not exist")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=args.top,
    )
    if runs.empty:
        print("No runs found")
        return
    columns = [
        "run_id",
        "metrics.roc_auc",
        "metrics.f1",
        "params.n_estimators",
        "params.max_depth",
    ]
    print(runs[columns].to_string(index=False))


if __name__ == "__main__":
    main()
