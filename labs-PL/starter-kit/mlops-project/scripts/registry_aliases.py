"""Register an MLflow run's model and point an alias at its new version."""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run ID printed by run_experiments.py")
    parser.add_argument("--name", default="churn-random-forest")
    parser.add_argument("--alias", default="candidate")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"runs:/{args.run_id}/model"
    version = mlflow.register_model(model_uri, args.name)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    client.set_registered_model_alias(args.name, args.alias, version.version)
    print(f"{args.name} version {version.version} is now @{args.alias}")


if __name__ == "__main__":
    main()
