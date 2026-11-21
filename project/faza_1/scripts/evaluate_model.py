"""Evaluate the saved baseline on the held-out test set and enforce a quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(
    model_path: Path,
    test_path: Path,
    target: str = "churn",
    min_roc_auc: float = 0.70,
) -> dict[str, float | bool]:
    """Return test metrics and whether the model meets the ROC-AUC threshold."""
    model = joblib.load(model_path)
    test = pd.read_parquet(test_path)
    features = getattr(model, "feature_names_in_", None)
    if features is None:
        features = [
            column for column in test.columns if column not in {target, "customer_id"}
        ]
    else:
        features = list(features)
    prediction = model.predict(test[features])
    probability = model.predict_proba(test[features])[:, 1]
    roc_auc = float(roc_auc_score(test[target], probability))
    return {
        "accuracy": float(accuracy_score(test[target], prediction)),
        "f1": float(f1_score(test[target], prediction, zero_division=0)),
        "roc_auc": roc_auc,
        "min_roc_auc": min_roc_auc,
        "quality_gate_passed": roc_auc >= min_roc_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/churn_model.pkl"))
    parser.add_argument(
        "--test", type=Path, default=Path("data/processed/test.parquet")
    )
    parser.add_argument("--target", default="churn")
    parser.add_argument("--min-roc-auc", type=float, default=0.70)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/evaluation_metrics.json")
    )
    args = parser.parse_args()
    result = evaluate_model(args.model, args.test, args.target, args.min_roc_auc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["quality_gate_passed"]:
        raise SystemExit("Model did not meet the minimum ROC-AUC quality gate")


if __name__ == "__main__":
    main()
