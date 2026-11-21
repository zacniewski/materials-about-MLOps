"""Check required phase-1 deliverables and the quality-gate report."""

import json
from pathlib import Path

REQUIRED = [
    Path("params.yaml"),
    Path("dvc.yaml"),
    Path("data/raw/customers.parquet"),
    Path("data/processed/train.parquet"),
    Path("data/processed/test.parquet"),
    Path("models/churn_model.pkl"),
    Path("reports/metrics/data_stats.json"),
    Path("reports/metrics/train_metrics.json"),
    Path("reports/validation.json"),
    Path("reports/evaluation_metrics.json"),
]


def main() -> int:
    missing = [path for path in REQUIRED if not path.exists()]
    for path in REQUIRED:
        print(f"{'[OK]' if path.exists() else '[ ]'} {path}")
    if missing:
        print(
            f"\nMissing {len(missing)} required deliverable(s). Run the setup steps and `dvc repro`."
        )
        return 1
    metrics = json.loads(
        Path("reports/evaluation_metrics.json").read_text(encoding="utf-8")
    )
    passed = metrics.get("quality_gate_passed") is True
    print(
        f"\nROC-AUC {metrics.get('roc_auc')} / minimum {metrics.get('min_roc_auc')}: {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
