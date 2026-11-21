"""Quality-gate tests; run after `dvc repro` has produced the evaluation report."""

import json
from pathlib import Path


def test_evaluation_report_meets_quality_gate():
    report_path = Path("reports/evaluation_metrics.json")
    assert report_path.exists(), "Run `dvc repro` before the model quality tests"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["roc_auc"] >= report["min_roc_auc"]
    assert report["quality_gate_passed"] is True


def test_evaluation_report_contains_required_metrics():
    report = json.loads(
        Path("reports/evaluation_metrics.json").read_text(encoding="utf-8")
    )
    assert {"accuracy", "f1", "roc_auc"}.issubset(report)
    assert all(0 <= report[name] <= 1 for name in ("accuracy", "f1", "roc_auc"))
