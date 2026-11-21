"""Check whether the files and outputs expected in Labs 1–3 are present."""

from pathlib import Path

CHECKS = {
    "Lab 1: raw data": Path("data/raw/customers.parquet"),
    "Lab 1: DVC pipeline": Path("dvc.yaml"),
    "Lab 1: processed train set": Path("data/processed/train.parquet"),
    "Lab 1: processed test set": Path("data/processed/test.parquet"),
    "Lab 1: baseline model": Path("models/churn_model.pkl"),
    "Lab 2: experiment runner": Path("scripts/run_experiments.py"),
    "Lab 3: validation module": Path("src/data/validation.py"),
    "Lab 3: feature store": Path("src/features/feature_store.py"),
}


def main() -> int:
    missing = []
    for label, path in CHECKS.items():
        passed = path.exists()
        print(f"{'[OK]' if passed else '[ ]'} {label}: {path}")
        if not passed:
            missing.append(label)
    print(f"\n{len(CHECKS) - len(missing)}/{len(CHECKS)} checkpoints complete.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
