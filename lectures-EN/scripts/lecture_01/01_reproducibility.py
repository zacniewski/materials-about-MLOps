"""Lecture 1, section 4.1 – Reproducibility.

Shows the difference between training without controlling randomness and training
with a fixed seed plus saving the experiment configuration.

Usage:
    python lectures-EN/scripts/lecture_01/01_reproducibility.py
"""

import hashlib
import json
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent / "outputs"


def train(seed: int | None) -> float:
    """Trains a RandomForest; seed=None means no control over randomness."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    X, y = make_classification(n_samples=2000, n_features=20, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model.fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


def main() -> None:
    print("❌ Without controlling randomness (3 runs):")
    for i in range(3):
        print(f"   run {i + 1}: accuracy = {train(seed=None):.4f}")

    print("\n✅ With fixed seed=42 (3 runs):")
    results = [train(seed=42) for _ in range(3)]
    for i, acc in enumerate(results):
        print(f"   run {i + 1}: accuracy = {acc:.4f}")
    assert len(set(results)) == 1, "Results should be identical!"

    config = {
        "seed": 42,
        "model": "RandomForestClassifier",
        "params": {"n_estimators": 50},
        "accuracy": results[0],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    config["config_hash"] = hashlib.md5(
        json.dumps(config["params"], sort_keys=True).encode()
    ).hexdigest()[:8]

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "experiment_config.json"
    path.write_text(json.dumps(config, indent=2))
    print(f"\n📝 Experiment configuration saved to: {path}")


if __name__ == "__main__":
    main()
