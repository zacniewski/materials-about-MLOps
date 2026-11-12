"""Lecture 2, section 10 – Common pitfalls: data leakage.

Two examples of information leakage:
1. Feature selection on the whole dataset before the split/CV – on random noise
   (y independent of X!) it yields deceptively high accuracy.
2. A feature "from the future" (target leakage) – a feature derived from the label.

Usage:
    python lectures-EN/scripts/lecture_02/04_data_leakage.py
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5000))   # pure noise
    y = rng.integers(0, 2, 200)        # random labels – true accuracy ≈ 0.5

    print("=== 1. Preprocessing/feature selection before the split ===")
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)  # ❌ sees the labels of the whole dataset
    bad = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()
    print(f"❌ WRONG (selection on all data):   CV accuracy = {bad:.3f}")

    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    good = cross_val_score(pipe, X, y, cv=5).mean()           # ✅ fitted within each fold
    print(f"✅ RIGHT (selection in Pipeline):     CV accuracy = {good:.3f}")
    print("➡️  The data is noise – the honest result is ~0.5. The 'WRONG' result is an illusion.\n")

    print("=== 2. Target leakage (feature from the future) ===")
    X2 = rng.normal(size=(1000, 5))
    y2 = (X2[:, 0] + rng.normal(scale=2, size=1000) > 0).astype(int)
    leak = (y2 + rng.normal(scale=0.3, size=1000)).reshape(-1, 1)  # e.g. "number of days since cancellation"
    base = cross_val_score(LogisticRegression(), X2, y2, cv=5).mean()
    leaked = cross_val_score(LogisticRegression(), np.hstack([X2, leak]), y2, cv=5).mean()
    print(f"Without leaking feature:{base:.3f}")
    print(f"With leaking feature:   {leaked:.3f}  ← this feature will not be available in production!")


if __name__ == "__main__":
    main()
