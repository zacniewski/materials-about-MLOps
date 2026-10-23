"""Pomocniczy generator danych dla 05_dvc_demo.sh.

Użycie: python _make_dataset.py <ścieżka.csv> <liczba_wierszy>
"""

import sys

import numpy as np
import pandas as pd

path, n = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(n)
pd.DataFrame({
    "age": rng.integers(18, 80, n),
    "income": rng.normal(5000, 1500, n).round(2),
    "churn": rng.integers(0, 2, n),
}).to_csv(path, index=False)
print(f"Zapisano {n} wierszy do {path}")
