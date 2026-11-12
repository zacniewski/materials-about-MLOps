"""Wykład 2, sekcja 10 – Typowe pułapki: data leakage.

Dwa przykłady wycieku informacji:
1. Selekcja cech na całym zbiorze przed podziałem/CV – na losowym szumie
   (y niezależne od X!) daje złudnie wysoką dokładność.
2. Cecha "z przyszłości" (target leakage) – cecha wyliczona z etykiety.

Uruchomienie:
    python lectures/scripts/lecture_02/04_data_leakage.py
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5000))   # czysty szum
    y = rng.integers(0, 2, 200)        # losowe etykiety – prawdziwa accuracy ≈ 0.5

    print("=== 1. Preprocessing/selekcja cech przed podziałem ===")
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)  # ❌ widzi etykiety całego zbioru
    bad = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()
    print(f"❌ ŹLE  (selekcja na całych danych): CV accuracy = {bad:.3f}")

    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    good = cross_val_score(pipe, X, y, cv=5).mean()           # ✅ dopasowanie w każdym foldzie
    print(f"✅ DOBRZE (selekcja w Pipeline):      CV accuracy = {good:.3f}")
    print("➡️  Dane to szum – uczciwy wynik to ~0.5. Wynik 'ŹLE' jest iluzją.\n")

    print("=== 2. Target leakage (cecha z przyszłości) ===")
    X2 = rng.normal(size=(1000, 5))
    y2 = (X2[:, 0] + rng.normal(scale=2, size=1000) > 0).astype(int)
    leak = (y2 + rng.normal(scale=0.3, size=1000)).reshape(-1, 1)  # np. "liczba dni od rezygnacji"
    base = cross_val_score(LogisticRegression(), X2, y2, cv=5).mean()
    leaked = cross_val_score(LogisticRegression(), np.hstack([X2, leak]), y2, cv=5).mean()
    print(f"Bez cechy z wyciekiem:  {base:.3f}")
    print(f"Z cechą z wyciekiem:    {leaked:.3f}  ← w produkcji ta cecha nie będzie dostępna!")


if __name__ == "__main__":
    main()
