"""Wykład 2, sekcja 5 – Pipeline przetwarzania danych (pandas + scikit-learn).

Kroki: wczytanie → czyszczenie → feature engineering → preprocessing
(ColumnTransformer) → trening → zapis całego pipeline'u do pliku.
Dane są generowane syntetycznie (z celowo wstawionymi brakami i duplikatami).

Uruchomienie:
    python lectures/scripts/lecture_02/02_data_pipeline.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
NUMERIC = ["age", "income", "tenure_months", "income_per_year_of_age"]
CATEGORICAL = ["city", "plan"]


def load_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Krok 1: wczytanie (tu: generowanie) surowych danych."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "age": rng.integers(18, 80, n).astype(float),
        "income": rng.normal(5000, 1500, n),
        "tenure_months": rng.integers(0, 120, n),
        "city": rng.choice(["Warszawa", "Kraków", "Gdańsk", None], n, p=[0.4, 0.3, 0.25, 0.05]),
        "plan": rng.choice(["basic", "premium", "pro"], n),
    })
    logit = -0.03 * df["tenure_months"] + (df["plan"] == "basic") * 1.0 - 0.0001 * df["income"] + 1
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # zanieczyszczenia
    df.loc[rng.choice(n, 200, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "income"] = -1  # błędne wartości
    df = pd.concat([df, df.sample(100, random_state=seed)])  # duplikaty
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Krok 2: czyszczenie."""
    before = len(df)
    df = df.drop_duplicates(subset="customer_id")
    df = df[(df["income"] > 0) | df["income"].isna()]
    df = df[df["age"].between(18, 100) | df["age"].isna()]
    print(f"🧹 Czyszczenie: {before} → {len(df)} wierszy")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Krok 3: feature engineering."""
    df = df.copy()
    df["income_per_year_of_age"] = df["income"] / df["age"]
    return df


def build_preprocessor() -> ColumnTransformer:
    """Krok 4: preprocessing – dopasowywany TYLKO na zbiorze treningowym."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def main() -> None:
    df = engineer_features(clean_data(load_data()))
    print(f"📊 Braki danych po czyszczeniu:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}\n")

    X, y = df[NUMERIC + CATEGORICAL], df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ])
    pipeline.fit(X_tr, y_tr)
    print(classification_report(y_te, pipeline.predict(X_te), digits=3))

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "churn_pipeline.joblib"
    joblib.dump(pipeline, path)
    print(f"💾 Pipeline (preprocessing + model) zapisany do: {path}")

    sample = pd.DataFrame([{"age": 30, "income": 4000, "tenure_months": 2,
                            "income_per_year_of_age": 4000 / 30, "city": "Kraków", "plan": "basic"}])
    print(f"🔮 P(churn) dla nowego klienta: {joblib.load(path).predict_proba(sample)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
