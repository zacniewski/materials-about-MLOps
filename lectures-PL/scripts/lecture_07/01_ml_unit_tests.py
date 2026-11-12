"""Wykład 7, sekcja 2 – Testowanie kodu ML: testy jednostkowe i behawioralne.

Plik jest jednocześnie modułem z „kodem produkcyjnym” (preprocessing +
trening) oraz zestawem testów pytest. Pokazuje cztery rodzaje testów:
  * testy jednostkowe funkcji przetwarzania (czyste funkcje, deterministyczne),
  * testy kontraktu (kształt i zakres wyjścia, determinizm treningu),
  * testy behawioralne modelu: niezmienniczość (INV) i kierunkowość (DIR),
  * test minimalnej funkcjonalności (MFT) – model musi pobić baseline.

Uruchomienie (z katalogu głównego repozytorium):
    python lectures/scripts/lecture_07/01_ml_unit_tests.py
    # lub bezpośrednio przez pytest:
    pytest -q -p no:cacheprovider lectures/scripts/lecture_07/01_ml_unit_tests.py
"""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["age", "income", "tenure_months", "support_calls", "income_per_age"]


# ---------------------------------------------------------------------------
# "Kod produkcyjny" (w prawdziwym projekcie: src/preprocessing.py, src/train.py)
# ---------------------------------------------------------------------------
def make_dataset(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Syntetyczne dane churn: więcej zgłoszeń do supportu -> większy churn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(6000, 1500, n).clip(1000).round(2),
        "tenure_months": rng.integers(1, 120, n),
        "support_calls": rng.poisson(2, n),
        "city": rng.choice(["Warszawa", "Kraków", "Gdańsk"], n),
    })
    logit = 0.6 * df["support_calls"] - 0.03 * df["tenure_months"] - 0.5
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa nierealistyczny wiek, braki w kolumnach kluczowych i duplikaty."""
    out = df.dropna(subset=["age", "income"])
    out = out[(out["age"] >= 0) & (out["age"] <= 120)]
    return out.drop_duplicates().reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cechy pochodne. Nie modyfikuje wejścia (brak efektów ubocznych)."""
    out = df.copy()
    out["income_per_age"] = out["income"] / out["age"].clip(lower=1)
    return out


def train_model(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(df[FEATURES], df["churn"])
    return pipe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    df = engineer_features(clean_data(make_dataset()))
    return train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["churn"])


@pytest.fixture(scope="module")
def model(data) -> Pipeline:
    train, _ = data
    return train_model(train)


# ---------------------------------------------------------------------------
# 1. Testy jednostkowe preprocessingu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_age", [-5, 200])
def test_clean_data_removes_unrealistic_age(bad_age):
    df = pd.DataFrame({"age": [30, bad_age], "income": [5000.0, 6000.0]})
    assert clean_data(df)["age"].tolist() == [30]


def test_clean_data_removes_duplicates_and_nans():
    df = pd.DataFrame({"age": [30, 30, np.nan], "income": [5000.0, 5000.0, 1.0]})
    assert len(clean_data(df)) == 1


def test_engineer_features_has_no_side_effects():
    df = pd.DataFrame({"age": [20, 40], "income": [4000.0, 8000.0]})
    before = df.copy()
    out = engineer_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert out["income_per_age"].tolist() == [200.0, 200.0]


def test_engineer_features_handles_zero_age():
    df = pd.DataFrame({"age": [0], "income": [1000.0]})
    assert np.isfinite(engineer_features(df)["income_per_age"]).all()


# ---------------------------------------------------------------------------
# 2. Testy kontraktu modelu
# ---------------------------------------------------------------------------
def test_predict_proba_contract(model, data):
    _, test = data
    proba = model.predict_proba(test[FEATURES])
    assert proba.shape == (len(test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_training_is_deterministic(data):
    train, test = data
    p1 = train_model(train).predict_proba(test[FEATURES])
    p2 = train_model(train).predict_proba(test[FEATURES])
    np.testing.assert_allclose(p1, p2)


# ---------------------------------------------------------------------------
# 3. Testy behawioralne (Ribeiro i in., „CheckList”, 2020 – idea dla tabel)
# ---------------------------------------------------------------------------
def test_invariance_to_city(model, data):
    """INV: zmiana cechy nieużywanej (miasto) nie może zmienić predykcji."""
    _, test = data
    changed = test.copy()
    changed["city"] = "Gdańsk"
    np.testing.assert_allclose(
        model.predict_proba(test[FEATURES]), model.predict_proba(changed[FEATURES])
    )


def test_directional_support_calls(model, data):
    """DIR: więcej zgłoszeń do supportu => prawdopodobieństwo churnu nie maleje."""
    _, test = data
    more = test.copy()
    more["support_calls"] += 3
    p_base = model.predict_proba(test[FEATURES])[:, 1]
    p_more = model.predict_proba(more[FEATURES])[:, 1]
    assert (p_more >= p_base - 1e-9).mean() > 0.99


# ---------------------------------------------------------------------------
# 4. Minimum Functionality Test – model musi pobić baseline
# ---------------------------------------------------------------------------
def test_model_beats_baseline(model, data):
    train, test = data
    dummy = DummyClassifier(strategy="prior").fit(train[FEATURES], train["churn"])
    auc_model = roc_auc_score(test["churn"], model.predict_proba(test[FEATURES])[:, 1])
    auc_dummy = roc_auc_score(test["churn"], dummy.predict_proba(test[FEATURES])[:, 1])
    assert auc_model > auc_dummy + 0.10, f"AUC {auc_model:.3f} vs baseline {auc_dummy:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-p", "no:cacheprovider", __file__]))
