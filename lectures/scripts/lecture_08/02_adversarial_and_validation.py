"""Wykład 8, sekcja 4.2 – Ataki adversarialne i walidacja wejść.

1. Trenuje regresję logistyczną (model "fraud") na danych syntetycznych.
2. Przeprowadza atak typu FGSM na model liniowy: przesuwa wejście w kierunku
   sign(w) o epsilon i mierzy, ile predykcji "fraud" udało się odwrócić.
3. Pokazuje obronę warstwową: walidacja schematu (pydantic) + detektor
   wartości odstających (z-score) – ataki z dużym epsilon zostają odrzucone.
4. Symuluje model extraction: atakujący odpytuje API i trenuje kopię modelu.

Uruchomienie:
    python lectures/scripts/lecture_08/02_adversarial_and_validation.py
"""

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Transaction(BaseModel):
    """Schemat żądania (pydantic v2) – pierwsza linia obrony."""

    amount: float = Field(..., gt=0, le=50_000)
    features: list[float] = Field(..., min_length=10, max_length=10)

    @field_validator("features")
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(v)):
            raise ValueError("NaN/Inf w cechach")
        return v


def main() -> None:
    rng = np.random.default_rng(42)
    X, y = make_classification(n_samples=6000, n_features=10, n_informative=6,
                               weights=[0.9, 0.1], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    w = model.coef_[0] / scaler.scale_  # gradient logitu względem surowego wejścia

    fraud = X_te[model.predict(scaler.transform(X_te)) == 1]
    print(f"Wykryte fraudy w zbiorze testowym: {len(fraud)}\n")

    mu, sd = X_tr.mean(0), X_tr.std(0)
    print("=== Atak FGSM: x_adv = x - eps * sign(w) ===")
    print(f"{'eps':>5} | {'odwrócone':>9} | {'przepuszczone przez walidator':>29}")
    for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x_adv = fraud - eps * np.sign(w)
        evaded = model.predict(scaler.transform(x_adv)) == 0
        z_ok = (np.abs((x_adv - mu) / sd) < 4).all(axis=1)  # obrona: z-score < 4
        print(f"{eps:>5} | {evaded.mean():>8.0%} | {(evaded & z_ok).mean():>28.0%}")
    print("→ Małe perturbacje są niewykrywalne przez prosty z-score – potrzebne")
    print("  adversarial training, monitoring i reguły biznesowe (defense in depth).\n")

    print("=== Walidacja schematu (pydantic) ===")
    payloads = [
        {"amount": 120.0, "features": list(X_te[0])},
        {"amount": -5, "features": list(X_te[0])},
        {"amount": 100, "features": [float("nan")] * 10},
        {"amount": 100, "features": [1.0] * 3},
    ]
    for p in payloads:
        try:
            Transaction(**p)
            print("  OK       ", p["amount"])
        except ValidationError as e:
            print(f"  ODRZUCONE amount={p['amount']}: {e.errors()[0]['msg']}")

    print("\n=== Model extraction (kradzież modelu przez API) ===")
    for n_queries in [50, 500, 5000]:
        Q = rng.normal(mu, sd, size=(n_queries, X.shape[1]))
        labels = model.predict(scaler.transform(Q))
        if len(set(labels)) < 2:
            print(f"  {n_queries:>5} zapytań: za mało różnorodnych odpowiedzi")
            continue
        clone = LogisticRegression(max_iter=1000).fit(Q, labels)
        agree = (clone.predict(X_te) == model.predict(scaler.transform(X_te))).mean()
        print(f"  {n_queries:>5} zapytań → zgodność kopii z oryginałem: {agree:.1%}")
    print("→ Rate limiting + zwracanie tylko etykiety (bez prawdopodobieństw) utrudnia atak.")


if __name__ == "__main__":
    main()
