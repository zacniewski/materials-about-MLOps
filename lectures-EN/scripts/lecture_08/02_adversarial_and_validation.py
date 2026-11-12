"""Lecture 8, section 4.2 – Adversarial attacks and input validation.

1. Trains a logistic regression ("fraud" model) on synthetic data.
2. Performs an FGSM-style attack on the linear model: shifts the input in the direction
   of sign(w) by epsilon and measures how many "fraud" predictions were flipped.
3. Shows layered defense: schema validation (pydantic) + an outlier
   detector (z-score) – attacks with a large epsilon are rejected.
4. Simulates model extraction: the attacker queries the API and trains a copy of the model.

Run:
    python lectures/scripts/lecture_08/02_adversarial_and_validation.py
"""

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Transaction(BaseModel):
    """Request schema (pydantic v2) – the first line of defense."""

    amount: float = Field(..., gt=0, le=50_000)
    features: list[float] = Field(..., min_length=10, max_length=10)

    @field_validator("features")
    @classmethod
    def finite(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(v)):
            raise ValueError("NaN/Inf in features")
        return v


def main() -> None:
    rng = np.random.default_rng(42)
    X, y = make_classification(n_samples=6000, n_features=10, n_informative=6,
                               weights=[0.9, 0.1], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    w = model.coef_[0] / scaler.scale_  # gradient of the logit w.r.t. the raw input

    fraud = X_te[model.predict(scaler.transform(X_te)) == 1]
    print(f"Detected frauds in the test set: {len(fraud)}\n")

    mu, sd = X_tr.mean(0), X_tr.std(0)
    print("=== FGSM attack: x_adv = x - eps * sign(w) ===")
    print(f"{'eps':>5} | {'flipped':>9} | {'passed through validator':>29}")
    for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x_adv = fraud - eps * np.sign(w)
        evaded = model.predict(scaler.transform(x_adv)) == 0
        z_ok = (np.abs((x_adv - mu) / sd) < 4).all(axis=1)  # defense: z-score < 4
        print(f"{eps:>5} | {evaded.mean():>8.0%} | {(evaded & z_ok).mean():>28.0%}")
    print("→ Small perturbations are undetectable by a simple z-score – you need")
    print("  adversarial training, monitoring and business rules (defense in depth).\n")

    print("=== Schema validation (pydantic) ===")
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
            print(f"  REJECTED amount={p['amount']}: {e.errors()[0]['msg']}")

    print("\n=== Model extraction (stealing the model via the API) ===")
    for n_queries in [50, 500, 5000]:
        Q = rng.normal(mu, sd, size=(n_queries, X.shape[1]))
        labels = model.predict(scaler.transform(Q))
        if len(set(labels)) < 2:
            print(f"  {n_queries:>5} queries: not enough diverse responses")
            continue
        clone = LogisticRegression(max_iter=1000).fit(Q, labels)
        agree = (clone.predict(X_te) == model.predict(scaler.transform(X_te))).mean()
        print(f"  {n_queries:>5} queries → copy agreement with the original: {agree:.1%}")
    print("→ Rate limiting + returning only the label (no probabilities) makes the attack harder.")


if __name__ == "__main__":
    main()
