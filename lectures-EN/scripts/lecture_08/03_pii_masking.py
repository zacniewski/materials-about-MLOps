"""Lecture 8, section 4.5 – Protecting personal data (PII) in an ML pipeline.

Shows three techniques applied before writing data to logs/feature store:
  * masking (regex) – e-mail, phone, PESEL, IBAN in log text,
  * pseudonymization – HMAC-SHA256 with a secret key (stable join, non-reversible),
  * generalization – age → band, postal code → prefix (k-anonymity).
Finally computes k-anonymity before and after generalization.

Run:
    python lectures/scripts/lecture_08/03_pii_masking.py
Output: lectures/scripts/lecture_08/outputs/customers_anonymized.csv
"""

import hashlib
import hmac
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "outputs"
SECRET = os.environ.get("PII_HMAC_KEY", "demo-only-key-change-me").encode()

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "IBAN": re.compile(r"\bPL\d{26}\b"),
    "PESEL": re.compile(r"\b\d{11}\b"),
    "PHONE": re.compile(r"(?:\+48[ -]?)?\b\d{3}[ -]?\d{3}[ -]?\d{3}\b"),
}


def mask_text(text: str) -> str:
    for name, pattern in PATTERNS.items():
        text = pattern.sub(f"<{name}>", text)
    return text


def pseudonymize(value: str) -> str:
    return hmac.new(SECRET, value.encode(), hashlib.sha256).hexdigest()[:16]


def k_anonymity(df: pd.DataFrame, quasi_ids: list[str]) -> int:
    return int(df.groupby(quasi_ids, observed=True).size().min())


def main() -> None:
    print("=== 1. Masking PII in logs ===")
    logs = [
        "Prediction for jan.kowalski@example.com: churn=0.83",
        "Customer phone +48 601 222 333, PESEL 90010112345 – fraud_score=0.97",
        "Transfer from PL61109010140000071219812874 rejected by the model",
    ]
    for line in logs:
        print(f"  BEFORE: {line}\n  AFTER:  {mask_text(line)}")

    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "email": [f"user{i}@example.com" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "zip": [f"{rng.integers(0, 99):02d}-{rng.integers(0, 999):03d}" for _ in range(n)],
        "income": rng.normal(6000, 1500, n).round(0),
    })
    quasi = ["age", "zip"]
    print(f"\n=== 2. k-anonymity (quasi-identifiers: {quasi}) ===")
    print(f"  Before generalization: k = {k_anonymity(df, quasi)} (unique records → re-identification)")

    anon = pd.DataFrame({
        "user_key": df["email"].map(pseudonymize),
        "age_band": pd.cut(df["age"], bins=[17, 30, 45, 60, 80]).astype(str),
        "zip_prefix": df["zip"].str[:1] + "x-xxx",
        "income": df["income"],
    })
    print(f"  After generalization:  k = {k_anonymity(anon, ['age_band', 'zip_prefix'])}")

    print("\n=== 3. HMAC pseudonymization ===")
    print(f"  user0@example.com → {pseudonymize('user0@example.com')} (same key = same pseudonym)")
    plain_sha = hashlib.sha256(b"user0@example.com").hexdigest()[:16]
    print(f"  plain SHA-256      → {plain_sha} (vulnerable to dictionary attack – do not use without a key!)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "customers_anonymized.csv"
    anon.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
