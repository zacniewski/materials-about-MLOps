"""Wykład 8, sekcja 4.5 – Ochrona danych osobowych (PII) w pipeline ML.

Pokazuje trzy techniki stosowane przed zapisaniem danych do logów/feature store:
  * maskowanie (regex) – e-mail, telefon, PESEL, IBAN w tekście logów,
  * pseudonimizacja – HMAC-SHA256 z sekretnym kluczem (stabilny join, brak odwrócenia),
  * generalizacja – wiek → przedział, kod pocztowy → prefiks (k-anonimowość).
Na końcu liczy k-anonimowość przed i po generalizacji.

Uruchomienie:
    python lectures/scripts/lecture_08/03_pii_masking.py
Wynik: lectures/scripts/lecture_08/outputs/customers_anonymized.csv
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
    print("=== 1. Maskowanie PII w logach ===")
    logs = [
        "Predykcja dla jan.kowalski@example.com: churn=0.83",
        "Klient tel. +48 601 222 333, PESEL 90010112345 – fraud_score=0.97",
        "Przelew z PL61109010140000071219812874 odrzucony przez model",
    ]
    for line in logs:
        print(f"  PRZED: {line}\n  PO:    {mask_text(line)}")

    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "email": [f"user{i}@example.com" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "zip": [f"{rng.integers(0, 99):02d}-{rng.integers(0, 999):03d}" for _ in range(n)],
        "income": rng.normal(6000, 1500, n).round(0),
    })
    quasi = ["age", "zip"]
    print(f"\n=== 2. k-anonimowość (quasi-identyfikatory: {quasi}) ===")
    print(f"  Przed generalizacją: k = {k_anonymity(df, quasi)} (rekordy unikalne → re-identyfikacja)")

    anon = pd.DataFrame({
        "user_key": df["email"].map(pseudonymize),
        "age_band": pd.cut(df["age"], bins=[17, 30, 45, 60, 80]).astype(str),
        "zip_prefix": df["zip"].str[:1] + "x-xxx",
        "income": df["income"],
    })
    print(f"  Po generalizacji:    k = {k_anonymity(anon, ['age_band', 'zip_prefix'])}")

    print("\n=== 3. Pseudonimizacja HMAC ===")
    print(f"  user0@example.com → {pseudonymize('user0@example.com')} (ten sam klucz = ten sam pseudonim)")
    plain_sha = hashlib.sha256(b"user0@example.com").hexdigest()[:16]
    print(f"  zwykły SHA-256     → {plain_sha} (podatny na atak słownikowy – nie używaj bez klucza!)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "customers_anonymized.csv"
    anon.to_csv(out, index=False)
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
