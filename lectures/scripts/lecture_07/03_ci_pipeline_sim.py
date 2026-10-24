"""Wykład 7, sekcje 3 i 6 – Symulacja pipeline'u CI/CD dla ML (fail-fast).

Lokalny odpowiednik workflow GitHub Actions / `make pipeline`: etapy
uruchamiane są po kolei, a pierwszy błąd zatrzymuje pipeline (fail-fast).
Pokazuje też:
  * smoke training na próbce danych (szybki feedback w PR),
  * hash danych + hash modelu jako metadane artefaktu (reprodukowalność),
  * zapis modelu i manifestu do outputs/ (odpowiednik upload-artifact).

Uruchomienie:
    python lectures/scripts/lecture_07/03_ci_pipeline_sim.py
    python lectures/scripts/lecture_07/03_ci_pipeline_sim.py --break-data   # błąd walidacji
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).parent / "outputs"
SEED = 42
EXPECTED_SCHEMA = {"f1": "float64", "f2": "float64", "f3": "float64", "label": "int64"}
CTX: dict = {}


def stage_load_data(break_data: bool) -> None:
    rng = np.random.default_rng(SEED)
    n = 20_000
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "f3": rng.normal(size=n)})
    df["label"] = ((df.f1 + 0.5 * df.f2 + rng.normal(0, 0.5, n)) > 0).astype("int64")
    if break_data:
        df.loc[: n // 5, "f2"] = np.nan  # 20% braków – np. awaria źródła
    CTX["df"] = df
    raw = pd.util.hash_pandas_object(df).values.tobytes()
    CTX["data_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    print(f"      {len(df)} wierszy, data_hash={CTX['data_hash']}")


def stage_validate_data(_: bool) -> None:
    df = CTX["df"]
    schema = {c: str(t) for c, t in df.dtypes.items()}
    assert schema == EXPECTED_SCHEMA, f"schemat {schema} != {EXPECTED_SCHEMA}"
    null_frac = df.isna().mean().max()
    assert null_frac <= 0.01, f"za dużo braków: {null_frac:.1%} > 1%"
    assert set(df["label"].unique()) <= {0, 1}, "niepoprawne etykiety"


def stage_smoke_train(_: bool) -> None:
    df = CTX["df"].sample(frac=0.1, random_state=SEED)  # próbka => szybki feedback
    X_tr, X_te, y_tr, y_te = train_test_split(df[["f1", "f2", "f3"]], df["label"],
                                              test_size=0.3, random_state=SEED)
    model = RandomForestClassifier(n_estimators=50, random_state=SEED).fit(X_tr, y_tr)
    CTX["model"] = model
    CTX["auc"] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    print(f"      smoke AUC (10% danych) = {CTX['auc']:.4f}")


def stage_quality_gate(_: bool) -> None:
    assert CTX["auc"] >= 0.80, f"AUC {CTX['auc']:.4f} < 0.80"


def stage_package(_: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "model.joblib"
    joblib.dump(CTX["model"], path)
    manifest = {
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        "data_hash": CTX["data_hash"],
        "auc_smoke": round(CTX["auc"], 4),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"      zapisano {path.name} + model_manifest.json")


def stage_smoke_test(_: bool) -> None:
    model = joblib.load(OUT_DIR / "model.joblib")
    sample = pd.DataFrame({"f1": [2.0, -2.0], "f2": [0.0, 0.0], "f3": [0.0, 0.0]})
    pred = model.predict(sample)
    assert list(pred) == [1, 0], f"nieoczekiwane predykcje {pred}"


STAGES = [
    ("load-data", stage_load_data),
    ("validate-data", stage_validate_data),
    ("smoke-train", stage_smoke_train),
    ("quality-gate", stage_quality_gate),
    ("package-model", stage_package),
    ("smoke-test", stage_smoke_test),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--break-data", action="store_true")
    args = parser.parse_args()

    total = time.perf_counter()
    for i, (name, fn) in enumerate(STAGES, 1):
        start = time.perf_counter()
        print(f"[{i}/{len(STAGES)}] {name} ...")
        try:
            fn(args.break_data)
        except AssertionError as exc:
            print(f"      ❌ FAILED ({time.perf_counter() - start:.2f}s): {exc}")
            skipped = [s for s, _ in STAGES[i:]]
            print(f"\nPipeline zatrzymany (fail-fast). Pominięte etapy: {', '.join(skipped)}")
            sys.exit(1)
        print(f"      ✅ OK ({time.perf_counter() - start:.2f}s)")
    print(f"\nPipeline zakończony sukcesem w {time.perf_counter() - total:.2f}s")


if __name__ == "__main__":
    main()
