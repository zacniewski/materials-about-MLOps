"""Wykład 4, sekcja 6a – Lokalny pipeline z artefaktami, lineage i bramką jakości.

Kroki: ingest → validate → train (kandydat) → evaluate → quality gate
(porównanie z modelem "champion" i progiem bezwzględnym) → promote/reject.
Każdy krok zapisuje artefakt oraz wpis do `metadata.jsonl` (run_id, krok,
hash wejścia/wyjścia, parametry) – to uproszczona wersja Metadata Store z KFP.

Uruchomienie:
    python lectures/scripts/lecture_04/02_quality_gate_pipeline.py [--auc-threshold 0.85]
"""

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent / "outputs" / "quality_gate"


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:10]


def log_meta(run_id, step, **fields):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run_id": run_id, "step": step, **fields}
    with open(OUT / "metadata.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ingest(run_id) -> Path:
    X, y = make_classification(n_samples=6000, n_features=15, n_informative=8,
                               weights=[0.8, 0.2], random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)]).assign(target=y)
    path = OUT / "dataset.parquet"
    df.to_parquet(path, index=False)
    log_meta(run_id, "ingest", output=str(path.name), output_hash=file_hash(path), rows=len(df))
    return path


def validate(run_id, path: Path) -> None:
    df = pd.read_parquet(path)
    checks = {
        "min_rows": len(df) >= 1000,
        "no_nulls": int(df.isnull().sum().sum()) == 0,
        "has_target": "target" in df.columns,
        "target_binary": set(df["target"].unique()) <= {0, 1},
        "pos_rate_in_range": 0.05 <= df["target"].mean() <= 0.5,
    }
    checks = {k: bool(v) for k, v in checks.items()}  # numpy.bool_ → bool (JSON)
    log_meta(run_id, "validate", input_hash=file_hash(path), checks=checks)
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Walidacja nie przeszła: {failed}")
    print(f"  ✓ validate: {checks}")


def split(path: Path):
    df = pd.read_parquet(path)
    X, y = df.drop(columns="target"), df["target"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def train(run_id, path: Path, model, name: str) -> Path:
    X_tr, _, y_tr, _ = split(path)
    model.fit(X_tr, y_tr)
    mpath = OUT / f"{name}.joblib"
    joblib.dump(model, mpath)
    log_meta(run_id, f"train_{name}", input_hash=file_hash(path),
             params=model.get_params(), output_hash=file_hash(mpath))
    return mpath


def evaluate(run_id, path: Path, mpath: Path) -> dict:
    _, X_te, _, y_te = split(path)
    model = joblib.load(mpath)
    prob = model.predict_proba(X_te)[:, 1]
    m = {"auc": round(roc_auc_score(y_te, prob), 4),
         "f1": round(f1_score(y_te, (prob >= 0.5).astype(int)), 4)}
    log_meta(run_id, f"evaluate_{mpath.stem}", model_hash=file_hash(mpath), metrics=m)
    return m


def quality_gate(cand: dict, champ: dict, abs_thr: float, min_gain: float) -> tuple[str, list]:
    reasons = []
    if cand["auc"] < abs_thr:
        reasons.append(f"AUC {cand['auc']} < próg {abs_thr}")
    if cand["auc"] - champ["auc"] < min_gain:
        reasons.append(f"zysk vs champion {cand['auc'] - champ['auc']:+.4f} < {min_gain}")
    if cand["f1"] < champ["f1"] - 0.02:
        reasons.append("regresja F1 > 0.02")
    return ("promote" if not reasons else "reject"), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auc-threshold", type=float, default=0.85)
    parser.add_argument("--min-gain", type=float, default=0.005)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata.jsonl").unlink(missing_ok=True)
    np.random.seed(42)
    run_id = uuid.uuid4().hex[:8]
    print(f"▶ run_id={run_id}")

    data = ingest(run_id)
    validate(run_id, data)
    champ_path = train(run_id, data, LogisticRegression(max_iter=1000), "champion")
    cand_path = train(run_id, data, GradientBoostingClassifier(random_state=42), "candidate")
    champ, cand = evaluate(run_id, data, champ_path), evaluate(run_id, data, cand_path)
    print(f"  champion  (LogReg): {champ}")
    print(f"  candidate (GBM)   : {cand}")

    decision, reasons = quality_gate(cand, champ, args.auc_threshold, args.min_gain)
    log_meta(run_id, "quality_gate", decision=decision, reasons=reasons)
    print(f"  🚦 decyzja: {decision.upper()} {reasons if reasons else ''}")

    print(f"\nLineage ({OUT / 'metadata.jsonl'}):")
    for line in (OUT / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        extra = {k: v for k, v in rec.items() if k not in ("ts", "run_id", "step", "params")}
        print(f"  {rec['step']:<20} {extra}")


if __name__ == "__main__":
    main()
