"""Lecture 4, section 2a – Mini-orchestrator: DAG, topological sort, retry and caching.

Shows "from the inside" what an orchestrator (KFP/Airflow/Prefect) does:
- builds a DAG from dependencies between steps and detects cycles,
- determines the execution order (Kahn's algorithm) and "waves" of steps
  that could run in parallel,
- retries steps that failed with a transient error (retry with backoff),
- caches step results by key = hash(step code + parameters + inputs).

A second run with a different `n_estimators` shows that the "ingest"
and "validate" steps are taken from the cache, while "train"/"evaluate" are recomputed.

Usage:
    python lectures-EN/scripts/lecture_04/01_mini_orchestrator.py
"""

import hashlib
import inspect
import json
import pickle
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CACHE_DIR = Path(__file__).parent / "outputs" / "cache"


class Step:
    def __init__(self, name, fn, deps=(), params=None, retries=0):
        self.name, self.fn, self.deps = name, fn, list(deps)
        self.params, self.retries = params or {}, retries


def topological_waves(steps: dict[str, Step]) -> list[list[str]]:
    """Kahn's algorithm; returns a list of "waves" – steps within one wave are independent."""
    indegree = {n: len(s.deps) for n, s in steps.items()}
    children = {n: [] for n in steps}
    for n, s in steps.items():
        for d in s.deps:
            children[d].append(n)
    queue = deque(sorted(n for n, k in indegree.items() if k == 0))
    waves, visited = [], 0
    while queue:
        wave = list(queue)
        queue.clear()
        waves.append(wave)
        for n in wave:
            visited += 1
            for c in children[n]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    queue.append(c)
    if visited != len(steps):
        raise ValueError("The graph contains a cycle – this is not a DAG!")
    return waves


def cache_key(step: Step, inputs: dict) -> str:
    h = hashlib.sha256()
    h.update(inspect.getsource(step.fn).encode())
    h.update(json.dumps(step.params, sort_keys=True).encode())
    for k in sorted(inputs):
        h.update(k.encode())
        h.update(pickle.dumps(inputs[k]))
    return h.hexdigest()[:16]


def run_pipeline(steps: dict[str, Step], use_cache: bool = True) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for i, wave in enumerate(topological_waves(steps)):
        print(f"  wave {i}: {wave}")
        for name in wave:
            step = steps[name]
            inputs = {d: results[d] for d in step.deps}
            key = cache_key(step, inputs)
            path = CACHE_DIR / f"{name}-{key}.pkl"
            if use_cache and path.exists():
                results[name] = pickle.loads(path.read_bytes())
                print(f"    💾 {name:<9} cache hit ({key})")
                continue
            for attempt in range(step.retries + 1):
                try:
                    start = time.perf_counter()
                    results[name] = step.fn(**inputs, **step.params)
                    print(f"    ✓  {name:<9} executed in {time.perf_counter() - start:.2f}s "
                          f"(attempt {attempt + 1})")
                    break
                except ConnectionError as exc:
                    if attempt == step.retries:
                        raise
                    delay = 0.1 * 2 ** attempt
                    print(f"    ⟳  {name:<9} error: {exc} – retry in {delay:.1f}s")
                    time.sleep(delay)
            path.write_bytes(pickle.dumps(results[name]))
    return results


# --- Pipeline steps ---------------------------------------------------------
_flaky = random.Random(7)


def ingest(n_samples):
    if _flaky.random() < 0.5:  # simulating an unstable data source
        raise ConnectionError("data warehouse timeout")
    X, y = make_classification(n_samples=n_samples, n_features=20,
                               n_informative=10, random_state=42)
    return X, y


def validate(ingest, min_rows):
    X, y = ingest
    assert len(X) >= min_rows, f"Too few rows: {len(X)}"
    assert not np.isnan(X).any(), "NaN in data"
    return {"rows": len(X), "pos_rate": float(y.mean())}


def train(ingest, validate, n_estimators):
    X, y = ingest
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                  n_jobs=-1).fit(X_tr, y_tr)


def stats(ingest):
    X, _ = ingest
    return {"mean": X.mean(axis=0).round(3).tolist()[:3]}


def evaluate(ingest, train):
    X, y = ingest
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return round(roc_auc_score(y_te, train.predict_proba(X_te)[:, 1]), 4)


def build(n_estimators):
    return {
        "ingest": Step("ingest", ingest, params={"n_samples": 5000}, retries=3),
        "validate": Step("validate", validate, ["ingest"], {"min_rows": 1000}),
        "stats": Step("stats", stats, ["ingest"]),
        "train": Step("train", train, ["ingest", "validate"], {"n_estimators": n_estimators}),
        "evaluate": Step("evaluate", evaluate, ["ingest", "train"]),
    }


def main() -> None:
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()

    print("▶ Run 1 (n_estimators=50)")
    print(f"  AUC = {run_pipeline(build(50))['evaluate']}")
    print("\n▶ Run 2 (n_estimators=150) – ingest/validate/stats from cache")
    print(f"  AUC = {run_pipeline(build(150))['evaluate']}")

    print("\n▶ Attempt with a cycle (train → ingest)")
    bad = build(50)
    bad["ingest"].deps = ["train"]
    try:
        topological_waves(bad)
    except ValueError as exc:
        print(f"  ❌ {exc}")


if __name__ == "__main__":
    main()
