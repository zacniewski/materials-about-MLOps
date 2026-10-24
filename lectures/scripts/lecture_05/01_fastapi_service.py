"""Wykład 5, sekcja 2 – Serwis predykcji w FastAPI (bez uruchamiania serwera).

Buduje minimalny, ale "produkcyjny" serwis: walidacja Pydantic v2,
ładowanie modelu raz w `lifespan`, endpointy /health, /ready, /v1/predict,
/v1/predict/batch. Następnie testuje go in-process przez `TestClient`
(httpx), więc nie zostaje żaden działający proces.

Uruchomienie:
    python lectures/scripts/lecture_05/01_fastapi_service.py

Jako prawdziwy serwer (opcjonalnie, Ctrl+C aby zakończyć):
    uvicorn --app-dir lectures/scripts/lecture_05 01_fastapi_service:app --port 8080
"""

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["age", "income", "tenure_months", "num_products"]
STATE: dict = {"model": None, "version": "unknown", "start": time.time()}


class PredictionRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    age: float = Field(..., ge=18, le=120)
    income: float = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=10)

    @field_validator("income")
    @classmethod
    def income_reasonable(cls, v: float) -> float:
        if v > 1_000_000:
            raise ValueError("Dochód wydaje się zbyt wysoki")
        return v


class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str
    model_version: str


def train_demo_model() -> RandomForestClassifier:
    rng = np.random.default_rng(42)
    n = 2000
    X = np.column_stack([
        rng.integers(18, 80, n),
        rng.normal(6000, 2000, n).clip(0),
        rng.integers(0, 120, n),
        rng.integers(1, 6, n),
    ])
    logit = -0.03 * X[:, 2] - 0.4 * X[:, 3] + 1.5
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model ładowany RAZ przy starcie (a nie przy każdym żądaniu)
    STATE["model"] = train_demo_model()
    STATE["version"] = "demo-v1.0"
    yield
    STATE["model"] = None


app = FastAPI(title="Churn API (demo)", lifespan=lifespan)


def risk(p: float) -> str:
    return "low" if p < 0.3 else "medium" if p < 0.7 else "high"


def to_features(reqs: list[PredictionRequest]) -> np.ndarray:
    return np.array([[getattr(r, f) for f in FEATURES] for r in reqs], dtype=float)


@app.get("/health")
def health() -> dict:
    """Liveness: proces żyje."""
    return {"status": "alive", "uptime_s": round(time.time() - STATE["start"], 2)}


@app.get("/ready")
def ready() -> dict:
    """Readiness: model załadowany, można kierować ruch."""
    if STATE["model"] is None:
        raise HTTPException(503, "Model nie jest załadowany")
    return {"status": "ready", "model_version": STATE["version"]}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    p = float(STATE["model"].predict_proba(to_features([req]))[0, 1])
    return PredictionResponse(user_id=req.user_id, churn_probability=p,
                              churn_prediction=p >= 0.5, risk_level=risk(p),
                              model_version=STATE["version"])


@app.post("/v1/predict/batch", response_model=list[PredictionResponse])
def predict_batch(reqs: list[PredictionRequest]) -> list[PredictionResponse]:
    if not reqs or len(reqs) > 1000:
        raise HTTPException(400, "Batch musi mieć 1..1000 rekordów")
    probs = STATE["model"].predict_proba(to_features(reqs))[:, 1]
    return [PredictionResponse(user_id=r.user_id, churn_probability=float(p),
                               churn_prediction=bool(p >= 0.5), risk_level=risk(float(p)),
                               model_version=STATE["version"])
            for r, p in zip(reqs, probs)]


def main() -> None:
    good = {"user_id": 1001, "age": 35, "income": 5500, "tenure_months": 3, "num_products": 1}
    with TestClient(app) as client:  # uruchamia lifespan
        print("GET /health  ->", client.get("/health").json())
        print("GET /ready   ->", client.get("/ready").json())
        r = client.post("/v1/predict", json=good)
        print("POST /v1/predict ->", r.status_code, r.json())

        bad = {**good, "age": -5, "income": 5_000_000}
        r = client.post("/v1/predict", json=bad)
        print("\nNiepoprawne wejście ->", r.status_code)
        for err in r.json()["detail"]:
            print(f"  pole={err['loc'][-1]!s:<8} błąd={err['msg']}")

        batch = [{**good, "user_id": i, "tenure_months": i * 10} for i in range(1, 6)]
        r = client.post("/v1/predict/batch", json=batch)
        print("\nPOST /v1/predict/batch ->", r.status_code)
        for row in r.json():
            print(f"  user={row['user_id']} p={row['churn_probability']:.3f} risk={row['risk_level']}")

        r = client.post("/v1/predict/batch", json=[])
        print("\nPusty batch ->", r.status_code, r.json()["detail"])
        print("\nDokumentacja OpenAPI dostępna pod /docs; ścieżki:",
              sorted(client.get("/openapi.json").json()["paths"]))


if __name__ == "__main__":
    main()
