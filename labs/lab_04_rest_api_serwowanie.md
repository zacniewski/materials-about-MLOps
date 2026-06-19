# Laboratorium 4: REST API i Serwowanie Modelu ML

## Informacje ogólne
- **Czas:** 2 godziny
- **Poziom:** Średni
- **Wymagania wstępne:** Lab 1–3 ukończone, podstawy HTTP/REST

## Cele laboratorium
Po tym laboratorium student:
- zbuduje REST API dla modelu ML z użyciem FastAPI,
- skonteneryzuje serwis w Dockerze,
- przetestuje API (unit testy + testy integracyjne),
- zmierzy wydajność serwisu (latencja, throughput).

---

## Część 1: Budowa REST API z FastAPI (50 min)

### Krok 1.1: Instalacja zależności

```bash
pip install fastapi==0.110.0 uvicorn==0.27.0 httpx==0.27.0 pytest-asyncio==0.23.0
```

### Krok 1.2: Serwis predykcji

Utwórz `src/serving/api.py`:

```python
# src/serving/api.py
"""REST API dla modelu predykcji churnu."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import joblib
import numpy as np
import time
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Modele Pydantic ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    customer_id: int = Field(..., gt=0, description="ID klienta")
    age: float = Field(..., ge=18, le=100, description="Wiek")
    income: float = Field(..., ge=0, description="Dochód miesięczny")
    tenure_months: int = Field(..., ge=0, le=600, description="Staż w miesiącach")
    num_products: int = Field(..., ge=1, le=10, description="Liczba produktów")
    has_credit_card: int = Field(..., ge=0, le=1, description="Czy ma kartę kredytową")
    is_active: int = Field(..., ge=0, le=1, description="Czy aktywny")
    balance: float = Field(..., ge=0, description="Saldo konta")
    num_transactions_30d: int = Field(..., ge=0, description="Transakcje w 30 dniach")

    @validator('income')
    def income_reasonable(cls, v):
        if v > 10_000_000:
            raise ValueError("Dochód zbyt wysoki")
        return round(v, 2)

    def to_features(self) -> np.ndarray:
        """Konwertuje żądanie na wektor cech."""
        income_per_age = self.income / max(self.age, 1)
        balance_per_product = self.balance / max(self.num_products, 1)
        is_long_tenure = int(self.tenure_months > 24)
        return np.array([[
            self.age, self.income, self.tenure_months, self.num_products,
            self.has_credit_card, self.is_active, self.balance,
            self.num_transactions_30d, income_per_age,
            balance_per_product, is_long_tenure
        ]])


class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_level: str
    model_version: str
    prediction_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest]

    @validator('requests')
    def max_batch_size(cls, v):
        if len(v) > 500:
            raise ValueError("Maksymalnie 500 rekordów w batchu")
        return v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


# ── Stan aplikacji ───────────────────────────────────────────────────────────

class AppState:
    model = None
    model_version = "unknown"
    start_time = time.time()
    total_predictions = 0
    prediction_log: list = []


state = AppState()


# ── Lifecycle ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ładuje model przy starcie aplikacji."""
    logger.info("Uruchamianie serwisu ML...")
    model_path = Path("models/churn_model.pkl")

    if model_path.exists():
        state.model = joblib.load(model_path)
        state.model_version = "v1.0.0"
        logger.info(f"Model załadowany: {state.model_version}")
    else:
        # Tryb demo – trenuj prosty model
        logger.warning("Brak pliku modelu – używam modelu demo")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=11, random_state=42)
        state.model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        state.model_version = "demo-v1.0"

    yield

    logger.info("Zamykanie serwisu ML...")


# ── Aplikacja ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="REST API do predykcji churnu klientów",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_risk(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    elif probability >= 0.3:
        return "medium"
    return "low"


def log_prediction_async(customer_id: int, probability: float, latency_ms: float):
    """Loguje predykcję (wywoływana w tle)."""
    entry = {
        "customer_id": customer_id,
        "probability": probability,
        "latency_ms": latency_ms,
        "timestamp": datetime.now().isoformat()
    }
    state.prediction_log.append(entry)
    # Zachowaj tylko ostatnie 1000 wpisów
    if len(state.prediction_log) > 1000:
        state.prediction_log.pop(0)


# ── Endpointy ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Sprawdza stan serwisu."""
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        uptime_seconds=round(time.time() - state.start_time, 1),
        total_predictions=state.total_predictions
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Zwraca metryki serwisu (ostatnie 1000 predykcji)."""
    if not state.prediction_log:
        return {"message": "Brak danych"}

    latencies = [e["latency_ms"] for e in state.prediction_log]
    probs = [e["probability"] for e in state.prediction_log]

    return {
        "total_predictions": state.total_predictions,
        "recent_predictions": len(state.prediction_log),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies))
        },
        "churn_probability": {
            "mean": float(np.mean(probs)),
            "high_risk_pct": float(np.mean([p >= 0.7 for p in probs]))
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Predykcja churnu dla jednego klienta."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model nie jest załadowany")

    t0 = time.time()

    try:
        features = request.to_features()
        probability = float(state.model.predict_proba(features)[0][1])
    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd predykcji: {str(e)}")

    latency_ms = (time.time() - t0) * 1000
    state.total_predictions += 1

    background_tasks.add_task(
        log_prediction_async, request.customer_id, probability, latency_ms
    )

    return PredictionResponse(
        customer_id=request.customer_id,
        churn_probability=round(probability, 4),
        churn_prediction=probability >= 0.5,
        risk_level=classify_risk(probability),
        model_version=state.model_version,
        prediction_time_ms=round(latency_ms, 2),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
async def predict_batch(batch: BatchPredictionRequest):
    """Predykcja churnu dla wielu klientów naraz."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model nie jest załadowany")

    t0 = time.time()

    features = np.vstack([r.to_features() for r in batch.requests])
    probabilities = state.model.predict_proba(features)[:, 1]

    total_time_ms = (time.time() - t0) * 1000
    per_request_ms = total_time_ms / len(batch.requests)
    state.total_predictions += len(batch.requests)

    return [
        PredictionResponse(
            customer_id=req.customer_id,
            churn_probability=round(float(prob), 4),
            churn_prediction=float(prob) >= 0.5,
            risk_level=classify_risk(float(prob)),
            model_version=state.model_version,
            prediction_time_ms=round(per_request_ms, 2),
            timestamp=datetime.now().isoformat()
        )
        for req, prob in zip(batch.requests, probabilities)
    ]
```

### Krok 1.3: Uruchomienie serwisu

```bash
# Uruchom serwis
uvicorn src.serving.api:app --host 0.0.0.0 --port 8080 --reload

# W osobnym terminalu – testuj API
# Health check
curl http://localhost:8080/health | python3 -m json.tool

# Predykcja
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1001,
    "age": 35,
    "income": 55000,
    "tenure_months": 24,
    "num_products": 2,
    "has_credit_card": 1,
    "is_active": 1,
    "balance": 15000,
    "num_transactions_30d": 12
  }' | python3 -m json.tool

# Batch predykcja
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"customer_id": 1, "age": 25, "income": 30000, "tenure_months": 6,
       "num_products": 1, "has_credit_card": 0, "is_active": 1, "balance": 500, "num_transactions_30d": 3},
      {"customer_id": 2, "age": 55, "income": 80000, "tenure_months": 60,
       "num_products": 3, "has_credit_card": 1, "is_active": 1, "balance": 25000, "num_transactions_30d": 20}
    ]
  }' | python3 -m json.tool

# Otwórz dokumentację Swagger
# http://localhost:8080/docs
```

---

## Część 2: Testy API (30 min)

### Krok 2.1: Testy jednostkowe i integracyjne

Utwórz `tests/model/test_api.py`:

```python
# tests/model/test_api.py
"""Testy REST API serwisu ML."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.serving.api import app

client = TestClient(app)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_request():
    return {
        "customer_id": 1001,
        "age": 35.0,
        "income": 55000.0,
        "tenure_months": 24,
        "num_products": 2,
        "has_credit_card": 1,
        "is_active": 1,
        "balance": 15000.0,
        "num_transactions_30d": 12
    }

@pytest.fixture
def high_risk_request():
    """Klient z wysokim ryzykiem churnu."""
    return {
        "customer_id": 9999,
        "age": 65.0,
        "income": 20000.0,
        "tenure_months": 2,
        "num_products": 1,
        "has_credit_card": 0,
        "is_active": 0,
        "balance": 100.0,
        "num_transactions_30d": 0
    }

# ── Testy health check ────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data

    def test_health_model_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

# ── Testy predykcji ───────────────────────────────────────────────────────────

class TestPrediction:

    def test_predict_returns_200(self, valid_request):
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 200

    def test_predict_response_schema(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "model_version" in data
        assert "prediction_time_ms" in data

    def test_predict_probability_in_range(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_customer_id_preserved(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        assert data["customer_id"] == valid_request["customer_id"]

    def test_predict_risk_level_valid(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_churn_consistent_with_probability(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        expected_prediction = data["churn_probability"] >= 0.5
        assert data["churn_prediction"] == expected_prediction

    def test_predict_invalid_age_returns_422(self, valid_request):
        invalid = {**valid_request, "age": -5}
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_predict_invalid_income_returns_422(self, valid_request):
        invalid = {**valid_request, "income": -1000}
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, valid_request):
        incomplete = {k: v for k, v in valid_request.items() if k != "age"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_latency_reasonable(self, valid_request):
        response = client.post("/predict", json=valid_request)
        data = response.json()
        assert data["prediction_time_ms"] < 1000  # < 1 sekunda

# ── Testy batch predykcji ─────────────────────────────────────────────────────

class TestBatchPrediction:

    def test_batch_predict_returns_correct_count(self, valid_request, high_risk_request):
        batch = {"requests": [valid_request, high_risk_request]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_batch_predict_all_probabilities_valid(self, valid_request):
        batch = {"requests": [valid_request] * 10}
        response = client.post("/predict/batch", json=batch)
        data = response.json()
        for item in data:
            assert 0.0 <= item["churn_probability"] <= 1.0

    def test_batch_too_large_returns_422(self, valid_request):
        batch = {"requests": [valid_request] * 501}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 422

# ── Testy metryk ──────────────────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_endpoint_exists(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_after_predictions(self, valid_request):
        # Wykonaj kilka predykcji
        for _ in range(5):
            client.post("/predict", json=valid_request)

        response = client.get("/metrics")
        data = response.json()
        assert "total_predictions" in data
        assert data["total_predictions"] >= 5
```

```bash
# Uruchom testy API
pytest tests/model/test_api.py -v

# Uruchom z pokryciem kodu
pytest tests/model/test_api.py -v --cov=src/serving --cov-report=term-missing
```

---

## Część 3: Konteneryzacja z Docker (30 min)

### Krok 3.1: Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

LABEL maintainer="mlops-lab"
LABEL description="Churn Prediction API"

WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalacja zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.27.0 \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    numpy==1.26.0 \
    joblib==1.3.0 \
    pydantic==2.6.0

# Kopiowanie kodu
COPY src/ ./src/
COPY models/ ./models/

# Użytkownik bez uprawnień root
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--log-level", "info"]
```

```bash
# Zbuduj obraz
docker build -t churn-api:latest .

# Uruchom kontener
docker run -d \
  --name churn-api \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  churn-api:latest

# Sprawdź logi
docker logs churn-api -f

# Przetestuj
curl http://localhost:8080/health

# Zatrzymaj kontener
docker stop churn-api && docker rm churn-api
```

### Krok 3.2: Docker Compose z monitoringiem

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
    environment:
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Nginx jako reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - ml-api
```

```nginx
# nginx.conf
upstream ml_api {
    server ml-api:8080;
}

server {
    listen 80;

    location /api/ {
        proxy_pass http://ml_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }

    location /health {
        proxy_pass http://ml_api/health;
    }
}
```

```bash
# Uruchom stack
docker-compose up -d

# Sprawdź status
docker-compose ps

# Zatrzymaj
docker-compose down
```

---

## Część 4: Pomiar wydajności (10 min)

### Krok 4.1: Prosty benchmark

```python
# scripts/benchmark_api.py
"""Benchmark wydajności API."""

import httpx
import time
import statistics
import asyncio
import json

BASE_URL = "http://localhost:8080"

SAMPLE_REQUEST = {
    "customer_id": 1001,
    "age": 35.0,
    "income": 55000.0,
    "tenure_months": 24,
    "num_products": 2,
    "has_credit_card": 1,
    "is_active": 1,
    "balance": 15000.0,
    "num_transactions_30d": 12
}

async def benchmark_single_predictions(n_requests: int = 100):
    """Mierzy latencję pojedynczych predykcji."""
    latencies = []

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        for i in range(n_requests):
            request = {**SAMPLE_REQUEST, "customer_id": i + 1}
            t0 = time.time()
            response = await client.post("/predict", json=request)
            latency_ms = (time.time() - t0) * 1000
            assert response.status_code == 200
            latencies.append(latency_ms)

    print(f"\n=== Benchmark: {n_requests} pojedynczych predykcji ===")
    print(f"  p50:  {statistics.median(latencies):.2f} ms")
    print(f"  p95:  {sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms")
    print(f"  p99:  {sorted(latencies)[int(0.99 * len(latencies))]:.2f} ms")
    print(f"  mean: {statistics.mean(latencies):.2f} ms")
    print(f"  min:  {min(latencies):.2f} ms")
    print(f"  max:  {max(latencies):.2f} ms")

async def benchmark_batch_predictions(batch_sizes: list[int] = [1, 10, 50, 100]):
    """Mierzy throughput batch predykcji."""
    print("\n=== Benchmark: Batch predykcje ===")
    print(f"{'Batch size':>12} | {'Czas (ms)':>10} | {'ms/rekord':>10} | {'Rekordów/s':>12}")
    print("-" * 55)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        for batch_size in batch_sizes:
            requests = [
                {**SAMPLE_REQUEST, "customer_id": i + 1}
                for i in range(batch_size)
            ]
            batch = {"requests": requests}

            t0 = time.time()
            response = await client.post("/predict/batch", json=batch)
            elapsed_ms = (time.time() - t0) * 1000

            assert response.status_code == 200
            ms_per_record = elapsed_ms / batch_size
            records_per_sec = 1000 / ms_per_record

            print(f"{batch_size:>12} | {elapsed_ms:>10.1f} | {ms_per_record:>10.2f} | {records_per_sec:>12.0f}")

async def main():
    # Sprawdź czy serwis działa
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/health")
        if response.status_code != 200:
            print("❌ Serwis nie działa!")
            return
        print(f"✅ Serwis działa: {response.json()['model_version']}")

    await benchmark_single_predictions(n_requests=100)
    await benchmark_batch_predictions(batch_sizes=[1, 10, 50, 100, 200])

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# Uruchom benchmark (serwis musi być uruchomiony)
python scripts/benchmark_api.py
```

---

## Zadania do samodzielnego wykonania

1. **Dodaj endpoint** `GET /predict/history/{customer_id}` zwracający historię predykcji dla klienta.
2. **Dodaj autentykację** API Key – żądania bez nagłówka `X-API-Key` powinny zwracać 401.
3. **Napisz test** sprawdzający, że batch predykcja jest szybsza niż N pojedynczych predykcji.
4. **Dodaj middleware** logujący każde żądanie (metoda, ścieżka, status, czas).

## Pytania kontrolne

1. Dlaczego używamy `BackgroundTasks` do logowania predykcji?
2. Jaka jest różnica między `TestClient` a prawdziwym klientem HTTP?
3. Kiedy warto używać batch API zamiast pojedynczych żądań?
4. Co robi `HEALTHCHECK` w Dockerfile i dlaczego jest ważny?

## Podsumowanie

W tym laboratorium:
- ✅ Zbudowałeś REST API z FastAPI z walidacją Pydantic
- ✅ Napisałeś testy jednostkowe i integracyjne dla API
- ✅ Skonteneryzowałeś serwis w Dockerze
- ✅ Zmierzyłeś wydajność API (latencja, throughput)
