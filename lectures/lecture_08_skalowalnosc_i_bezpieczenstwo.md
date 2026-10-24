# Wykład 8: Skalowalność, Bezpieczeństwo i Optymalizacja Kosztów w MLOps

## Cel wykładu
Po tym wykładzie student:
- rozumie wzorce skalowania systemów ML,
- zna techniki optymalizacji modeli (quantization, pruning, distillation),
- potrafi zabezpieczyć system ML (IAM, szyfrowanie, audyt),
- zna strategie optymalizacji kosztów w chmurze.

### Skrypty demonstracyjne

Skrypty demonstracyjne znajdują się w `lectures/scripts/lecture_08/`. Uruchamiaj je z katalogu głównego repozytorium (`python lectures/scripts/lecture_08/<skrypt>`). Wymagają tylko pakietów z `requirements.txt` (numpy, scikit-learn, pandas, pydantic).

| Skrypt | Sekcja | Co pokazuje |
|---|---|---|
| `01_batching_latency_throughput.py` | 2.3 | Wpływ rozmiaru batcha na latencję i przepustowość, symulacja dynamic batchingu (p50/p95/p99) |
| `02_adversarial_and_validation.py` | 4.2 | Atak FGSM na model liniowy, walidacja pydantic + z-score, model extraction |
| `03_pii_masking.py` | 4.5 | Maskowanie PII w logach, pseudonimizacja HMAC, k-anonimowość |

---

## 1. Skalowalność systemów ML

### Wymiary skalowalności

```mermaid
graph TB
    SCALE[Skalowalność\nSystemu ML] --> DS[Data Scale\nWolumen danych]
    SCALE --> MS[Model Scale\nRozmiar modelu]
    SCALE --> RS[Request Scale\nLiczba żądań]
    SCALE --> TS[Training Scale\nCzas treningu]

    DS --> DS1[Distributed Storage\nParquet, Delta Lake]
    DS --> DS2[Distributed Processing\nSpark, Dask]
    MS --> MS1[Model Compression\nQuantization, Pruning]
    MS --> MS2[Distributed Training\nHorovod, DeepSpeed]
    RS --> RS1[Horizontal Scaling\nKubernetes HPA]
    RS --> RS2[Load Balancing\nnginx, Envoy]
    TS --> TS1[GPU Clusters\nVertex AI Training]
    TS --> TS2[Hyperparameter\nParallel Search]
```

### Skalowanie wertykalne vs horyzontalne

| Cecha | Wertykalne (scale up) | Horyzontalne (scale out) |
|---|---|---|
| Mechanizm | Większa maszyna (więcej CPU/RAM/GPU) | Więcej replik za load balancerem |
| Limit | Największa dostępna instancja | Praktycznie brak (koszt, limity quota) |
| Dostępność | Single point of failure | Wysoka – awaria repliki nie zatrzymuje usługi |
| Wymagania wobec aplikacji | Brak | Bezstanowość (stateless), współdzielony storage/cache |
| Typowe użycie w ML | Trening dużego modelu na 1 węźle, duży model w RAM | Serwowanie online, batch scoring, przetwarzanie danych |
| Czas reakcji | Minuty (restart) | Sekundy–minuty (nowe pody, cold start) |

### Kluczowe pojęcia i prawa

- **Latencja** (czas odpowiedzi jednego żądania, mierzona percentylami p50/p95/p99) vs **przepustowość** (żądania/predykcje na sekundę). Optymalizacja jednej często pogarsza drugą – patrz batching w sekcji 2.3.
- **Prawo Little'a:** `L = λ · W` – średnia liczba żądań w systemie = tempo napływu × średni czas przebywania. Przy 200 RPS i latencji 50 ms w systemie jest średnio 10 żądań „w locie” → potrzebujemy co najmniej tylu równoległych slotów obsługi.
- **Prawo Amdahla:** jeśli tylko część `p` pracy da się zrównoleglić, maksymalne przyspieszenie na `N` workerach to `1 / ((1 - p) + p/N)`. Przy `p = 0.9` nawet nieskończenie wiele GPU daje maks. 10×. Dlatego w distributed training kluczowa jest redukcja komunikacji (all-reduce) i I/O.
- **Utilization a kolejki:** gdy wykorzystanie serwera zbliża się do 100%, czas oczekiwania w kolejce rośnie nieliniowo (model M/M/1: `W ~ 1/(1-ρ)`). Stąd cele HPA rzędu 60–70% CPU, a nie 95%.

```
latencja
  ^                                   |
  |                                  /
  |                                 /    ← „kolano” krzywej: powyżej ~70-80%
  |                              _/        wykorzystania kolejka eksploduje
  |                         __--
  |  ______________-----‾‾‾
  +-------------------------------------> wykorzystanie (ρ)
  0%            50%           80%  100%
```

---

## 2. Skalowanie horyzontalne serwisu ML

### Kubernetes Horizontal Pod Autoscaler (HPA)

```yaml
# kubernetes/ml-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-predictor
  labels:
    app: churn-predictor
    version: v2.1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-predictor
  template:
    metadata:
      labels:
        app: churn-predictor
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: churn-api
          image: gcr.io/my-project/churn-api:v2.1.0
          ports:
            - containerPort: 8080
            - containerPort: 8000  # metrics
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          env:
            - name: MODEL_VERSION
              value: "v2.1.0"
            - name: LOG_LEVEL
              value: "INFO"
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-predictor
  minReplicas: 2
  maxReplicas: 20
  metrics:
    # Skalowanie na podstawie CPU
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Skalowanie na podstawie niestandardowej metryki (RPS)
    - type: Pods
      pods:
        metric:
          name: ml_predictions_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minut przed skalowaniem w dół
```

> ℹ️ **Uwaga do manifestu:** `prometheus.io/port: "8000"` wskazuje osobny port metryk – w kontenerze musi faktycznie działać endpoint `/metrics` (np. `prometheus_client.start_http_server(8000)`). Metryka `ml_predictions_per_second` typu `Pods` wymaga zainstalowanego **Prometheus Adapter** (lub KEDA), który wystawia ją przez Custom Metrics API – sam HPA nie czyta Prometheusa.

### Jak działa HPA – algorytm

HPA co ~15 s liczy żądaną liczbę replik:

```
desiredReplicas = ceil( currentReplicas × currentMetricValue / targetMetricValue )
```

Przykład: 4 repliki, średnie CPU 105%, cel 70% → `ceil(4 × 105/70) = 6`. Przy wielu metrykach HPA wybiera **największą** wyliczoną wartość. Sekcja `behavior` ogranicza tempo zmian (zapobiega „flappingowi”).

```mermaid
sequenceDiagram
    participant P as Prometheus / metrics-server
    participant H as HPA controller
    participant D as Deployment
    participant S as Scheduler / Cluster Autoscaler
    loop co 15 s
        H->>P: pobierz CPU i ml_predictions_per_second
        P-->>H: średnie wartości per pod
        H->>H: desired = ceil(current × metric/target)
        H->>D: zmień replicas (z limitami behavior)
    end
    D->>S: nowe pody Pending
    S->>S: brak miejsca → dodaj węzeł (Cluster Autoscaler)
    Note over D: readinessProbe OK → pod dostaje ruch
```

### Porównanie mechanizmów autoskalowania

| Mechanizm | Co skaluje | Sygnał | Scale-to-zero | Kiedy używać |
|---|---|---|---|---|
| HPA | Liczbę podów | CPU/RAM, custom metrics | Nie (min 1) | Stały ruch online |
| VPA | Requests/limits poda | Historyczne zużycie | Nie | Right-sizing, joby batch |
| KEDA | Liczbę podów | Kolejki (Kafka, Pub/Sub), Prometheus, cron | Tak | Ruch zdarzeniowy, kolejki inferencji |
| Knative / KServe | Liczbę podów | Współbieżność żądań | Tak | Serverless inference, wiele rzadko używanych modeli |
| Cluster Autoscaler | Liczbę węzłów | Pody Pending | – | Zawsze razem z HPA/KEDA |

### Cold start modeli ML

Skalowanie modeli jest wolniejsze niż zwykłych API: pobranie obrazu (często kilka GB z CUDA), pobranie wag (setki MB – dziesiątki GB), załadowanie do GPU, warmup (kompilacja JIT, alokacja pamięci). Typowe techniki:

- **małe obrazy** (multi-stage build, `python:3.x-slim`, brak narzędzi buildowych),
- **wagi poza obrazem** (PVC/GCS + cache na węźle) albo odwrotnie – wypieczone w obrazie dla przewidywalności,
- **`startupProbe`** zamiast długiego `initialDelaySeconds` w `livenessProbe` – nie zabijamy poda w trakcie ładowania modelu,
- **zapas pojemności** (`minReplicas` > 1, overprovisioning podami o niskim priorytecie),
- **predictive scaling** – skalowanie przed znanym szczytem (CronJob / KEDA cron trigger).

```yaml
# fragment kontenera – startupProbe dla wolno ładującego się modelu
startupProbe:
  httpGet:
    path: /health/ready
    port: 8080
  periodSeconds: 5
  failureThreshold: 60     # do 5 minut na załadowanie modelu
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  periodSeconds: 10
```

### Load Testing z Locust

```python
# locustfile.py
from locust import HttpUser, task, between
import json
import random

class MLServiceUser(HttpUser):
    """Symuluje użytkownika serwisu ML."""
    
    wait_time = between(0.1, 1.0)  # czas między żądaniami
    
    def on_start(self):
        """Inicjalizacja użytkownika."""
        # Sprawdź health przed testami
        response = self.client.get("/health")
        assert response.status_code == 200
    
    @task(10)  # waga 10 - najczęstsze żądanie
    def predict_single(self):
        """Test pojedynczej predykcji."""
        payload = {
            "user_id": random.randint(1, 100000),
            "age": random.uniform(18, 80),
            "income": random.uniform(20000, 150000),
            "tenure_months": random.randint(1, 120),
            "num_products": random.randint(1, 5)
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if not (0 <= data['churn_probability'] <= 1):
                    response.failure("Nieprawidłowe prawdopodobieństwo")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)  # waga 2 - rzadsze żądanie
    def predict_batch(self):
        """Test predykcji wsadowej."""
        batch_size = random.randint(10, 100)
        payload = [
            {
                "user_id": i,
                "age": random.uniform(18, 80),
                "income": random.uniform(20000, 150000),
                "tenure_months": random.randint(1, 120),
                "num_products": random.randint(1, 5)
            }
            for i in range(batch_size)
        ]
        
        self.client.post("/predict/batch", json=payload)
    
    @task(1)
    def health_check(self):
        """Test health check."""
        self.client.get("/health")

# Uruchomienie:
# locust -f locustfile.py --host=http://localhost:8080 --users=100 --spawn-rate=10
```

### Rodzaje testów wydajnościowych

| Test | Profil ruchu | Pytanie, na które odpowiada |
|---|---|---|
| Load test | Oczekiwany ruch (np. 200 RPS) przez 15–30 min | Czy spełniamy SLO przy normalnym obciążeniu? |
| Stress test | Rosnący ruch aż do awarii | Gdzie jest punkt załamania i jak system się degraduje? |
| Spike test | Nagły skok 10× | Czy autoscaling zdąży? Jak długi jest cold start? |
| Soak test | Stały ruch przez wiele godzin | Wycieki pamięci, rosnące kolejki, degradacja cache |

**Dobre praktyki:** testuj na środowisku zbliżonym do produkcji (te same limity CPU/RAM), używaj realistycznego rozkładu danych wejściowych (np. próbki z logów po anonimizacji), raportuj percentyle, nie średnią, i zapisuj wyniki jako artefakt w CI, aby wykrywać regresje wydajności między wersjami modelu.

### 2.3 Batching – latencja vs przepustowość

Każde wywołanie modelu ma **koszt stały** (narzut frameworka, serializacja, uruchomienie kernela GPU) i **koszt zmienny** (proporcjonalny do liczby przykładów). Grupowanie żądań w batch rozkłada koszt stały na wiele predykcji – na GPU różnica bywa 10–100×.

```mermaid
graph LR
    R1[Żądanie 1] --> Q[Kolejka\nmax_batch=32\nmax_wait=5 ms]
    R2[Żądanie 2] --> Q
    R3[Żądanie N] --> Q
    Q -->|batch gdy pełny\nlub minął timeout| M[Model\njedno wywołanie]
    M --> S[Rozdzielenie\nodpowiedzi]
    S --> O1[Odp. 1]
    S --> O2[Odp. 2]
    S --> O3[Odp. N]
```

| Tryb | Latencja | Przepustowość | Przykłady narzędzi |
|---|---|---|---|
| Brak batchingu | Najniższa przy małym ruchu | Niska | Prosty FastAPI |
| Dynamic (server-side) batching | + `max_wait` | Wysoka | Triton Inference Server, TorchServe, BentoML, Ray Serve |
| Continuous batching (LLM) | Niska mimo różnej długości odpowiedzi | Bardzo wysoka | vLLM, TGI |
| Batch prediction (offline) | Minuty–godziny | Maksymalna | Spark, Vertex AI Batch Prediction |

> **▶️ Demo na żywo:** `python lectures/scripts/lecture_08/01_batching_latency_throughput.py`  
> Skrypt mierzy przepustowość RandomForest dla batchy 1–512, a następnie symuluje kolejkę przy 800 RPS. Bez batchingu pojedynczy worker nie nadąża i latencja rośnie do sekund; z `max_batch=8, max_wait=2 ms` p99 spada do ~10 ms, a dłuższe okno (20 ms) znów podnosi latencję.

**Trade-off:** `max_wait` dobieramy tak, aby `max_wait + czas_wywołania(max_batch)` mieścił się w budżecie latencji SLO (np. p99 < 100 ms). Batching ma sens głównie dla modeli z dużym kosztem stałym (sieci neuronowe, GPU); dla bardzo lekkich modeli (regresja logistyczna) zysk jest niewielki.

---

## 3. Optymalizacja modeli

### Przegląd technik optymalizacji modeli

| Technika | Typowa redukcja rozmiaru | Przyspieszenie | Utrata jakości | Koszt wdrożenia |
|---|---|---|---|---|
| Dynamic quantization (INT8) | ~4× dla wag Linear | 1.5–3× na CPU | Zwykle <1 p.p. | Niski (1 linia kodu) |
| Static quantization / QAT | ~4× | 2–4× | Bardzo mała (QAT) | Średni (kalibracja / ponowny trening) |
| Pruning (unstructured) | 2–10× (po kompresji) | Małe bez sparse kerneli | Zależy od stopnia | Średni |
| Pruning (structured – kanały/warstwy) | 1.5–3× | Realne | Wymaga fine-tuningu | Średni |
| Knowledge distillation | 2–100× | 2–100× | 1–3 p.p. | Wysoki (nowy trening) |
| Eksport do ONNX Runtime / TensorRT | – | 1.5–5× | Brak (FP32) / mała (FP16) | Niski–średni |
| Caching | – | ∞ dla trafień | Ryzyko nieaktualnych wyników | Niski |

```mermaid
graph TD
    START[Model za wolny / za drogi?] --> PROF{Profiling:\ngdzie jest czas?}
    PROF -->|I/O, preprocessing, sieć| IO[Optymalizuj pipeline\nfeature fetch, serializację]
    PROF -->|inferencja| HW{Sprzęt?}
    HW -->|CPU| Q[Dynamic quantization\n+ ONNX Runtime]
    HW -->|GPU| G[FP16/INT8 + TensorRT\n+ dynamic batching]
    Q --> OK{SLO spełnione?}
    G --> OK
    OK -->|nie| D[Distillation /\nmniejsza architektura]
    OK -->|tak| END[Wdrożenie + monitoring jakości]
    D --> END
```

### 3.1 Quantization – zmniejszenie precyzji

**Teoria.** Quantization mapuje liczby zmiennoprzecinkowe na liczby całkowite: `q = round(x / scale) + zero_point`, gdzie `scale = (max - min) / 255` dla INT8. Wyróżniamy:
- **dynamic quantization** – wagi kwantyzowane offline, aktywacje „w locie” (dobre dla Linear/LSTM na CPU),
- **static (post-training) quantization** – zakresy aktywacji kalibrowane na próbce danych,
- **quantization-aware training (QAT)** – symulacja kwantyzacji podczas treningu, najmniejsza utrata jakości.

Pułapki: (1) zawsze porównuj metryki jakości (AUC, accuracy) przed i po – nie tylko latencję; (2) dla bardzo małych modeli (jak poniżej) narzut kwantyzacji może zjeść zysk; (3) nie każdy sprzęt ma szybkie instrukcje INT8.

```python
import torch
import torch.nn as nn
import numpy as np
import time

class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def benchmark_model(model, input_data: torch.Tensor, n_runs: int = 1000) -> dict:
    """Mierzy wydajność modelu."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_data)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            output = model(input_data)
    elapsed = time.time() - start
    
    return {
        "avg_latency_ms": elapsed / n_runs * 1000,
        "throughput_rps": n_runs / elapsed
    }

# Oryginalny model (FP32)
model_fp32 = SimpleModel()
input_data = torch.randn(1, 20)

# Dynamic Quantization (INT8)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8
)

# Porównanie
fp32_stats = benchmark_model(model_fp32, input_data)
int8_stats = benchmark_model(model_int8, input_data)

# Rozmiar modeli
import io
def get_model_size_mb(model) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024 / 1024

fp32_size = get_model_size_mb(model_fp32)
int8_size = get_model_size_mb(model_int8)

print("=== Porównanie FP32 vs INT8 ===")
print(f"FP32: {fp32_size:.2f} MB, latencja: {fp32_stats['avg_latency_ms']:.3f}ms")
print(f"INT8: {int8_size:.2f} MB, latencja: {int8_stats['avg_latency_ms']:.3f}ms")
print(f"Redukcja rozmiaru: {(1 - int8_size/fp32_size)*100:.0f}%")
print(f"Przyspieszenie: {fp32_stats['avg_latency_ms']/int8_stats['avg_latency_ms']:.1f}x")
```

### 3.2 Knowledge Distillation

**Teoria (Hinton i in., 2015).** Student uczy się nie tylko z twardych etykiet, ale z **miękkich prawdopodobieństw** nauczyciela, które niosą informację o podobieństwie klas („dark knowledge”). Funkcja straty: `L = α · CE(y, p_student) + (1-α) · T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))`, gdzie `T` – temperatura wygładzająca rozkład.

```mermaid
graph LR
    subgraph "Teacher Model (duży, dokładny)"
        T[RandomForest\n500 drzew\nAUC=0.92]
    end
    
    subgraph "Distillation"
        SOFT[Soft Labels\nprawdopodobieństwa\nnauczyciela]
    end
    
    subgraph "Student Model (mały, szybki)"
        S[LogisticRegression\nlub mała sieć\nAUC=0.89]
    end
    
    T -->|generuje| SOFT
    SOFT -->|trenuje| S
```

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

# Dane
X, y = make_classification(n_samples=50000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Teacher: duży Random Forest
teacher = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
teacher.fit(X_train, y_train)

# Generowanie soft labels (temperatura T=2 wygładza rozkład)
T = 2.0
teacher_probs = teacher.predict_proba(X_train)
soft_labels = teacher_probs[:, 1]  # prawdopodobieństwo klasy pozytywnej

# Student: mała regresja logistyczna trenowana na soft labels
# UWAGA: progowanie soft_labels > 0.5 zamienia je z powrotem w twarde etykiety –
# to w praktyce "model mimicry", nie pełna distillation. Wersja na miękkich
# etykietach znajduje się w bloku poniżej (regresja na logitach nauczyciela).
student = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
student.fit(X_train, soft_labels > 0.5)

# Porównanie
teacher_auc = roc_auc_score(y_test, teacher.predict_proba(X_test)[:, 1])
student_auc = roc_auc_score(y_test, student.predict_proba(X_test)[:, 1])

# Latencja
sample = X_test[:1]
t0 = time.time()
for _ in range(1000): teacher.predict_proba(sample)
teacher_latency = (time.time() - t0) / 1000 * 1000

t0 = time.time()
for _ in range(1000): student.predict_proba(sample)
student_latency = (time.time() - t0) / 1000 * 1000

print("=== Knowledge Distillation ===")
print(f"Teacher: AUC={teacher_auc:.4f}, latencja={teacher_latency:.2f}ms")
print(f"Student: AUC={student_auc:.4f}, latencja={student_latency:.2f}ms")
print(f"Utrata AUC: {(teacher_auc - student_auc):.4f}")
print(f"Przyspieszenie: {teacher_latency/student_latency:.1f}x")
```

Właściwa distillation na miękkich etykietach – student (model liniowy) regresuje **logity** nauczyciela, dzięki czemu uczy się także „pewności” nauczyciela:

```python
from sklearn.linear_model import Ridge

eps = 1e-4
p = np.clip(teacher_probs[:, 1], eps, 1 - eps)
teacher_logits = np.log(p / (1 - p)) / T          # logity wygładzone temperaturą

soft_student = Ridge(alpha=1.0).fit(X_train, teacher_logits)
soft_scores = soft_student.predict(X_test)         # ranking wystarcza do AUC
print(f"Soft-student AUC: {roc_auc_score(y_test, soft_scores):.4f}")
```

**Kiedy distillation się opłaca:** wysoki ruch (koszt treningu amortyzuje się na milionach predykcji), twarde ograniczenia latencji (mobile, edge, real-time bidding), dostęp do dużej ilości **nieoznaczonych** danych (nauczyciel etykietuje je za darmo). **Pułapka:** student dziedziczy błędy i biasy nauczyciela – waliduj go na danych z prawdziwymi etykietami, także w podgrupach (fairness).

### 3.3 Caching predykcji

| Poziom cache | Co przechowuje | Zysk | Ryzyko |
|---|---|---|---|
| Cache predykcji (Redis) | Wynik dla identycznego wejścia | Brak wywołania modelu | Nieaktualność po zmianie modelu → klucz musi zawierać wersję modelu |
| Cache cech (feature cache) | Cechy online z Feature Store | Mniej zapytań do DB | Staleness cech |
| Cache embeddingów | Wektory dla tekstów/obrazów | Duży – embedding bywa najdroższy | Zużycie pamięci |
| Precomputed batch | Predykcje liczone nocą dla wszystkich użytkowników | Latencja = odczyt z KV | Brak reakcji na zmiany w ciągu dnia |

> ⚠️ Klucz cache powinien zawierać **wersję modelu** (np. `pred:v2.1.0:<hash>`), inaczej po wdrożeniu nowego modelu przez TTL serwujemy stare predykcje. MD5 jest tu akceptowalny (nie jest użyty kryptograficznie), ale do pseudonimizacji danych osobowych używaj HMAC (sekcja 4.5).

```python
import hashlib
import json
import time
from functools import lru_cache
from typing import Optional
import redis

class PredictionCache:
    """Cache predykcji z Redis."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 3600,
        cache_threshold: float = 0.05  # cache tylko gdy prob < 0.05 lub > 0.95
    ):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.threshold = cache_threshold
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, features: dict) -> str:
        """Tworzy klucz cache z cech."""
        # Zaokrąglij wartości numeryczne (unikaj problemów z float)
        rounded = {k: round(v, 2) if isinstance(v, float) else v 
                   for k, v in sorted(features.items())}
        return f"pred:{hashlib.md5(json.dumps(rounded).encode()).hexdigest()}"
    
    def get(self, features: dict) -> Optional[float]:
        """Pobiera predykcję z cache."""
        key = self._make_key(features)
        cached = self.redis.get(key)
        
        if cached is not None:
            self._hits += 1
            return float(cached)
        
        self._misses += 1
        return None
    
    def set(self, features: dict, probability: float):
        """Zapisuje predykcję do cache (tylko dla pewnych predykcji)."""
        # Cache tylko gdy model jest pewny (unikamy cache dla granicznych przypadków)
        if probability < self.threshold or probability > (1 - self.threshold):
            key = self._make_key(features)
            self.redis.setex(key, self.ttl, str(probability))
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

# Użycie w serwisie
class CachedMLService:
    def __init__(self, model, cache: PredictionCache):
        self.model = model
        self.cache = cache
    
    def predict(self, features: dict) -> dict:
        # Sprawdź cache
        cached_prob = self.cache.get(features)
        
        if cached_prob is not None:
            return {
                "probability": cached_prob,
                "source": "cache",
                "latency_ms": 0.1
            }
        
        # Oblicz predykcję
        t0 = time.time()
        X = [[features[k] for k in sorted(features.keys())]]
        probability = float(self.model.predict_proba(X)[0][1])
        latency = (time.time() - t0) * 1000
        
        # Zapisz do cache
        self.cache.set(features, probability)
        
        return {
            "probability": probability,
            "source": "model",
            "latency_ms": latency
        }
```

---

## 4. Bezpieczeństwo systemów ML

### 4.1 Zagrożenia specyficzne dla ML

```mermaid
graph TD
    THREATS[Zagrożenia ML] --> ADV[Adversarial Attacks\nManipulacja wejść]
    THREATS --> POISON[Data Poisoning\nZatrucie danych treningowych]
    THREATS --> EXTRACT[Model Extraction\nKradzież modelu]
    THREATS --> INVERT[Model Inversion\nOdtworzenie danych treningowych]
    THREATS --> PRIV[Privacy Leakage\nWyciek danych osobowych]

    ADV --> DEF1[Adversarial Training\nInput Validation]
    POISON --> DEF2[Data Provenance\nAnomaly Detection]
    EXTRACT --> DEF3[Rate Limiting\nQuery Monitoring]
    INVERT --> DEF4[Differential Privacy\nOutput Perturbation]
    PRIV --> DEF5[Data Anonymization\nFederated Learning]
```

### Powierzchnia ataku w cyklu życia ML

```mermaid
graph LR
    D[Dane\ntreningowe] -->|data poisoning| T[Trening]
    L[Biblioteki,\nmodele z Hub] -->|supply chain:\nzłośliwy pickle| T
    T --> R[Model\nRegistry]
    R -->|kradzież artefaktu,\nbrak podpisu| S[Serving API]
    U[Użytkownik /\natakujący] -->|adversarial input,\nprompt injection| S
    S -->|model extraction,\nmembership inference| U
    S --> LOG[Logi]
    LOG -->|wyciek PII| X[Osoby\nnieuprawnione]
```

| Zagrożenie | Etap | Cel atakującego | Przykład | Główne zabezpieczenia |
|---|---|---|---|---|
| Evasion (adversarial) | Inference | Ominięcie modelu | Fraudster zmienia kwotę 10 000 → 9 990 | Adversarial training, reguły biznesowe, ensemble |
| Data poisoning | Trening | Wpływ na zachowanie modelu / backdoor | Fałszywe recenzje w danych do retreningu | Data provenance, walidacja danych, ręczna akceptacja |
| Model extraction | Inference | Kradzież IP | Tysiące zapytań i trening kopii | Rate limiting, zwracanie tylko etykiet, watermarking |
| Membership inference | Inference | Czy rekord był w treningu? | Ustalenie, że pacjent był w badaniu | Differential privacy, regularyzacja |
| Supply chain | Build | Wykonanie kodu | `pickle.load` złośliwego modelu z internetu | `safetensors`/ONNX, skanowanie, podpisy (Sigstore) |
| Prompt injection (LLM) | Inference | Przejęcie instrukcji | „Zignoruj poprzednie polecenia…” | Separacja danych i instrukcji, filtry, least privilege narzędzi |

Użyteczne katalogi zagrożeń: **MITRE ATLAS** (taktyki ataków na ML), **OWASP Machine Learning Security Top 10** oraz **OWASP Top 10 for LLM Applications**.

### 4.2 Walidacja i sanityzacja wejść

**Atak adversarialny (FGSM).** Dla modelu z gradientem wejścia `∇x L` atakujący tworzy `x_adv = x + ε · sign(∇x L)`. Dla modelu liniowego gradient to po prostu wektor wag `w`, więc wystarczy przesunąć każdą cechę o `ε` „w stronę” klasy negatywnej. Kluczowa obserwacja: perturbacja jest **mała w każdej cesze**, więc przechodzi przez proste walidatory zakresów.

```
  cecha 2
    ^         klasa "fraud"
    |      x •
    |        \  ε·sign(w)
    |---------\----------- granica decyzyjna
    |          • x_adv     klasa "OK"
    +---------------------> cecha 1
```

Walidacja wejść to **pierwsza warstwa** obrony (defense in depth): odrzuca błędne typy, NaN, wartości spoza domeny i oczywiste anomalie, ale nie zatrzyma wyrafinowanego ataku. Kolejne warstwy: reguły biznesowe, monitoring rozkładu wejść (wykład 6), adversarial training, człowiek w pętli dla decyzji wysokiego ryzyka.

> ℹ️ Poniższy kod używa API **pydantic v1** (`@validator`). W projekcie (`pydantic>=2.8`) używaj `@field_validator` z `@classmethod` – przykład w skrypcie `02_adversarial_and_validation.py`. Zauważ też, że `Field(ge=18)` już wymusza wiek ≥ 18, więc warunek w walidatorze jest redundantny (pozostawiony dla ilustracji).

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SecurePredictionRequest(BaseModel):
    """Bezpieczny schemat żądania z walidacją."""
    
    user_id: int = Field(..., gt=0, lt=10_000_000)
    age: float = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0, le=10_000_000)
    tenure_months: int = Field(..., ge=0, le=600)
    num_products: int = Field(..., ge=1, le=20)
    
    @validator('age')
    def age_must_be_adult(cls, v):
        if v < 18:
            raise ValueError("Wiek musi być >= 18")
        return round(v, 1)  # zaokrąglij do 1 miejsca
    
    @validator('income')
    def income_sanity_check(cls, v):
        if v > 5_000_000:
            logger.warning(f"Podejrzanie wysoki dochód: {v}")
        return round(v, 2)

class InputValidator:
    """Walidator wejść dla serwisu ML."""
    
    def __init__(self, reference_stats: dict):
        """
        Args:
            reference_stats: statystyki z danych treningowych
                {'age': {'mean': 40, 'std': 10, 'min': 18, 'max': 90}, ...}
        """
        self.stats = reference_stats
        self._anomaly_count = 0
    
    def validate_features(self, features: dict) -> tuple[bool, list[str]]:
        """
        Sprawdza czy cechy są w rozsądnym zakresie.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        for feature, value in features.items():
            if feature not in self.stats:
                continue
            
            stat = self.stats[feature]
            mean, std = stat['mean'], stat['std']
            
            # Sprawdź czy wartość jest w zakresie mean ± 5*std
            z_score = abs(value - mean) / max(std, 1e-6)
            if z_score > 5:
                warnings.append(
                    f"Cecha '{feature}' ma wartość {value:.2f} "
                    f"(z-score={z_score:.1f} > 5)"
                )
                self._anomaly_count += 1
        
        is_valid = len(warnings) == 0
        if not is_valid:
            logger.warning(f"Anomalie w wejściu: {warnings}")
        
        return is_valid, warnings

# Przykład użycia
reference_stats = {
    'age': {'mean': 40, 'std': 10, 'min': 18, 'max': 90},
    'income': {'mean': 50000, 'std': 15000, 'min': 0, 'max': 500000},
    'tenure_months': {'mean': 24, 'std': 18, 'min': 0, 'max': 240}
}

validator = InputValidator(reference_stats)

# Normalne wejście
normal_features = {'age': 35, 'income': 55000, 'tenure_months': 24}
is_valid, warnings = validator.validate_features(normal_features)
print(f"Normalne: valid={is_valid}, warnings={warnings}")

# Podejrzane wejście (potencjalny atak adversarialny)
suspicious_features = {'age': 999, 'income': -1, 'tenure_months': 10000}
is_valid, warnings = validator.validate_features(suspicious_features)
print(f"Podejrzane: valid={is_valid}, warnings={warnings}")
```

> ⚠️ W przykładzie zmienna `validator` przesłania zaimportowany dekorator `validator` z pydantic – w prawdziwym module nazwij ją np. `input_validator`.

> **▶️ Demo na żywo:** `python lectures/scripts/lecture_08/02_adversarial_and_validation.py`  
> Skrypt pokazuje, że perturbacja ε=1.0 odwraca 100% wykrytych fraudów, a walidator z-score zatrzymuje je dopiero przy ε=5. Dodatkowo kopia modelu wytrenowana na zaledwie 50 odpowiedziach API zgadza się z oryginałem w ~98% – to argument za rate limitingiem i ograniczaniem informacji w odpowiedzi.

### 4.3 Rate Limiting i ochrona API

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
from collections import defaultdict
import asyncio

class RateLimiter:
    """Prosty rate limiter oparty na sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """Sprawdza czy klient może wykonać żądanie."""
        now = time.time()
        window_start = now - self.window
        
        # Usuń stare żądania
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]
        
        current_count = len(self._requests[client_id])
        
        if current_count >= self.max_requests:
            return False, {
                "error": "Rate limit exceeded",
                "limit": self.max_requests,
                "window_seconds": self.window,
                "retry_after": int(self._requests[client_id][0] + self.window - now)
            }
        
        self._requests[client_id].append(now)
        return True, {
            "remaining": self.max_requests - current_count - 1,
            "reset_at": int(window_start + self.window)
        }

app = FastAPI()
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware rate limitingu."""
    # Identyfikacja klienta (IP lub API key)
    client_id = request.headers.get("X-API-Key") or request.client.host
    
    allowed, info = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=info,
            headers={"Retry-After": str(info.get("retry_after", 60))}
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
    return response
```

> ⚠️ **Poprawka:** w middleware FastAPI/Starlette `raise HTTPException` **nie** jest obsługiwany przez handler wyjątków (middleware działa poza routerem) – skończy się błędem 500. W middleware zwracaj odpowiedź bezpośrednio:
>
> ```python
> from fastapi.responses import JSONResponse
> if not allowed:
>     return JSONResponse(status_code=429, content=info,
>                         headers={"Retry-After": str(info.get("retry_after", 60))})
> ```
>
> Ponadto limiter w pamięci procesu nie działa poprawnie przy wielu replikach (każda liczy osobno) – w produkcji stan trzymamy w Redis lub delegujemy limitowanie do API Gateway (Envoy, Kong, Cloud Armor).

| Algorytm | Zasada | Zalety | Wady |
|---|---|---|---|
| Fixed window | Licznik na okno (np. minutę) | Prosty, tani | Podwójny burst na granicy okien |
| Sliding window log | Lista znaczników czasu (jak wyżej) | Dokładny | Pamięć O(liczba żądań) |
| Token bucket | Żetony dokładane w stałym tempie | Dopuszcza kontrolowane bursty | Trudniejszy w rozproszeniu |
| Leaky bucket | Kolejka o stałym tempie wypływu | Wygładza ruch | Dodaje opóźnienie |

### 4.4 Differential Privacy

```python
import numpy as np

class DifferentialPrivacyMechanism:
    """
    Implementacja mechanizmu Laplace'a dla differential privacy.
    
    Differential Privacy gwarantuje, że wyniki zapytań do modelu
    nie ujawniają informacji o konkretnych rekordach treningowych.
    """
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """
        Args:
            epsilon: parametr prywatności (mniejszy = więcej prywatności)
            sensitivity: czułość funkcji (max zmiana przy usunięciu 1 rekordu)
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def add_noise(self, value: float) -> float:
        """Dodaje szum Laplace'a do wartości."""
        noise = np.random.laplace(0, self.scale)
        return value + noise
    
    def privatize_prediction(self, probability: float) -> float:
        """Dodaje szum do predykcji modelu."""
        noisy_prob = self.add_noise(probability)
        return float(np.clip(noisy_prob, 0, 1))
    
    def privatize_statistics(self, values: np.ndarray) -> dict:
        """Oblicza prywatne statystyki."""
        return {
            "mean": self.add_noise(float(values.mean())),
            "count": int(self.add_noise(len(values))),
            "std": max(0, self.add_noise(float(values.std())))
        }

# Przykład
dp = DifferentialPrivacyMechanism(epsilon=0.5)

true_probability = 0.85
private_probability = dp.privatize_prediction(true_probability)
print(f"Prawdziwe prawdopodobieństwo: {true_probability:.4f}")
print(f"Prywatne prawdopodobieństwo: {private_probability:.4f}")
print(f"Szum: {abs(true_probability - private_probability):.4f}")
```

**Definicja.** Mechanizm `M` jest `ε`-różnicowo prywatny, jeśli dla dowolnych dwóch zbiorów danych `D`, `D'` różniących się jednym rekordem i każdego wyniku `S`: `P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S]`. Mechanizm Laplace'a osiąga to, dodając szum `Lap(Δf/ε)`, gdzie `Δf` to czułość funkcji.

| ε | Prywatność | Użyteczność | Przykład zastosowania |
|---|---|---|---|
| 0.1 | Bardzo silna | Niska (duży szum) | Statystyki o bardzo wrażliwych danych medycznych |
| 1 | Silna | Średnia | Publikowane agregaty, analityka |
| 8–10 | Umiarkowana | Wysoka | DP-SGD w treningu sieci (np. Apple, Google Gboard) |

**Pułapki:** (1) budżet prywatności **sumuje się** – każde kolejne zapytanie zużywa ε, więc trzeba go księgować; (2) w kodzie powyżej `privatize_statistics` wydaje 3·ε (trzy zapytania); (3) `count` ma czułość 1, ale `mean` zależy od zakresu danych – czułość trzeba wyznaczyć po przycięciu (clipping) wartości. Do treningu modeli z DP stosuje się **DP-SGD** (przycinanie gradientów per przykład + szum), np. biblioteki Opacus (PyTorch) i TensorFlow Privacy.

### 4.5 Ochrona danych osobowych (PII) w pipeline ML

RODO (GDPR) wymaga minimalizacji danych, ograniczenia celu i prawa do usunięcia. W systemach ML dane osobowe „przeciekają” w nieoczywistych miejscach: logi żądań/odpowiedzi, feature store, zbiory treningowe w DVC, artefakty MLflow, dashboardy monitoringu.

| Technika | Odwracalna? | Zachowuje join? | Status wg RODO | Kiedy stosować |
|---|---|---|---|---|
| Usunięcie kolumny | Nie | Nie | Brak danych osobowych | Cecha niepotrzebna modelowi |
| Maskowanie (`<EMAIL>`) | Nie | Nie | Anonimizacja tekstu | Logi, teksty wolne |
| Pseudonimizacja (HMAC z kluczem) | Tylko z kluczem/tablicą | Tak | Nadal dane osobowe | Klucze łączenia tabel, feature store |
| Tokenizacja (vault) | Tak (przez vault) | Tak | Dane osobowe | Płatności, numery kart |
| Generalizacja (wiek → przedział) | Nie | Częściowo | Może prowadzić do anonimizacji (k-anonimowość) | Udostępnianie zbiorów, analityka |
| Differential privacy | Nie | – | Silna gwarancja matematyczna | Agregaty, trening na wrażliwych danych |

```mermaid
graph LR
    SRC[Źródło\nCRM, transakcje] --> ING[Ingestion]
    ING --> PII{Detektor PII}
    PII -->|identyfikatory| HMAC[Pseudonimizacja\nHMAC + klucz w KMS]
    PII -->|quasi-identyfikatory| GEN[Generalizacja]
    PII -->|tekst wolny| MASK[Maskowanie regex/NER]
    HMAC --> FS[Feature Store /\nzbiór treningowy]
    GEN --> FS
    MASK --> FS
    FS --> TRAIN[Trening]
    API[Serving API] --> LOGM[Maskowanie\nprzed logowaniem] --> LOGS[Logi]
```

> **▶️ Demo na żywo:** `python lectures/scripts/lecture_08/03_pii_masking.py`  
> Skrypt maskuje e-mail, telefon, PESEL i IBAN w logach, pseudonimizuje identyfikatory HMAC-SHA256 i pokazuje, że generalizacja wieku i kodu pocztowego podnosi k-anonimowość z k=1 (każdy rekord unikalny) do k≥10. Wynik trafia do `lectures/scripts/lecture_08/outputs/`.

**Pułapki:** zwykły hash (SHA-256 bez klucza) z e-maila czy PESEL-u jest łatwy do odwrócenia atakiem słownikowym – przestrzeń PESEL-i to ~10¹⁰ wartości; regexy nie złapią wszystkiego (imiona, adresy w tekście) – dla tekstu stosuje się modele NER (np. Microsoft Presidio); model sam może „zapamiętać” PII (szczególnie LLM) – stąd DP i deduplikacja danych treningowych.

### 4.6 IAM, sekrety, szyfrowanie i audyt

- **Least privilege:** osobne konta serwisowe dla pipeline'u treningowego (odczyt danych, zapis do registry) i serwisu inferencji (tylko odczyt modelu). Żadnych kluczy JSON w repozytorium – Workload Identity / IRSA.
- **Sekrety:** Secret Manager / Vault / Kubernetes Secrets (z szyfrowaniem etcd), rotacja kluczy, skanowanie repo (gitleaks, pre-commit).
- **Szyfrowanie:** at rest (CMEK/KMS dla bucketów z danymi i modelami), in transit (TLS, mTLS w service mesh).
- **Integralność artefaktów:** podpisuj obrazy i modele (Sigstore/cosign), zapisuj hash modelu w registry, preferuj `safetensors`/ONNX zamiast `pickle`/`joblib` dla modeli z niezaufanych źródeł – deserializacja pickle może wykonać dowolny kod.
- **Audyt:** każda predykcja w systemach wysokiego ryzyka powinna mieć `request_id`, wersję modelu, wersję cech i znacznik czasu (wymóg AI Act dla systemów high-risk – rejestrowanie zdarzeń).

```yaml
# Kubernetes: minimalne uprawnienia kontenera inferencji
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

---

## 5. Optymalizacja kosztów w chmurze

### 5.1 Strategie oszczędzania

```mermaid
graph TB
    COST[Optymalizacja\nKosztów ML] --> COMP[Compute\nOptimization]
    COST --> STOR[Storage\nOptimization]
    COST --> NET[Network\nOptimization]

    COMP --> C1[Spot/Preemptible VMs\ndo 80% taniej]
    COMP --> C2[Right-sizing\ndobór rozmiaru instancji]
    COMP --> C3[Auto-scaling\nzero gdy brak ruchu]
    COMP --> C4[Committed Use\nrezerwacje 1-3 lata]

    STOR --> S1[Lifecycle Policies\narchiwizacja starych danych]
    STOR --> S2[Compression\nParquet zamiast CSV]
    STOR --> S3[Deduplication\nunikanie duplikatów]

    NET --> N1[Regional Endpoints\nunikanie cross-region]
    NET --> N2[CDN\ndla statycznych artefaktów]
```

### Modele zakupu mocy obliczeniowej

| Model | Rabat vs on-demand | Ryzyko | Pasuje do |
|---|---|---|---|
| On-demand | 0% | Brak | Eksperymenty, nieprzewidywalny ruch |
| Spot / Preemptible | 60–90% | Wywłaszczenie z ~30 s ostrzeżeniem | Trening z checkpointami, batch scoring, HPO |
| Committed use / Reserved / Savings Plans | 30–70% | Płacisz nawet bez użycia | Stały ruch serwujący (baseline) |
| Serverless (Cloud Run, Lambda) | – | Cold start, limity | Rzadki, nieregularny ruch, modele CPU |

**Wzorzec mieszany:** baseline serwowania na instancjach z rezerwacją, szczyty na on-demand (HPA), trening na spot z checkpointem co N kroków. Trening na spot **musi** być wznawialny – zapisuj stan modelu, optymalizatora i pozycję w danych do GCS/S3.

### Unit economics – koszt na 1000 predykcji

Najbardziej użyteczna metryka FinOps dla ML to **koszt na jednostkę biznesową** (1000 predykcji, jeden klient, jedna transakcja). Przykład: węzeł za $0.10/h obsługuje 100 RPS przy 50% wykorzystania → `0.10 / (100 × 3600) × 1000 ≈ $0.00028` na 1000 predykcji. Jeśli ruch spada do 1 RPS, koszt jednostkowy rośnie 100× – dlatego scale-to-zero i współdzielenie węzłów przez wiele modeli (multi-model serving) mają tak duże znaczenie.

### 5.2 Monitorowanie kosztów

```python
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class MLCostTracker:
    """Śledzenie kosztów operacji ML."""
    
    # Ceny GCP (USD) - przykładowe
    TRAINING_GPU_HOUR = 2.48      # n1-standard-8 + T4 GPU
    TRAINING_CPU_HOUR = 0.38      # n1-standard-8
    ENDPOINT_HOUR = 0.10          # per node
    STORAGE_GB_MONTH = 0.02       # Cloud Storage Standard
    BIGQUERY_TB_QUERY = 5.00      # BigQuery on-demand
    
    def estimate_training_cost(
        self,
        duration_hours: float,
        use_gpu: bool = True,
        n_nodes: int = 1
    ) -> dict:
        """Szacuje koszt treningu."""
        rate = self.TRAINING_GPU_HOUR if use_gpu else self.TRAINING_CPU_HOUR
        cost = rate * duration_hours * n_nodes
        
        return {
            "duration_hours": duration_hours,
            "n_nodes": n_nodes,
            "use_gpu": use_gpu,
            "cost_usd": round(cost, 2),
            "cost_pln": round(cost * 4.0, 2)  # przybliżony kurs
        }
    
    def estimate_serving_cost(
        self,
        requests_per_day: int,
        avg_latency_ms: float = 50,
        n_replicas: int = 2
    ) -> dict:
        """Szacuje miesięczny koszt serwowania."""
        # Koszt endpointu (per godzinę)
        endpoint_monthly = self.ENDPOINT_HOUR * 24 * 30 * n_replicas
        
        # Szacowany koszt na żądanie
        cost_per_request = endpoint_monthly / (requests_per_day * 30)
        
        return {
            "requests_per_day": requests_per_day,
            "n_replicas": n_replicas,
            "monthly_cost_usd": round(endpoint_monthly, 2),
            "cost_per_1k_requests_usd": round(cost_per_request * 1000, 4)
        }
    
    def recommend_optimizations(self, monthly_cost_usd: float) -> list[str]:
        """Rekomenduje optymalizacje kosztów."""
        recommendations = []
        
        if monthly_cost_usd > 500:
            recommendations.append(
                "💡 Rozważ Committed Use Discounts (1-3 lata) - oszczędność do 57%"
            )
        
        recommendations.extend([
            "💡 Użyj Spot/Preemptible VMs do treningu - oszczędność do 80%",
            "💡 Skonfiguruj auto-scaling z minReplicas=0 dla środowisk dev/staging",
            "💡 Archiwizuj dane starsze niż 90 dni do Coldline Storage",
            "💡 Używaj Parquet zamiast CSV - redukcja rozmiaru o 60-80%",
            "💡 Włącz caching pipeline'ów KFP - unikaj ponownego treningu",
            "💡 Monitoruj i usuwaj nieużywane Endpointy"
        ])
        
        return recommendations

# Przykład użycia
tracker = MLCostTracker()

# Koszt treningu
training = tracker.estimate_training_cost(
    duration_hours=4,
    use_gpu=True,
    n_nodes=2
)
print(f"Koszt treningu: ${training['cost_usd']} ({training['cost_pln']} PLN)")

# Koszt serwowania
serving = tracker.estimate_serving_cost(
    requests_per_day=100_000,
    n_replicas=3
)
print(f"Miesięczny koszt serwowania: ${serving['monthly_cost_usd']}")
print(f"Koszt na 1000 żądań: ${serving['cost_per_1k_requests_usd']}")

# Rekomendacje
recs = tracker.recommend_optimizations(serving['monthly_cost_usd'])
print("\nRekomendacje optymalizacji:")
for rec in recs:
    print(f"  {rec}")
```

---

## 6. Distributed Training

### Horovod – rozproszony trening

```python
import horovod.torch as hvd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def train_distributed(model, X_train, y_train, epochs=10, lr=0.001):
    """
    Trenuje model w trybie rozproszonym z Horovod.
    
    Uruchomienie:
        horovodrun -np 4 python train_distributed.py
    """
    # Inicjalizacja Horovod
    hvd.init()
    
    # Przypisanie GPU do procesu
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        device = torch.device(f'cuda:{hvd.local_rank()}')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Skalowanie learning rate proporcjonalnie do liczby workerów
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * hvd.size())
    
    # Distributed optimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters()
    )
    
    # Broadcast parametrów z rank 0 do wszystkich workerów
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    # Distributed sampler - każdy worker dostaje inną część danych
    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank()
    )
    loader = DataLoader(dataset, batch_size=256, sampler=sampler)
    
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        
        # Loguj tylko z rank 0
        if hvd.rank() == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model
```

---

## 7. Architektura referencyjna MLOps

```mermaid
graph TB
    subgraph "Data Layer"
        GCS[Cloud Storage\nData Lake]
        BQ[BigQuery\nData Warehouse]
        FS[Vertex AI\nFeature Store]
    end

    subgraph "ML Platform"
        EXP[Vertex AI\nExperiments]
        PIPE[Vertex AI\nPipelines KFP]
        REG[Vertex AI\nModel Registry]
    end

    subgraph "Serving Layer"
        EP[Vertex AI\nEndpoints]
        BATCH[Batch\nPrediction Jobs]
    end

    subgraph "Observability"
        MON[Vertex AI\nModel Monitoring]
        PROM[Prometheus\n+ Grafana]
        LOG[Cloud Logging\n+ Alerting]
    end

    subgraph "CI/CD"
        GH[GitHub\nRepository]
        GA[GitHub Actions\nCI/CD Pipeline]
        CB[Cloud Build\nContainer Registry]
    end

    GCS --> BQ --> FS --> PIPE
    PIPE --> EXP
    PIPE --> REG
    REG --> EP
    REG --> BATCH
    EP --> MON
    EP --> PROM
    EP --> LOG
    MON -->|drift alert| PIPE
    GH --> GA --> CB --> EP
```

---

## 8. Typowe pułapki w skalowalności i bezpieczeństwie ML

> ⚠️ **Pułapka 1: Przedwczesna optymalizacja**
> Zespół spędza tygodnie na quantization i distillation modelu, który obsługuje 10 żądań na minutę. Najpierw zmierz, potem optymalizuj. Profiling przed optymalizacją!

> ⚠️ **Pułapka 2: Brak load testów przed wdrożeniem**
> Model działa świetnie na laptopie, ale pada przy 100 RPS na produkcji. Zawsze przeprowadzaj load testy (Locust) przed wdrożeniem.

> ⚠️ **Pułapka 3: Ignorowanie kosztów GPU**
> Zostawienie uruchomionych instancji GPU na noc/weekend. Jedna instancja z GPU T4 kosztuje ~$2.50/h = ~$1800/miesiąc. Używaj auto-scaling z minReplicas=0 dla środowisk dev.

> ⚠️ **Pułapka 4: Brak ochrony przed atakami adversarialnymi**
> Model jest dostępny publicznie bez rate limitingu i walidacji wejść. Atakujący może wyekstrahować model (model extraction) lub znaleźć słabości (adversarial examples).

> ⚠️ **Pułapka 5: Skalowanie na złej metryce**
> HPA na CPU dla modelu na GPU – CPU jest niskie, GPU nasycone, autoscaler nic nie robi. Skaluj na metryce, która jest wąskim gardłem (wykorzystanie GPU, długość kolejki, współbieżność, latencja p95).

> ⚠️ **Pułapka 6: PII w logach i artefaktach**
> Logowanie pełnych payloadów żądań „do debugowania” – po roku logi z e-mailami i PESEL-ami trafiają do wielu systemów. Maskuj przed zapisem i ustaw retencję logów.

> ⚠️ **Pułapka 7: Ładowanie niezaufanego pickle**
> `joblib.load("model_z_internetu.pkl")` może wykonać dowolny kod. Ładuj modele tylko z własnego registry z weryfikacją hash/podpisu.

> ⚠️ **Pułapka 8: Cache bez wersji modelu**
> Po wdrożeniu nowego modelu użytkownicy przez godzinę dostają predykcje starego, bo klucz cache nie zawierał wersji.

### Case Study: Uber Michelangelo — skalowanie platformy ML

**Uber** obsługuje tysiące modeli (ETA, ceny, fraud) w jednej platformie:
- Modele online serwowane z latencją rzędu milisekund, z cechami z **online feature store** (Cassandra) i precomputed features liczonymi w Sparku/Flinku.
- **Multi-model serving** – wiele modeli w jednym kontenerze, aby nie płacić za tysiące osobnych deploymentów.
- Wnioski: większość latencji to pobieranie cech, nie sam model – optymalizacja zaczyna się od danych.

### Case Study: ataki na systemy ML w praktyce

- **Microsoft Tay (2016)** – chatbot uczony online na wypowiedziach użytkowników został w ciągu doby „zatruty” obraźliwymi treściami (data poisoning przez interakcje). Lekcja: nie uczyć się online z niezaufanego wejścia bez filtrowania i moderacji.
- **Złośliwe modele na publicznych hubach** – badacze bezpieczeństwa wielokrotnie znajdowali modele w formacie pickle wykonujące kod przy ładowaniu. Lekcja: format `safetensors` i skanowanie artefaktów.
- **Ekstrakcja modeli z publicznych API** (Tramèr i in., 2016) – kopie modeli komercyjnych ML-as-a-Service odtwarzano z kilku tysięcy zapytań, szczególnie gdy API zwracało pełne prawdopodobieństwa.

### Case Study: OpenAI — skalowalność GPT

**OpenAI** skaluje modele GPT do milionów użytkowników:
- Używają **model parallelism** — jeden model jest rozdzielony na wiele GPU (tensor parallelism, pipeline parallelism).
- **KV-cache** — cache dla attention mechanism, który drastycznie redukuje koszt generowania kolejnych tokenów.
- **Quantization** — modele są kwantyzowane do INT8/INT4 dla inference, co zmniejsza wymagania pamięciowe 2-4x.
- Koszt treningu GPT-4 szacowany na **$100M+** — optymalizacja kosztów jest krytyczna.

### Case Study: Revolut — bezpieczeństwo ML w fintech

**Revolut** używa ML do wykrywania fraudów i musi spełniać rygorystyczne wymagania regulacyjne:
- **Differential privacy** — model nie może ujawniać informacji o konkretnych transakcjach klientów.
- **Explainability** — regulacje (GDPR, PSD2) wymagają wyjaśnialności decyzji modelu. Używają SHAP i LIME.
- **Adversarial robustness** — model musi być odporny na próby obejścia przez fraudsterów (np. drobne modyfikacje kwot transakcji).
- **Audit trail** — pełna historia decyzji modelu jest przechowywana przez 7 lat.

---

## Pytania kontrolne i do dyskusji

1. Wyjaśnij różnicę między skalowaniem horyzontalnym a wertykalnym. Kiedy użyjesz każdego?
2. Jak działa quantization i jakie są trade-offy (rozmiar vs dokładność)?
3. Co to jest knowledge distillation i kiedy warto ją zastosować?
4. Wymień trzy zagrożenia bezpieczeństwa specyficzne dla systemów ML.
5. Jak działa differential privacy i kiedy jest wymagana?
6. Jakie strategie optymalizacji kosztów zastosujesz dla serwisu ML w chmurze?
7. **Dyskusja:** Czy model ML powinien być traktowany jako „czarna skrzynka" w kontekście regulacji (GDPR, AI Act)? Jakie są implikacje?
8. Na czym polega trade-off latencji i przepustowości w dynamic batchingu? Jak dobrać `max_wait`?
9. Dlaczego pseudonimizacja HMAC jest lepsza od zwykłego hasha SHA-256 i czy dane pseudonimizowane przestają być danymi osobowymi?
10. Jakie metryki wybierzesz do autoskalowania modelu serwowanego na GPU i dlaczego nie CPU?


### Proponowane odpowiedzi

1. Skalowanie wertykalne zwiększa zasoby pojedynczej maszyny (więcej CPU/RAM/GPU), a horyzontalne dodaje kolejne repliki/usługi. Wertykalne jest prostsze na start, horyzontalne lepsze dla wysokiej dostępności i dużego ruchu.
2. Quantization obniża precyzję wag/aktywacji (np. FP32→INT8), zmniejszając model i przyspieszając inference. Trade-off: oszczędność zasobów kosztem możliwej utraty części dokładności.
3. Knowledge distillation przenosi wiedzę z dużego „teacher” do mniejszego „studenta”, który uczy się z miękkich predykcji. Warto ją stosować, gdy potrzebna jest niższa latencja/tańsze serwowanie przy zachowaniu sensownej jakości.
4. Przykłady zagrożeń: adversarial examples, model extraction przez masowe zapytania, data poisoning. Dodatkowo ryzykiem jest wyciek danych treningowych (privacy leakage).
5. Differential privacy dodaje kontrolowany szum/statystyczną ochronę, aby pojedyncza obserwacja nie była możliwa do odtworzenia z modelu lub wyników. Jest wymagana szczególnie przy danych wrażliwych i rygorach regulacyjnych (np. finanse, zdrowie).
6. Strategie kosztowe: auto-scaling (w tym minReplicas=0 w dev), right-sizing CPU/GPU, użycie Spot/Preemptible, batchowanie predykcji, cache, harmonogramy wyłączania środowisk i monitoring kosztów per model. Dodatkowo optymalizacja modelu (quantization/distillation) obniża koszt inferencji.
7. Coraz rzadziej można traktować model jako pełną „czarną skrzynkę”, bo regulacje wymagają wyjaśnialności, audytu i zarządzania ryzykiem. Im większy wpływ decyzji na użytkownika, tym większa potrzeba transparentności procesu i wyników.
8. Większy batch zwiększa przepustowość (koszt stały dzielony na więcej predykcji), ale żądania czekają na zebranie batcha. `max_wait` dobieramy tak, by czas oczekiwania plus czas wywołania pełnego batcha mieścił się w budżecie SLO p99, a weryfikujemy load testem.
9. Bez klucza hash identyfikatora o małej przestrzeni (PESEL, e-mail) można odwrócić atakiem słownikowym; HMAC wymaga tajnego klucza (trzymanego w KMS). Dane pseudonimizowane nadal są danymi osobowymi wg RODO, bo z kluczem da się je powiązać z osobą.
10. Wykorzystanie GPU, długość kolejki inferencji, współbieżność żądań lub latencja p95 – to one odzwierciedlają wąskie gardło. CPU w serwisie GPU bywa niskie mimo przeciążenia, więc HPA na CPU nie zareaguje.


---

## Podsumowanie

- **Skalowalność** osiągamy przez Kubernetes HPA, load balancing i distributed training.
- **Optymalizacja modeli**: quantization (4x mniejszy model), distillation (szybszy student), caching (eliminacja powtarzających się obliczeń).
- **Bezpieczeństwo ML**: walidacja wejść, rate limiting, differential privacy, ochrona przed atakami adversarialnymi.
- **Optymalizacja kosztów**: Spot VMs (80% taniej), auto-scaling, lifecycle policies, Parquet zamiast CSV.
- Architektura referencyjna łączy wszystkie komponenty MLOps w spójny system.
- Regulacje (GDPR, AI Act) wymagają explainability, audit trail i ochrony prywatności.
- **Batching** zwiększa przepustowość kosztem latencji – parametry dobieramy pod SLO na podstawie percentyli.
- **Dane osobowe** chronimy warstwowo: minimalizacja, maskowanie logów, pseudonimizacja HMAC, generalizacja (k-anonimowość) i DP.
- **Bezpieczeństwo łańcucha dostaw** (podpisane artefakty, brak niezaufanego pickle) i least privilege są równie ważne jak obrona przed atakami adversarialnymi.

## Literatura i zasoby

- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Horovod Documentation](https://horovod.readthedocs.io/)
- [Differential Privacy – Google's DP Library](https://github.com/google/differential-privacy)
- [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)
- [Locust Load Testing](https://locust.io/)
- [EU AI Act – Summary](https://artificialintelligenceact.eu/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [KEDA – Kubernetes Event-driven Autoscaling](https://keda.sh/)
- [Microsoft Presidio – wykrywanie i anonimizacja PII](https://microsoft.github.io/presidio/)
- [Opacus – DP-SGD dla PyTorch](https://opacus.ai/)
- Hinton, Vinyals, Dean – *Distilling the Knowledge in a Neural Network* (2015)
- Goodfellow, Shlens, Szegedy – *Explaining and Harnessing Adversarial Examples* (2015)
