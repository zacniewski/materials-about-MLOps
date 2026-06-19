# Wykład 1: Wprowadzenie do MLOps i Inżynierii Systemów ML

## Cel wykładu
Po tym wykładzie student:
- rozumie, czym jest MLOps i dlaczego jest potrzebny,
- zna cykl życia modelu ML w produkcji,
- potrafi opisać różnice między tradycyjnym DevOps a MLOps,
- zna główne wyzwania wdrażania modeli ML.

---

## 1. Czym jest MLOps?

**MLOps** (Machine Learning Operations) to zbiór praktyk, narzędzi i kultury organizacyjnej, które mają na celu ustandaryzowanie i usprawnienie cyklu życia modeli uczenia maszynowego — od eksperymentu do produkcji i z powrotem.

> „MLOps to DevOps zastosowany do uczenia maszynowego."

MLOps łączy trzy światy:

```mermaid
graph TD
    A[Data Engineering] --> D[MLOps]
    B[Machine Learning / Data Science] --> D
    C[Software Engineering / DevOps] --> D
    D --> E[Niezawodne systemy ML w produkcji]
```

### Dlaczego MLOps jest trudny?

W odróżnieniu od klasycznego oprogramowania, systemy ML mają dodatkowy wymiar złożoności:

| Aspekt | Klasyczne oprogramowanie | System ML |
|--------|--------------------------|-----------|
| Kod | Deterministyczny | Probabilistyczny |
| Testowanie | Unit/integration tests | Testy danych + modelu |
| Wersjonowanie | Kod (git) | Kod + dane + model + pipeline |
| Degradacja | Błędy w kodzie | Data drift, concept drift |
| Reprodukowalność | Łatwa | Trudna (losowość, dane, środowisko) |
| Zależności | Biblioteki | Biblioteki + dane + hardware (GPU) |
| Debugowanie | Stack trace | Analiza rozkładów, metryk, cech |

### Dług techniczny w systemach ML

Artykuł Google „Hidden Technical Debt in Machine Learning Systems" (NIPS 2015) pokazuje, że **sam kod modelu to zaledwie ~5% całego systemu ML**. Reszta to infrastruktura:

```mermaid
graph TB
    subgraph "Cały system ML"
        CONFIG[Zarządzanie\nkonfiguracją]
        DATA[Zbieranie\ni weryfikacja danych]
        FE[Feature\nEngineering]
        CODE[Kod ML\n~5% systemu]
        SERVE[Infrastruktura\nserwowania]
        MON[Monitoring]
        PROC[Zarządzanie\nprocesami]
        RES[Zarządzanie\nzasobami]
        AUTO[Automatyzacja]
    end
```

> 💡 **Pułapka:** Wiele zespołów koncentruje się wyłącznie na trenowaniu modelu (Jupyter Notebook), ignorując 95% pracy potrzebnej do wdrożenia i utrzymania modelu w produkcji. To główny powód, dla którego według badań Gartner **85% projektów ML nigdy nie trafia do produkcji**.

### Case Study: Uber Michelangelo

**Uber** zbudował wewnętrzną platformę MLOps o nazwie **Michelangelo**, która obsługuje tysiące modeli ML w produkcji (ETA, dynamiczne ceny, wykrywanie fraudów). Kluczowe wnioski:
- Przed Michelangelo każdy zespół budował własne rozwiązania — brak standaryzacji prowadził do duplikacji pracy i niespójności.
- Platforma ustandaryzowała cały cykl: od zarządzania danymi, przez trening, po serwowanie i monitoring.
- Wprowadzenie centralnego **Feature Store** wyeliminowało problem, w którym różne zespoły obliczały te same cechy na różne sposoby.

### Case Study: Spotify — rekomendacje muzyczne

**Spotify** wykorzystuje MLOps do obsługi systemu rekomendacji dla ponad 500 milionów użytkowników:
- Modele są retrenowane codziennie na nowych danych o odsłuchach.
- Używają **Kubeflow Pipelines** do orkiestracji pipeline'ów ML.
- Monitoring dryfu danych jest kluczowy — preferencje muzyczne zmieniają się sezonowo (np. kolędy w grudniu).

---

## 2. Cykl życia modelu ML

```mermaid
flowchart LR
    A[Definicja\nproblemu] --> B[Zbieranie\ndanych]
    B --> C[Przygotowanie\ndanych]
    C --> D[Trening\nmodelu]
    D --> E[Ewaluacja]
    E -->|Niezadowalający| C
    E -->|OK| F[Wdrożenie]
    F --> G[Monitoring]
    G -->|Data drift| B
    G -->|Model degradacja| D
```

### Fazy cyklu życia

1. **Definicja problemu** – co chcemy przewidzieć? Jaka jest metryka sukcesu?
2. **Zbieranie danych** – skąd dane? Jak często są aktualizowane?
3. **Przygotowanie danych** – czyszczenie, feature engineering, podział na zbiory.
4. **Trening modelu** – wybór algorytmu, hiperparametry, eksperymenty.
5. **Ewaluacja** – metryki ML (AUC, RMSE, F1) i metryki biznesowe.
6. **Wdrożenie** – serwowanie predykcji (online/batch).
7. **Monitoring** – czy model nadal działa poprawnie?

---

## 3. Poziomy dojrzałości MLOps

Google definiuje trzy poziomy dojrzałości MLOps:

```mermaid
graph LR
    L0["MLOps Level 0\nRęczny proces\n(Jupyter Notebooks)"]
    L1["MLOps Level 1\nAutomatyzacja treningu\n(ML Pipelines)"]
    L2["MLOps Level 2\nAutomatyzacja CI/CD\n(Pełna automatyzacja)"]
    L0 --> L1 --> L2
```

### Level 0 – Ręczny proces
- Data Scientist trenuje model lokalnie w notebooku.
- Model jest eksportowany i wdrażany ręcznie (np. przez email lub pendrive).
- Brak wersjonowania danych i modeli.
- Brak monitoringu — nikt nie wie, czy model nadal działa poprawnie.
- Brak automatycznych testów — „działa na moim laptopie".

**Problem:** „Works on my machine" – model działa u Data Scientista, ale nie w produkcji.

> 💡 **Typowa pułapka Level 0:** Data Scientist wysyła plik `model_final_v3_FINAL.pkl` na Slacku. Nikt nie wie, jakie dane i parametry zostały użyte. Po 3 miesiącach model przestaje działać, a odtworzenie go jest niemożliwe.

**Przykład z życia:** Startup fintech trenuje model scoringowy w Jupyter Notebook. Model działa świetnie na danych testowych (AUC=0.92). Po wdrożeniu na produkcję okazuje się, że preprocessing danych różni się od tego w notebooku — AUC spada do 0.65.

### Level 1 – Automatyzacja treningu
- Zdefiniowane ML Pipelines (automatyczny trening od danych do modelu).
- Wersjonowanie danych, kodu i modeli (DVC, MLflow).
- Automatyczne wyzwalanie retreningu (np. po wykryciu driftu lub według harmonogramu).
- Monitoring modelu w produkcji (metryki ML + metryki operacyjne).
- Feature Store zapewnia spójność cech między treningiem a produkcją.

**Przykład:** Bank automatycznie retrenuje model scoringowy co tydzień na nowych danych transakcyjnych. Pipeline waliduje dane, trenuje model, porównuje metryki z poprzednią wersją i rejestruje model w Model Registry.

### Level 2 – CI/CD dla ML
- Pełna automatyzacja: od commita do wdrożenia (GitHub Actions, Cloud Build).
- Testy jednostkowe dla komponentów pipeline'u.
- Automatyczne testy A/B nowych modeli na produkcji.
- Canary deployments (stopniowe wdrożenie), rollback w razie problemów.
- Quality Gates — automatyczne bramki jakości blokujące wdrożenie słabych modeli.

**Przykład:** Google Ads używa Level 2 MLOps — każda zmiana w kodzie modelu automatycznie uruchamia pipeline CI/CD, który trenuje model, testuje go, wdraża canary na 5% ruchu i monitoruje metryki biznesowe.

### Jak ocenić swój poziom dojrzałości?

| Pytanie | Level 0 | Level 1 | Level 2 |
|---------|---------|---------|---------|
| Jak często retrenujecie model? | Ręcznie, ad hoc | Automatycznie, regularnie | Automatycznie, na trigger |
| Czy możecie odtworzyć model sprzed 3 miesięcy? | Nie | Tak (DVC/MLflow) | Tak + pełny audit trail |
| Ile czasu zajmuje wdrożenie nowego modelu? | Dni–tygodnie | Godziny | Minuty |
| Kto monitoruje model w produkcji? | Nikt | Zespół ML | Automatyczny system |

---

## 4. Główne wyzwania MLOps

### 4.1 Reprodukowalność

```python
# Złe podejście - brak kontroli losowości
import numpy as np
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Dobre podejście - pełna reprodukowalność
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SEED = 42
np.random.seed(SEED)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Zapisz parametry eksperymentu
experiment_config = {
    "seed": SEED,
    "model": "RandomForestClassifier",
    "n_estimators": 100,
    "data_version": "v2024-01-15",
    "train_size": len(X_train)
}
```

### 4.2 Data Drift

**Data drift** – rozkład danych wejściowych zmienia się w czasie.

```python
import numpy as np
from scipy import stats

def detect_data_drift(reference_data: np.ndarray, 
                       current_data: np.ndarray,
                       threshold: float = 0.05) -> dict:
    """
    Wykrywa data drift za pomocą testu Kolmogorova-Smirnova.
    
    Args:
        reference_data: dane referencyjne (z treningu)
        current_data: aktualne dane produkcyjne
        threshold: próg p-value dla odrzucenia H0
    
    Returns:
        dict z wynikami testu
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < threshold,
        "severity": "high" if p_value < 0.01 else "medium" if p_value < threshold else "none"
    }

# Przykład użycia
reference = np.random.normal(0, 1, 1000)   # dane treningowe
current = np.random.normal(0.5, 1.2, 500)  # dane produkcyjne (z dryftem)

result = detect_data_drift(reference, current)
print(f"Drift wykryty: {result['drift_detected']}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Nasilenie: {result['severity']}")
```

### 4.3 Concept Drift

**Concept drift** – relacja między cechami a etykietą zmienia się w czasie.

Przykład: model przewidujący ceny mieszkań przestaje działać po kryzysie finansowym.

```mermaid
graph LR
    subgraph "Przed dryftem"
        A1[Cechy X] -->|stara relacja| B1[Etykieta Y]
    end
    subgraph "Po dryfcie"
        A2[Cechy X] -->|nowa relacja| B2[Etykieta Y']
    end
```

---

## 5. Architektura systemu MLOps

```mermaid
graph TB
    subgraph "Data Layer"
        DS[Data Sources] --> DL[Data Lake]
        DL --> DW[Data Warehouse]
        DW --> FS[Feature Store]
    end
    
    subgraph "ML Platform"
        FS --> EXP[Experiment\nTracking]
        EXP --> TR[Model Training]
        TR --> REG[Model Registry]
    end
    
    subgraph "Serving Layer"
        REG --> ONL[Online Serving\nREST API]
        REG --> BAT[Batch Serving]
    end
    
    subgraph "Observability"
        ONL --> MON[Monitoring &\nAlerting]
        BAT --> MON
        MON -->|Retrain trigger| TR
    end
```

---

## 6. Kluczowe narzędzia ekosystemu MLOps

| Kategoria | Narzędzia Open Source | Narzędzia Cloud |
|-----------|----------------------|-----------------|
| Experiment Tracking | MLflow, Weights & Biases | Vertex AI Experiments |
| Pipeline Orchestration | Kubeflow, Apache Airflow | Vertex AI Pipelines |
| Model Registry | MLflow Registry | Vertex AI Model Registry |
| Feature Store | Feast | Vertex AI Feature Store |
| Model Serving | BentoML, Seldon | Vertex AI Endpoints |
| Monitoring | Evidently AI, Prometheus | Vertex AI Model Monitoring |
| CI/CD | GitHub Actions, Jenkins | Cloud Build |

---

## 7. MLOps vs DevOps – kluczowe różnice

```mermaid
graph TD
    subgraph DevOps
        D1[Code] --> D2[Build] --> D3[Test] --> D4[Deploy] --> D5[Monitor]
        D5 -->|feedback| D1
    end
    
    subgraph MLOps
        M1[Code + Data] --> M2[Train] --> M3[Evaluate] --> M4[Deploy] --> M5[Monitor]
        M5 -->|data drift| M1
        M5 -->|model decay| M2
        M3 -->|poor metrics| M1
    end
```

**Dodatkowe elementy w MLOps:**
- Wersjonowanie **danych** (nie tylko kodu)
- Wersjonowanie **modeli** (artefakty binarne)
- **Metryki ML** jako warunek wdrożenia
- **Automatyczny retraining** po wykryciu driftu
- **Explainability** – wyjaśnialność predykcji

---

## 8. Przykład: Prosty system MLOps od zera

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
import hashlib

@dataclass
class ModelVersion:
    """Reprezentuje wersję modelu w rejestrze."""
    name: str
    version: str
    metrics: dict
    data_version: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "staging"  # staging | production | archived
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "metrics": self.metrics,
            "data_version": self.data_version,
            "created_at": self.created_at,
            "status": self.status
        }

class SimpleModelRegistry:
    """Prosty rejestr modeli (demonstracja konceptu)."""
    
    def __init__(self):
        self.models: dict[str, list[ModelVersion]] = {}
    
    def register(self, model: ModelVersion) -> str:
        """Rejestruje nową wersję modelu."""
        if model.name not in self.models:
            self.models[model.name] = []
        self.models[model.name].append(model)
        print(f"✅ Zarejestrowano: {model.name} v{model.version}")
        return model.version
    
    def promote_to_production(self, name: str, version: str):
        """Promuje wersję modelu do produkcji."""
        for mv in self.models.get(name, []):
            if mv.status == "production":
                mv.status = "archived"
            if mv.version == version:
                mv.status = "production"
                print(f"🚀 Wdrożono: {name} v{version} → produkcja")
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Zwraca aktualny model produkcyjny."""
        for mv in self.models.get(name, []):
            if mv.status == "production":
                return mv
        return None

# Użycie
registry = SimpleModelRegistry()

# Rejestracja modelu v1
v1 = ModelVersion(
    name="churn_predictor",
    version="1.0.0",
    metrics={"auc": 0.82, "f1": 0.75},
    data_version="data-2024-01"
)
registry.register(v1)
registry.promote_to_production("churn_predictor", "1.0.0")

# Rejestracja lepszego modelu v2
v2 = ModelVersion(
    name="churn_predictor",
    version="2.0.0",
    metrics={"auc": 0.89, "f1": 0.83},
    data_version="data-2024-06"
)
registry.register(v2)
registry.promote_to_production("churn_predictor", "2.0.0")

# Sprawdzenie aktualnego modelu produkcyjnego
prod = registry.get_production_model("churn_predictor")
print(f"\nAktualny model produkcyjny: {prod.name} v{prod.version}")
print(f"Metryki: {prod.metrics}")
```

---

## 9. Role w zespole MLOps

Wdrożenie MLOps wymaga współpracy wielu specjalistów:

| Rola | Odpowiedzialność | Typowe narzędzia |
|------|------------------|------------------|
| **Data Scientist** | Eksperymentowanie, trening modeli | Jupyter, scikit-learn, PyTorch |
| **ML Engineer** | Produktyzacja modeli, pipeline'y | KFP, Airflow, Docker |
| **Data Engineer** | Pipeline'y danych, Feature Store | Spark, Kafka, dbt |
| **DevOps/Platform Engineer** | Infrastruktura, CI/CD, monitoring | Kubernetes, Terraform, Prometheus |
| **Product Manager** | Definiowanie metryk biznesowych | A/B testing, dashboardy |

> 💡 **Wskazówka:** W małych zespołach jedna osoba często pełni kilka ról. Kluczowe jest, aby **każda z tych odpowiedzialności była pokryta**, nawet jeśli przez tę samą osobę.

---

## Pytania kontrolne i do dyskusji

1. Wyjaśnij, dlaczego wdrożenie modelu ML w produkcji jest trudniejsze niż wdrożenie klasycznej aplikacji webowej.
2. Na jakim poziomie dojrzałości MLOps (0, 1, 2) znajduje się Twoja organizacja lub projekt? Uzasadnij.
3. Podaj trzy przykłady sytuacji, w których model ML może zdegradować się w produkcji.
4. Czym różni się data drift od concept drift? Podaj przykład każdego z nich.
5. Dlaczego sam kod modelu stanowi tylko ~5% całego systemu ML? Co stanowi pozostałe 95%?
6. Jakie korzyści daje Feature Store w porównaniu z sytuacją, gdy każdy Data Scientist oblicza cechy samodzielnie?
7. **Dyskusja:** Czy każdy projekt ML potrzebuje pełnego MLOps (Level 2)? Kiedy Level 0 jest wystarczający?

---

## Podsumowanie

- **MLOps** to praktyki łączące ML, inżynierię oprogramowania i operacje IT.
- Główne wyzwania: reprodukowalność, data drift, concept drift, skalowalność.
- Trzy poziomy dojrzałości: ręczny → automatyzacja treningu → pełne CI/CD.
- Kluczowe komponenty: Feature Store, Experiment Tracking, Model Registry, Monitoring.
- Sam kod modelu to ~5% systemu — reszta to infrastruktura, dane i procesy.
- MLOps to nie jednorazowy projekt, ale **ciągły proces** wymagający współpracy wielu ról.

## Literatura i zasoby

- [Google MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Sculley et al. "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Chip Huyen "Designing Machine Learning Systems" (O'Reilly, 2022)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [ml-ops.org](https://ml-ops.org/)
- [Uber Michelangelo – ML Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Made With ML – MLOps Course](https://madewithml.com/)
- [Awesome MLOps – lista narzędzi i zasobów](https://github.com/visenger/awesome-mlops)
