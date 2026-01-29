# Harmonogram przedmiotu "MLOps i inżynieria systemów"

## Rozkład godzinowy
* **Laboratoria:** 15 spotkań po 2 godziny (łącznie 30h)
* **Projekt:** 15 spotkań po 2 godziny (łącznie 30h) - praca nadzorowana i konsultacje

## Program Laboratoriów (30h)
*Laboratoria bazują na Google Cloud Platform (Vertex AI) w ramach darmowego limitu oraz narzędziach Open Source.*

### Część I: Fundamenty i Środowisko (6h)
1. **Lab 1:** Wprowadzenie do Google Cloud Platform i Vertex AI. Konfiguracja projektu i Vertex AI Workbench (Managed Notebooks).
2. **Lab 2:** Wersjonowanie kodu i eksperymentów. Integracja Vertex AI z GitHub/GitLab.
3. **Lab 3:** Praca z danymi w chmurze: Cloud Storage jako Data Lake, wprowadzenie do BigQuery.

### Część II: Inżynieria Danych i Cech (6h)
4. **Lab 4:** Vertex AI Feature Store – zarządzanie cechami w skali.
5. **Lab 5:** Wersjonowanie danych i artefaktów za pomocą Vertex AI Metadata.
6. **Lab 6:** Budowanie potoków przetwarzania danych (TFX / Kubeflow na Vertex AI Pipelines).

### Część III: Trening i Eksperymentowanie (6h)
7. **Lab 7:** Vertex AI Training: Trening modeli w kontenerach (Custom Training Jobs).
8. **Lab 8:** Hyperparameter Tuning z wykorzystaniem Vertex AI Vizier.
9. **Lab 9:** Śledzenie eksperymentów: Vertex AI Experiments vs MLflow.

### Część IV: Deployment i Serwowanie (6h)
10. **Lab 10:** Vertex AI Model Registry – zarządzanie cyklem życia modelu.
11. **Lab 11:** Wdrażanie modeli na Vertex AI Endpoints (Online Prediction).
12. **Lab 12:** Batch Prediction – przetwarzanie wsadowe w Vertex AI.

### Część V: Monitoring i Automatyzacja (6h)
13. **Lab 13:** Vertex AI Model Monitoring – wykrywanie dryfu danych i modelu (Skew & Drift detection).
14. **Lab 14:** CI/CD dla ML (Vertex AI Pipelines + GitHub Actions).
15. **Lab 15:** Podsumowanie, bezpieczeństwo (IAM) i optymalizacja kosztów w Google Cloud.

## Program Projektowy (30h)
*Projekt polega na budowie kompletnego systemu MLOps dla wybranego problemu.*

### Milestones (Etapy):
1. **M1 (2h):** Wybór problemu, analiza dostępności danych i zdefiniowanie metryk biznesowych/ML.
2. **M2 (4h):** Setup środowiska GCP, repozytorium i struktury projektu (Cookiecutter Data Science).
3. **M3 (4h):** Inżynieria danych: Ingest danych do Cloud Storage/BigQuery, czyszczenie i EDA.
4. **M4 (4h):** Opracowanie modelu bazowego (Baseline) i ramy treningowej w Vertex AI Training.
5. **M5 (4h):** Implementacja ML Pipeline (Vertex AI Pipelines) łączącego procesy od danych do modelu.
6. **M6 (4h):** Wdrożenie (Deployment) na Endpoint i przygotowanie testów API.
7. **M7 (4h):** Implementacja monitoringu i mechanizmów CI/CD.
8. **M8 (4h):** Finalne testy, dokumentacja techniczna i przygotowanie prezentacji.

## Praktyczna Checklista Studenta (Krok po kroku)

### 1. Przygotowanie (Free Tier Setup)
- [ ] Załóż konto Google Cloud (skorzystaj z darmowych 300$ na start).
- [ ] Utwórz nowy projekt w konsoli GCP.
- [ ] Włącz API: Vertex AI, Cloud Storage, Compute Engine.

### 2. Rozwój (Development)
- [ ] Uruchom Vertex AI Workbench Instance (wybierz najmniejszą instancję, aby oszczędzać środki).
- [ ] Sklonuj repozytorium projektu do Workbench.
- [ ] Przeprowadź EDA w JupyterLab.

### 3. Automatyzacja (Pipelines)
- [ ] Zdefiniuj komponenty Kubeflow Pipelines (KFP).
- [ ] Skompiluj i uruchom potok w Vertex AI Pipelines.
- [ ] Sprawdź artefakty w Vertex AI Metadata.

### 4. Produkcja (Production)
- [ ] Zarejestruj wytrenowany model w Model Registry.
- [ ] Wdróż model na Endpoint (pamiętaj o usunięciu Endpointu po testach!).
- [ ] Wyślij testowe żądanie (Predict request) do API.

### 5. Utrzymanie (Monitoring)
- [ ] Skonfiguruj Model Monitoring Job.
- [ ] Zasymuluj dryf danych i sprawdź powiadomienia.
- [ ] Skonfiguruj GitHub Action do automatycznego uruchamiania potoku.
