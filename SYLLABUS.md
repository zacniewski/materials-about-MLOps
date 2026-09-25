# Sylabus przedmiotu: MLOps i Inżynieria Systemów ML

## Informacje podstawowe

- **Nazwa przedmiotu:** MLOps i Inżynieria Systemów ML
- **Forma zajęć:** wykład + laboratorium + projekt
- **Łączny wymiar godzin:** 32h
  - wykład: 16h (8 × 2h)
  - laboratorium: 14h (7 × 2h)
  - projekt: 2h (konsultacje + kickoff) + praca własna

## Cele przedmiotu

Po ukończeniu kursu student:
- rozumie pełny cykl życia systemu ML w środowisku produkcyjnym,
- potrafi zaprojektować i wdrożyć podstawowy pipeline MLOps,
- zna praktyki wersjonowania kodu, danych i modeli,
- potrafi monitorować model w produkcji oraz reagować na drift.

## Plan wykładów

| # | Temat | Zakres |
|---|---|---|
| 1 | Wprowadzenie do MLOps | MLOps vs DevOps, cykl życia ML, poziomy dojrzałości |
| 2 | Inżynieria Danych dla ML | Data Lake, ETL/ELT, Feature Store, DVC |
| 3 | Eksperymentowanie i Tracking | MLflow, metryki, reproducibility |
| 4 | ML Pipelines | DAG, Kubeflow Pipelines, automatyzacja |
| 5 | Model Serving | REST API, FastAPI, Docker, deployment patterns |
| 6 | Monitoring modeli | Data drift, concept drift, alerting |
| 7 | CI/CD dla ML | Testowanie, quality gates, GitHub Actions |
| 8 | Skalowalność i bezpieczeństwo | Koszty, optymalizacja, bezpieczeństwo systemów ML |

## Plan laboratoriów

| # | Temat | Narzędzia |
|---|---|---|
| 1 | Środowisko i Git/DVC | Git, DVC, pytest, pre-commit |
| 2 | MLflow tracking | MLflow, scikit-learn |
| 3 | Walidacja danych | Great Expectations, pandas |
| 4 | REST API dla modelu | FastAPI, Pydantic, Docker |
| 5 | Monitoring i drift | SciPy, Prometheus, Grafana |
| 6 | Pipeline ML | Kubeflow Pipelines |
| 7 | CI/CD | GitHub Actions |
| 8 | Blok projektowy i konsultacje | Przegląd postępu projektów, wsparcie implementacyjne |

## Plan części projektowej

| Etap | Kiedy | Zakres |
|---|---|---|
| 1 | Start równolegle z Lab 1–2 | Wybór problemu ML i karta projektu |
| 2 | Równolegle z Lab 3–5 | Implementacja komponentów (dane, model, API, monitoring) |
| 3 | Równolegle z Lab 6–7 | Integracja end-to-end, CI/CD, dokumentacja |
| 4 | Tydzień końcowy | Prezentacja i oddanie raportu końcowego |

## Forma zaliczenia

- aktywność i wykonanie laboratoriów (Lab 1–7),
- projekt końcowy realizowany równolegle z laboratoriami (z konsultacjami w bloku projektowym),
- raport końcowy z decyzji projektowych i wyników.

## Wymagania wstępne

- podstawy Python,
- podstawy ML (trening/ewaluacja modeli),
- znajomość podstaw Git i pracy z terminalem Linux.

## Materiały

- `README.md` — mapa kursu i szybki start,
- katalog `lectures/` — materiały teoretyczne,
- katalog `labs/` — ćwiczenia laboratoryjne,
- katalog `project/` — materiały ścieżki projektowej,
- katalog `reports/` — miejsce na raporty i artefakty projektowe.