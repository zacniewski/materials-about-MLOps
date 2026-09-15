# Sylabus przedmiotu: MLOps i Inżynieria Systemów ML

## Informacje podstawowe

- **Nazwa przedmiotu:** MLOps i Inżynieria Systemów ML
- **Forma zajęć:** wykład + laboratorium
- **Łączny wymiar godzin:** 32h
  - wykład: 16h (8 × 2h)
  - laboratorium: 16h (8 × 2h)

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
| 8 | Projekt końcowy | Integracja pełnego przepływu MLOps |

## Forma zaliczenia

- aktywność i wykonanie laboratoriów,
- projekt końcowy (Lab 8) obejmujący kompletny mini-system MLOps,
- raport końcowy z decyzji projektowych i wyników.

## Wymagania wstępne

- podstawy Python,
- podstawy ML (trening/ewaluacja modeli),
- znajomość podstaw Git i pracy z terminalem Linux.

## Materiały

- `README.md` — mapa kursu i szybki start,
- katalog `lectures/` — materiały teoretyczne,
- katalog `labs/` — ćwiczenia praktyczne,
- katalog `reports/` — miejsce na raporty i artefakty projektowe.