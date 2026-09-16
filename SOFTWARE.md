# Oprogramowanie wymagane do laboratoriów

Poniżej znajduje się lista narzędzi, które należy mieć zainstalowane, aby uruchamiać laboratoria z tego repozytorium lokalnie.

## Wymagane (minimum)

- Python 3.11+
- Git 2.40+
- Docker 24+

## Zalecane narzędzia wspierające

- `pip` (do instalacji zależności z `requirements.txt`)
- `venv` (izolowane środowisko Python)
- `pre-commit` (hooki jakości kodu)
- `pytest` (uruchamianie testów)

## Zależności Pythona

Po instalacji Pythona uruchom:

```bash
python -m pip install -r requirements.txt
```

To zainstaluje biblioteki używane na laboratoriach (m.in. `scikit-learn`, `pandas`, `numpy`, `mlflow`, `dvc`, `fastapi`, `kfp`, `prometheus-client`).

## Narzędzia wykorzystywane tematycznie na laboratoriach

- MLflow – śledzenie eksperymentów
- DVC – wersjonowanie danych
- Kubeflow Pipelines – budowa pipeline’ów ML
- Prometheus + Grafana – monitoring modeli i metryk
- GitHub Actions – CI/CD