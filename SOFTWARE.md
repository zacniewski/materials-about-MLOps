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

## Windows: terminal z komendami Linux

Jeśli pracujesz na Windows i chcesz używać komend linuxowych z laboratoriów (np. `source`, `mkdir -p`, `touch`, `cat`), zainstaluj jedno z poniższych narzędzi:

- **WSL 2 (Windows Subsystem for Linux)** — rekomendowane, daje pełne środowisko Linux (np. Ubuntu) w Windows.
- **Git Bash** — lekkie rozwiązanie instalowane razem z Git for Windows; wystarcza do większości podstawowych komend z laboratoriów.
- **MSYS2/Cygwin** — alternatywy z szerokim zestawem narzędzi unixowych.

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