# Skrypty demonstracyjne do wykładów 1–3

Skrypty do uruchamiania na żywo w trakcie wykładów. Każdy odpowiada jednej sekcji wykładu
i działa samodzielnie na danych syntetycznych, więc nie trzeba pobierać żadnych zbiorów danych.

## Przygotowanie środowiska

```bash
# w katalogu głównym repozytorium
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install optuna                 # opcjonalnie, wykład 3 (bez niej skrypt użyje random search)
```

Wszystkie polecenia uruchamiaj **z katalogu głównego repozytorium**. Wyniki (pliki JSON, PNG,
modele, dane) trafiają do `lectures/scripts/lecture_0X/outputs/`, a runy MLflow do `./mlruns`.
Oba katalogi są w `.gitignore`.

## Wykład 1 – Wprowadzenie do MLOps

| Skrypt | Sekcja | Co pokazuje | Czas |
|---|---|---|---|
| `lecture_01/01_reproducibility.py` | 4.1 | Wyniki bez seeda i z seedem, zapis konfiguracji eksperymentu do JSON | ~5 s |
| `lecture_01/02_data_drift.py [--plot]` | 4.2–4.3 | Test KS dla 3 scenariuszy dryftu, concept drift (X bez zmian, accuracy spada do ~0.5) | ~2 s |
| `lecture_01/03_simple_mlops_system.py` | 8 | Rejestr modeli: trening → rejestracja → promocja tylko lepszego modelu → predykcja | ~5 s |

```bash
python lectures/scripts/lecture_01/01_reproducibility.py
python lectures/scripts/lecture_01/02_data_drift.py --plot
python lectures/scripts/lecture_01/03_simple_mlops_system.py
```

## Wykład 2 – Inżynieria danych

| Skrypt | Sekcja | Co pokazuje | Czas |
|---|---|---|---|
| `lecture_02/01_csv_vs_parquet.py [--rows N]` | 3 | Rozmiar, szybkość, column pruning i zachowanie typów: CSV vs Parquet | ~10 s (1 mln wierszy) |
| `lecture_02/02_data_pipeline.py` | 5 | Czyszczenie → feature engineering → `ColumnTransformer` → model → zapis pipeline'u | ~5 s |
| `lecture_02/03_data_validation.py [--corrupt]` | 6 | Walidacja danych jako bramka jakości (kod wyjścia 1 przy błędach) | ~1 s |
| `lecture_02/04_data_leakage.py` | 10 | Data leakage: selekcja cech przed CV (0.85 na szumie!) i target leakage | ~5 s |
| `lecture_02/05_dvc_demo.sh` | 8 | DVC: dwie wersje danych, `dvc push`, powrót do v1 przez `git checkout` + `dvc checkout` | ~10 s |

```bash
python lectures/scripts/lecture_02/01_csv_vs_parquet.py --rows 1000000
python lectures/scripts/lecture_02/02_data_pipeline.py
python lectures/scripts/lecture_02/03_data_validation.py
python lectures/scripts/lecture_02/03_data_validation.py --corrupt; echo "exit code: $?"
python lectures/scripts/lecture_02/04_data_leakage.py
bash   lectures/scripts/lecture_02/05_dvc_demo.sh     # wymaga git i dvc (Linux/macOS/WSL/Git Bash)
```

Demo DVC działa w katalogu tymczasowym (`mktemp -d`), więc nie zmienia tego repozytorium.
Na końcu wypisuje polecenie do usunięcia katalogów tymczasowych.

## Wykład 3 – Eksperymentowanie i tracking (MLflow)

| Skrypt | Sekcja | Co pokazuje | Czas |
|---|---|---|---|
| `lecture_03/01_mlflow_experiments.py` | 3 | 5 runów: parametry, metryki, CV, metryka krokowa, artefakty (macierz pomyłek), model z sygnaturą | ~40 s |
| `lecture_03/02_compare_runs.py` | 4 | `search_runs`, filtrowanie, ładowanie najlepszego modelu, Model Registry + alias `champion` | ~5 s |
| `lecture_03/03_hyperparameter_tuning.py [--trials N]` | 6 | HPO z zagnieżdżonymi runami (Optuna TPE lub random search) | ~3 s/próbę |
| `lecture_03/04_reproducibility.py` | 8, 10 | Seed, commit git, hash danych, wersje bibliotek w tagach; run ze statusem FAILED | ~5 s |

```bash
python lectures/scripts/lecture_03/01_mlflow_experiments.py
python lectures/scripts/lecture_03/02_compare_runs.py           # po 01
python lectures/scripts/lecture_03/03_hyperparameter_tuning.py --trials 15
python lectures/scripts/lecture_03/04_reproducibility.py
mlflow ui --port 5000                                            # → http://localhost:5000
```

Domyślnie skrypty zapisują runy lokalnie w `./mlruns`. Żeby użyć serwera MLflow:

```bash
mlflow server --host 127.0.0.1 --port 5000      # osobny terminal
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Reset przed kolejnym wykładem: `rm -rf mlruns`.

## Wskazówki dla prowadzącego

- Przed zajęciami uruchom każdy skrypt raz. Pierwszy import MLflow lub sklearn trwa dłużej.
- Skrypty są deterministyczne (`random_state=42`), więc wyniki na slajdach zgadzają się z tymi na żywo.
  Wyjątkiem są czasy wykonania i celowo losowa część `01_reproducibility.py`.
- Na wolnym sprzęcie zmniejsz parametry: `--rows 200000` w CSV vs Parquet albo `--trials 5` w HPO.
