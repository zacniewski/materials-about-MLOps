# Blok 1 — ściągawka (VM Ubuntu 24.04)

Trzymajcie ten plik otwarty obok terminala. Wszystkie polecenia wykonujecie w VM, w katalogu głównym swojej kopii projektu (`~/mlops/mlops-project` na laboratorium, `~/mlops/projekt-ZESPOL` na projekcie), z aktywnym venv. Na część projektową jest osobna ściąga: [`project/faza_1/BLOK_1_SCIAGA_projekt.md`](../project/faza_1/BLOK_1_SCIAGA_projekt.md).

## 1. Zanim wpiszecie jakiekolwiek polecenie

| Sprawdź | Jak | Dobrze, jeśli… |
|---|---|---|
| Czy jestem w dobrym katalogu? | `pwd` i `ls` | widać `dvc.yaml`, `params.yaml`, `src/`, `scripts/` |
| Czy venv jest aktywny? | spójrz na początek linii | jest `(.venv)`; jeśli nie: `source ~/mlops/materials-about-MLOps/.venv/bin/activate` |
| Czy środowisko jest kompletne? | `python scripts/check_env.py` | brak pozycji `[BŁĄD]` |

**Terminal w Ubuntu:** wklejanie `Ctrl+Shift+V`, kopiowanie `Ctrl+Shift+C`. Samo `Ctrl+C` **przerywa** działający program (tak zatrzymujecie serwer MLflow). Nowa karta terminala: `Ctrl+Shift+T` — przejdźcie w niej do katalogu projektu (`cd ~/mlops/mlops-project`) i sprawdźcie `(.venv)`. Strzałka w górę przywołuje poprzednie polecenie, `Tab` dopełnia nazwy plików.

## 2. Słowniczek — minimum na blok 1

| Pojęcie | Co to znaczy w naszym projekcie |
|---|---|
| **venv** | Odizolowany Python z pakietami kursu (`~/mlops/materials-about-MLOps/.venv`). Ubuntu 24.04 nie pozwala instalować pakietów globalnie. |
| **Git** | Wersjonuje **kod, konfigurację i małe pliki wskaźnikowe** (`*.dvc`, `dvc.lock`). |
| **DVC** | Wersjonuje **duże pliki** (dane, modele) w swoim cache i pilnuje, które etapy pipeline'u trzeba przeliczyć. |
| **plik `.dvc`** | Mały plik tekstowy z hashem `md5` danych, np. `data/raw/customers.parquet.dvc`. Trafia do Gita zamiast danych. |
| **stage (etap)** | Krok w `dvc.yaml`: polecenie `cmd`, wejścia `deps`, parametry `params`, wyjścia `outs`, metryki `metrics`. |
| **`dvc.lock`** | Zapis hashy wejść i wyjść po ostatnim `dvc repro`. Dzięki niemu DVC wie, że „nic się nie zmieniło”. |
| **DAG** | Graf zależności etapów (`dvc dag`): `customers.parquet.dvc → preprocess → train`. |
| **baseline** | Pierwszy, prosty model odniesienia. Ma być powtarzalny, nie najlepszy. |
| **ROC-AUC** | Jak dobrze model **szereguje** klientów od najbardziej do najmniej zagrożonych odejściem (0,5 = losowo, 1,0 = idealnie). |
| **F1** | Średnia harmoniczna precision i recall dla klasy „churn”. Przy ~11% churnu i progu 0,5 bywa niska — to normalne. |
| **CV (cross-validation)** | Ocena na kilku podziałach zbioru treningowego. Uczciwsza niż wynik „na treningu”. |
| **MLflow run** | Jedno uruchomienie eksperymentu z zapisanymi parametrami, metrykami i artefaktami. |
| **experiment** | Grupa runów do porównania w UI MLflow. |
| **Model Registry / alias** | Katalog wersji modeli; alias (np. `candidate`) to etykieta wskazująca wybraną wersję. |

## 3. Komendy z oczekiwanym wynikiem (laboratorium)

| Kiedy | Polecenie | Co powinniście zobaczyć |
|---|---|---|
| Generowanie danych | `python scripts/generate_data.py` | `Wygenerowano 10,000 rekordów`, `Churn rate: 10.95%` |
| Start repozytorium | `git init` oraz `dvc init` | `Initialized empty Git repository…`; DVC tworzy katalog `.dvc/` |
| Wersjonowanie danych | `dvc add data/raw/customers.parquet` | podpowiedź `git add data/raw/.gitignore data/raw/customers.parquet.dvc` |
| Podgląd wskaźnika | `cat data/raw/customers.parquet.dvc` | `md5: …`, `size: …`, `path: customers.parquet` |
| Pipeline | `dvc repro` | `Running stage 'preprocess'`, `Running stage 'train'`, `Train AUC: 0.90…`, `Updating lock file 'dvc.lock'` |
| Drugi raz bez zmian | `dvc repro` | `Stage 'preprocess' didn't change, skipping` — DVC nie liczy niczego od nowa |
| Graf | `dvc dag` | trzy prostokąty: `customers.parquet.dvc → preprocess → train` |
| Metryki | `dvc metrics show` | `total_rows 10000`, `train_rows 8000`, `test_rows 2000`, `train_auc ≈ 0.904`, `train_f1 ≈ 0.21` |
| Stan | `dvc status` | `Data and pipelines are up to date.` |
| Testy | `pytest tests/unit -q` | `6 passed` |
| Postęp | `python scripts/check_progress.py` | `[OK]` przy pozycjach Lab 1 i Lab 2 (pełne `8/8` po Lab 3) |
| Commit | `git add .` oraz `git commit -m "Lab 1: pipeline DVC"` | Do Gita trafiają kod, `*.dvc`, `dvc.lock`, metryki JSON; **dane i model nie** (pilnuje tego `.gitignore`) |

### Eksperyment z parametrem (jeśli zostanie czas)

1. W `params.yaml` zmień np. `max_depth: 8` na `max_depth: 4`.
2. `dvc status` — DVC pokaże, że etap `train` jest nieaktualny (zmienione `params`).
3. `dvc repro` — przeliczy się tylko `train`, `preprocess` zostanie pominięty.
4. `dvc params diff` i `dvc metrics diff` — porównanie z ostatnim commitem.
5. Wróć do poprzedniej wartości albo zrób commit z opisem, co zmieniliście.

### MLflow — dwa terminale

| Terminal 1 (serwer — zostaw uruchomiony) | Terminal 2 (praca) |
|---|---|
| `cd ~/mlops/mlops-project` | `cd ~/mlops/mlops-project` |
| `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000` | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000` |
| czekaj na linię `Listening at: http://127.0.0.1:5000` | `python -m scripts.run_experiments --quick` |
| na koniec zajęć: `Ctrl+C` | `python -m scripts.compare_experiments` |

UI otwieracie w **Firefoksie wewnątrz VM**: `http://127.0.0.1:5000`. Oczekiwane wyniki `--quick`: 2 runy, CV ROC-AUC ok. 0,79–0,80. Zmienna `MLFLOW_TRACKING_URI` obowiązuje tylko w terminalu, w którym ją ustawiliście — w nowej karcie ustawcie ją ponownie.

> **Zasada kursu:** MLflow zawsze na `127.0.0.1` — w `--host`, w `MLFLOW_TRACKING_URI` i w przeglądarce. Nie używajcie `0.0.0.0` (MLflow nie ma logowania) ani `localhost` (bywa rozwiązywany na IPv6 `::1`). Chcecie mimo to UI w przeglądarce **hosta**? Nie zmieniajcie `--host`; na hoście uruchomcie tunel SSH `ssh -p 2222 -N -L 5000:127.0.0.1:5000 student@127.0.0.1` (wymaga SSH z Lab 0, część 4) i otwórzcie `http://127.0.0.1:5000`. Samo przekierowanie portu 5000 w VirtualBoxie tu nie zadziała.

## 4. Jak czytać liczby (dane demo churn)

| Liczba | Typowa wartość | Interpretacja |
|---|---|---|
| Churn rate | ~11% | Klasy są niezbalansowane: model „zawsze nie odejdzie” miałby accuracy ~89%. Dlatego accuracy nie jest dobrą metryką. |
| Train AUC | ~0,90 | Wynik na danych, na których model się uczył — optymistyczny. |
| CV / test ROC-AUC | ~0,79–0,80 | Uczciwsza ocena. Różnica względem ~0,90 to sygnał lekkiego przeuczenia, nie błąd. |
| F1 | 0,03–0,21 | Niskie przy domyślnym progu 0,5 i rzadkiej klasie. Nie „naprawiajcie” tego na bloku 1 — zapiszcie jako obserwację. |

## 5. Typowe błędy i szybkie naprawy

| Komunikat / objaw | Przyczyna | Naprawa |
|---|---|---|
| `error: externally-managed-environment` | venv nieaktywny | `source ~/mlops/materials-about-MLOps/.venv/bin/activate` |
| `ModuleNotFoundError: No module named 'src'` / `'scripts'` | uruchomienie spoza katalogu projektu | `cd ~/mlops/mlops-project`; skrypty z kropką uruchamiaj jako `python -m scripts.nazwa` |
| `FileNotFoundError: data/raw/customers.parquet` | brak danych | `python scripts/generate_data.py`, potem `dvc add …` |
| `ERROR: … not a git repository` / `you are not inside of a DVC repository` | brak `git init` / `dvc init` | wykonaj brakujące `init` (tylko raz) |
| `ERROR: output 'data/raw/customers.parquet' is already tracked by SCM` | plik dodano wcześniej do Gita | `git rm -r --cached data/raw/customers.parquet`, potem `dvc add …` |
| DVC zgłasza błąd uruchamiania powłoki / `zsh: not found` | zła zmienna `SHELL` | `export SHELL=/bin/bash` |
| `dvc repro` nic nie robi | nic się nie zmieniło — to poprawne zachowanie | `dvc repro --force`, jeśli naprawdę chcesz przeliczyć |
| `Connection refused` przy `run_experiments` | serwer MLflow nie działa albo zła zmienna | sprawdź terminal 1; `echo $MLFLOW_TRACKING_URI` w terminalu 2 |
| `Address already in use` / port 5000 zajęty | stary serwer nadal działa (np. zamknięto okno terminala zamiast `Ctrl+C`) | `Ctrl+C` w starym terminalu; jeśli terminala już nie ma: `pkill -f "mlflow server"`; awaryjnie `--port 5001` i `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` |
| błąd importu / migracji z `sqlalchemy` przy `mlflow server` | SQLAlchemy 2.1 | `python -m pip install 'sqlalchemy<2.1'` i ponowny start serwera |
| runy nie pojawiają się w UI | runy zapisały się lokalnie do `mlruns/` (brak zmiennej) | ustaw `MLFLOW_TRACKING_URI` i uruchom `run_experiments` ponownie |
| `Please tell me who you are` przy commicie | brak autora w Git | `git config --global user.name "…"` i `user.email "…"` |
| Wszystko się posypało | — | nowa kopia starter kitu do nowego katalogu albo przywrócenie migawki VM |

**Zasada pierwszego błędu:** czytajcie komunikat od dołu — ostatnia linia zwykle mówi, co się stało. Naprawiajcie pierwszy błąd, zanim uruchomicie kolejne polecenia. Jeśli po 10 minutach nadal stoicie, podnieście rękę i pokażcie pełny komunikat.

## 6. Czego nie commitować

Danych (`data/raw/*.parquet`, `data/processed/`), modeli (`models/*.pkl`), bazy `mlflow.db`, katalogów `mlartifacts/` i `mlruns/`, haseł i tokenów. Starter kit ma już odpowiedni `.gitignore`. Jeśli `git status` pokazuje któryś z tych plików do dodania, **zatrzymajcie się** i sprawdźcie, czy wykonaliście `dvc add`.

## 7. Pytania kontrolne na koniec laboratorium

1. Co dokładnie zapisuje Git, a co DVC, dla pliku `customers.parquet`?
2. Dlaczego drugie `dvc repro` bez zmian nic nie przelicza i skąd DVC to wie?
3. Który etap przeliczy się po zmianie `max_depth` w `params.yaml`, a który zostanie pominięty?
4. Dlaczego Train AUC (~0,90) jest wyższe od CV ROC-AUC (~0,80)?
5. Dlaczego accuracy jest mylącą metryką przy 11% churnu?
6. Po co alias `candidate` w Model Registry, skoro mamy `run_id`?
