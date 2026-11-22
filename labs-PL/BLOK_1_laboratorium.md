# Blok 1 — laboratorium (180 min)

## Cel i organizacja

W pierwszym bloku każdy zespół uruchamia powtarzalny pipeline DVC, sprawdza jego wyniki i wykonuje krótki eksperyment śledzony w MLflow. Pracujcie w parach: osoba przy terminalu wykonuje polecenia, druga zapisuje decyzje i wyniki; zamieńcie się rolami w połowie zajęć.

W kursie są trzy spotkania po 6 godzin: **blok 1: Lab 1 + Lab 2**, **blok 2: Lab 3 + Lab 4**, **blok 3: Lab 5 + Lab 7** (Lab 6 jest dodatkiem, jeśli wystarczy czasu). Szczegółowe materiały Lab 1 i Lab 2 pozostają dostępne osobno, ale na pierwsze spotkanie używajcie poniższej checklisty 180-minutowej.

## Przed zajęciami

Pracujemy na **maszynie wirtualnej Ubuntu 24.04 w VirtualBoxie** ([Lab 0](00_przygotowanie_maszyny_wirtualnej.md)). Wszystkie polecenia wykonujecie w terminalu VM, a UI MLflow otwieracie w Firefoksie wewnątrz VM.

- [ ] Przejdźcie listę [`BLOK_1_PRZED_ZAJECIAMI.md`](BLOK_1_PRZED_ZAJECIAMI.md) (20–30 min): aktualne repozytorium kursu, venv, kopia `~/mlops/mlops-project` i `python scripts/check_env.py` bez `[BŁĄD]`.
- [ ] Otwórzcie [`BLOK_1_SCIAGA.md`](BLOK_1_SCIAGA.md) — słowniczek, oczekiwane wyniki komend i typowe błędy. Gdy coś nie działa, najpierw sprawdźcie sekcję 5 ściągawki.
- [ ] Prowadzący: przypomnijcie o przygotowaniu na kilka dni przed zajęciami i poproście o wyniki `check_env.py` od osób z problemami; miejcie awaryjną VM/kopię starter kitu z zainstalowanym venv. Zespoły nadal samodzielnie wykonują `dvc add` i `dvc repro`.

## Jak pracujemy

- **Kod jest gotowy — nie programujecie.** Uruchamiacie polecenia z kolumny „Kod / komenda”, czytacie wyniki i porównujecie je z kolumną „Oczekiwany wynik”. Pliki `.py` otwierajcie, żeby zrozumieć, co robią, a nie żeby je poprawiać.
- **Każdy wiersz tabeli to punkt kontrolny.** Nie przechodźcie dalej, jeśli wynik wyraźnie różni się od oczekiwanego.
- **Reguła 10 minut:** jeśli utkniecie na dłużej niż 10 minut, zgłoście się z pełnym komunikatem błędu (ostatnie ~20 linii).
- **Minuty są orientacyjne.** Obowiązkowe jest minimum z sekcji „Kryteria wyjścia”; zadania „jeśli zostanie czas” są dla szybszych zespołów.

## Checklista na 3 godziny

| Minuty | Zadanie / checklista | Kod / komenda — kiedy i jak użyć | Oczekiwany wynik |
|---:|---|---|---|
| 0–10 | Ustalcie role, otwórzcie katalog projektu i potwierdźcie gotowość środowiska. | `cd ~/mlops/mlops-project`; `source ~/mlops/materials-about-MLOps/.venv/bin/activate` (jeśli brak `(.venv)`); `python scripts/check_env.py`. Jeśli nie macie kopii, zróbcie ją wg `BLOK_1_PRZED_ZAJECIAMI.md`, punkt 4. | Brak pozycji `[BŁĄD]`; na końcu `Środowisko gotowe na zajęcia.` |
| 10–25 | Przejrzyjcie instrukcję, konfigurację pipeline'u i parametry; wskażcie wejście oraz etapy. | Otwórz `README.md`, `dvc.yaml`, `params.yaml`. Na razie nie zmieniaj kodu — pipeline jest gotowy do uruchomienia. | Potraficie wskazać dane raw, preprocessing, trening i artefakty wynikowe. |
| 25–45 | Przygotujcie dane i zainicjujcie Git/DVC (pomijajcie inicjalizację, jeśli repozytorium jest już gotowe). | `python scripts/generate_data.py`; `git init`; `dvc init`; `dvc add data/raw/customers.parquet`. Po `dvc add` wykonajcie `cat data/raw/customers.parquet.dvc` i `cat data/raw/.gitignore`. | `Wygenerowano 10,000 rekordów`, `Churn rate: 10.95%`; plik `.dvc` zawiera `md5` i `size`; `data/raw/.gitignore` wyklucza `customers.parquet` z Gita. |
| 45–70 | Odtwórzcie pipeline i sprawdźcie, co zostało wytworzone. Uruchomcie go drugi raz. | `dvc repro` — wykonuje etapy z `dvc.yaml`; jeśli etap się nie powiedzie, naprawcie błąd przed dalszymi krokami. Potem `dvc repro` jeszcze raz i `ls data/processed models reports/metrics`. | Pierwszy raz: `Running stage 'preprocess'`, `Running stage 'train'`, `Train AUC: 0.90…`. Drugi raz: `didn't change, skipping`. Są pliki `train/test.parquet`, `churn_model.pkl` i metryki JSON. |
| 70–85 | Odczytajcie zależności i wynik, zamiast oceniać model po jednej liczbie. Zróbcie pierwszy commit. | `dvc dag`; `dvc status`; `dvc metrics show`; `pytest tests/unit -q`; potem `git add .` i `git commit -m "Lab 1: pipeline DVC"`. | DAG: `customers.parquet.dvc → preprocess → train`; `6 passed`; `train_auc ≈ 0.904`, `train_f1 ≈ 0.21` (niski F1 przy 11% churnu jest normalny). W commicie nie ma danych ani modelu. |
| 85–95 | **Przerwa / punkt kontrolny.** Zapiszcie jedną rzecz, którą Git śledzi, i jedną, którą śledzi DVC. | Bez komend. Jeśli `dvc repro` nie działa, poproście prowadzącego o pomoc teraz, zamiast przechodzić dalej. | Zespół ma działający pipeline lub konkretny komunikat błędu do rozwiązania. |
| 95–115 | Uruchomcie lokalny serwer MLflow i sprawdźcie UI. | Nowa karta terminala (`Ctrl+Shift+T`), `cd ~/mlops/mlops-project`, venv, potem `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000`. UI: Firefox w VM, `http://127.0.0.1:5000`. Tej karty już nie używajcie do innych poleceń. | W terminalu serwera linia `Listening at: http://127.0.0.1:5000`; UI MLflow otwiera się w VM. |
| 115–140 | Wykonajcie szybkie porównanie eksperymentów i obejrzyjcie runy. | W pierwszym (roboczym) terminalu: `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; `python -m scripts.run_experiments --quick`; `python -m scripts.compare_experiments`. W UI zaznaczcie oba runy i kliknijcie **Compare**. | W MLflow widać dwa runy z CV ROC-AUC ok. 0,79–0,80; ranking tekstowy zgadza się z UI. |
| 140–160 | Zarejestrujcie jeden run, wybierając jego `RUN_ID` z wyniku porównania. | `python -m scripts.registry_aliases RUN_ID --name churn-random-forest --alias candidate` — zastąp `RUN_ID` prawdziwym identyfikatorem runu. | W Model Registry jest wersja modelu z aliasem `candidate`. |
| 160–175 | Zapiszcie dowody, sprawdźcie status i przygotujcie krótkie podsumowanie. Odpowiedzcie na pytania kontrolne ze ściągawki (sekcja 7). | `git status` (nie powinno być `mlflow.db` ani `mlartifacts/` — wyklucza je `.gitignore`); `git add .`; `git commit -m "Lab 2: eksperymenty MLflow"`. W notatce: wynik AUC, ID wybranego runu, różnica Git/DVC, jeden artefakt. | Każdy zespół może pokazać pipeline, wynik i run; w Git są tylko kod, konfiguracja, wskaźniki i metryki. |
| 175–180 | Zatrzymajcie serwer MLflow i sprawdźcie kryteria wyjścia. | W terminalu serwera naciśnij `Ctrl+C`. | Proces serwera jest zatrzymany; notatki są zapisane. |

## Jeśli zostanie czas (dla szybszych zespołów)

- [ ] **Eksperyment z parametrem w DVC:** zmieńcie w `params.yaml` `max_depth: 8` na `max_depth: 4`, potem `dvc status`, `dvc repro`, `dvc params diff`, `dvc metrics diff`. Oczekiwane: przelicza się tylko `train` (`preprocess` jest pomijany), `train_auc` spada do ok. 0,82, a `train_f1` do ok. 0,02. Wnioski zapiszcie, a parametr przywróćcie (`git checkout params.yaml` i `dvc repro`).
- [ ] **Pełna siatka eksperymentów:** `python -m scripts.run_experiments` (bez `--quick`) i porównanie w UI — który parametr najbardziej wpływa na ROC-AUC?
- [ ] **Postęp:** `python scripts/check_progress.py` — pozycje Lab 1 i Lab 2 powinny mieć `[OK]`.

## Kryteria wyjścia — minimum

- [ ] `dvc repro` kończy się powodzeniem; zespół rozumie DAG i lokalizację wyników.
- [ ] `pytest tests/unit -q` przechodzi.
- [ ] Zespół potrafi wyjaśnić, że Git przechowuje kod/konfigurację/wskaźniki, a DVC wersjonuje zawartość danych i artefaktów.
- [ ] W MLflow są widoczne dwa runy z parametrami i metrykami; zapisany jest jeden `run_id`.
- [ ] Serwer był związany z `127.0.0.1` i został zatrzymany po zajęciach.

## Jeśli coś nie działa

- **Brakuje danych:** uruchom `python scripts/generate_data.py`, a następnie `dvc add data/raw/customers.parquet` i `dvc repro`.
- **Git/DVC już zainicjowane:** nie powtarzaj `git init` ani `dvc init`; przejdź do kroku, którego brakuje.
- **Pełna lista objawów i napraw:** [`BLOK_1_SCIAGA.md`](BLOK_1_SCIAGA.md), sekcja 5.
- **Brak `(.venv)` / `externally-managed-environment`:** `source ~/mlops/materials-about-MLOps/.venv/bin/activate`.
- **DVC zgłasza błąd powłoki:** `export SHELL=/bin/bash`.
- **Port 5000 zajęty:** zatrzymaj stary serwer (`Ctrl+C` albo `pkill -f "mlflow server"`).
- **Błąd połączenia / migracji MLflow z SQLAlchemy:** sprawdź venv i wersję poleceniem `python -m pip show sqlalchemy`. Starter kit wymaga `sqlalchemy<2.1`; w aktywnym venv napraw zależność poleceniem `python -m pip install 'sqlalchemy<2.1'`, po czym uruchom serwer ponownie.
- **Kończy się czas:** obowiązkowe są działające `dvc repro`, testy i kontrola metryk; MLflow wykonaj w trybie `--quick`, pomijając pełną siatkę eksperymentów.
- **Nie commituj:** danych osobowych, sekretów, bazy `mlflow.db` ani artefaktów modelu.

## Co zabrać do części projektowej

Zachowajcie kartę wyników: problem, potencjalny target, dostępne dane, metrykę oraz ryzyko. W projekcie zdecydujecie, czy na pierwszym spotkaniu uruchamiacie bezpieczny wariant demo churn, czy macie już dane gotowe do szybkiej inspekcji.