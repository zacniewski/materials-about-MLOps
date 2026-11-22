# Blok 1 — ściągawka projektu (faza 1, VM Ubuntu 24.04)

Trzymajcie ten plik otwarty obok terminala podczas 3 godzin projektu. Plan minutowy jest w [`BLOK_1_projekt.md`](BLOK_1_projekt.md), a ogólne podstawy (terminal, Git/DVC, MLflow) w [ściągawce z laboratorium](../../labs-PL/BLOK_1_SCIAGA.md). Ta ściąga dotyczy tylko projektu.

## 1. Trzy rzeczy do sprawdzenia przed każdym poleceniem

| Sprawdź | Jak | Dobrze, jeśli… |
|---|---|---|
| Katalog | `pwd` | `/home/<login>/mlops/projekt-ZESPOL`, a **nie** `~/mlops/mlops-project` (laboratorium) ani `~/mlops/materials-about-MLOps` (repo kursu) |
| venv | początek linii | `(.venv)`; jeśli brak: `source ~/mlops/materials-about-MLOps/.venv/bin/activate` |
| `COURSE_REPO` (tylko przy kopiowaniu plików) | `echo $COURSE_REPO` | wypisuje ścieżkę; jeśli pusto: `COURSE_REPO=~/mlops/materials-about-MLOps` |

## 2. Role w zespole (ustalcie w pierwszych 5 minutach)

| Rola | Co robi na tym bloku | Czego **nie** robi |
|---|---|---|
| **Właściciel pipeline'u** | Jedyna osoba, która wpisuje polecenia w repo projektu na swojej VM, robi commity. | Nie wypełnia karty w tym samym czasie. |
| **Osoba od danych** | Czyta raport `inspect_dataset.py`, sprawdza target, klucz, zgodę/licencję, porównuje wyniki z tabelą w sekcji 5. | Nie pobiera nowych zbiorów po minucie 45. |
| **Notujący** | Na bieżąco wypełnia `KARTA_PROJEKTU.md` (sekcja 3), pilnuje czasu i reguły 10 minut. | Nie zostawia pól „do uzupełnienia później”. |

W zespole 2-osobowym połączcie „dane” i „notującego”. Pozostałe osoby mogą równolegle odtwarzać te same kroki na swoich VM, ale commit robi tylko właściciel.

## 3. Karta projektu — ściąga do wypełnienia (ścieżka A, churn)

Możecie przepisać, ale **każde pole przeczytajcie i zostawcie tylko to, z czym się zgadzacie**. Przy ścieżce B wpiszcie własne wartości z raportu inspekcji.

| Pole karty | Przykładowa odpowiedź dla demo churn |
|---|---|
| Problem i użytkownik | Dział retencji banku chce co tydzień dostać listę klientów z największym ryzykiem odejścia, żeby zadzwonić do nich z ofertą. |
| Typ zadania | Klasyfikacja binarna (`churn` = 1 — klient odchodzi). |
| Co nie wchodzi do fazy 1 | API, chmura, monitoring produkcyjny, tuning hiperparametrów, dane prawdziwych klientów. |
| Źródło i licencja / zgoda | Dane syntetyczne z `scripts/generate_data.py` (seed 42); brak danych osobowych. |
| Liczba rekordów i cech | 10 000 rekordów; 8 cech surowych + `customer_id` + `churn`; model dostaje 11 cech (8 surowych + 3 wyliczone w `preprocess.py`, lista w `params.yaml`). |
| Jednostka obserwacji / klucz | Jeden klient; klucz `customer_id` (0 duplikatów). |
| Kolumna docelowa | `churn` (0/1); 10,95% klasy 1 — dane niezbalansowane. |
| Braki i ograniczenia | 0 braków; dane syntetyczne, więc wynik nie przenosi się na prawdziwy bank. |
| Podział danych | 80/20 (8 000 / 2 000), stratyfikowany po `churn`, `random_state` w `params.yaml`; preprocessing liczony tylko na danych treningowych. |
| Metryka główna i próg | ROC-AUC na teście ≥ 0,70 (ranking klientów do telefonu; próg demonstracyjny). |
| Metryka pomocnicza | F1 dla klasy 1 (pokazuje problem z progiem 0,5). |
| Model bazowy | RandomForest z `params.yaml` (`n_estimators=100`, `max_depth=8`). |
| Warunek jakości danych | Etap `validate` przechodzi (`reports/validation.json`: `"passed": true`). |
| Warunek powtarzalności | `git clone` → `python scripts/generate_data.py` → `dvc repro` daje ten sam ROC-AUC. |
| Największe ryzyko | Niskie F1 przy progu 0,5 → w kolejnej fazie: `class_weight="balanced"` albo dobór progu. |

## 4. Mapa plików — co skąd się bierze

| Plik w `~/mlops/projekt-ZESPOL` | Skąd | Do czego | W Gicie? |
|---|---|---|---|
| `KARTA_PROJEKTU.md` | `project/faza_1/` | decyzje zespołu | tak |
| `dvc.yaml` | `project/faza_1/` (nadpisuje wersję startera) | 4 etapy: preprocess → validate → train → evaluate | tak |
| `params.yaml` | starter kit | parametry splitu i modelu | tak |
| `scripts/inspect_dataset.py` | `project/faza_1/scripts/` | szybki raport danych | tak |
| `scripts/evaluate_model.py` | `project/faza_1/scripts/` | ROC-AUC na teście + bramka jakości | tak |
| `scripts/check_phase1.py` | `project/faza_1/scripts/` | czy są wszystkie artefakty i czy jest PASS | tak |
| `tests/model/test_model_quality.py` | `project/faza_1/tests/model/` | test progu jakości modelu | tak |
| `data/raw/customers.parquet` | `generate_data.py` | dane surowe | **nie** — tylko `customers.parquet.dvc` |
| `data/processed/*.parquet`, `models/churn_model.pkl` | `dvc repro` | dane po podziale, model | **nie** — pilnuje ich `dvc.lock` |
| `reports/validation.json`, `reports/evaluation_metrics.json` | `dvc repro` | wynik walidacji i ewaluacji | **nie** — są wyjściami DVC (`reports/.gitignore`), ich hash jest w `dvc.lock` |
| `reports/metrics/*.json` | `dvc repro` | statystyki danych i metryki treningu | tak (metryki DVC) |

## 5. Polecenia z prawdziwymi wynikami

Wyniki pochodzą z przejścia na świeżej kopii starter kitu. Wasze liczby dla churnu powinny być **identyczne** (dane i model mają stały seed).

**Raport danych (minuty 45–60):**

```bash
python scripts/generate_data.py
python scripts/inspect_dataset.py --input data/raw/customers.parquet --target churn --id customer_id
```

```
Rows: 10,000
Columns (10): customer_id, age, income, tenure_months, num_products, has_credit_card, is_active, balance, num_transactions_30d, churn
Missing values: 0
ID customer_id: 0 duplicate row(s) after first occurrence; 0 missing value(s)
Target distribution (churn):
  0: 8,905 (89.05%)
  1: 1,095 (10.95%)
```

Jak czytać raport przy własnych danych (ścieżka B): `duplicate row(s)` > 0 oznacza, że klucz nie jest unikalny — wybierzcie inny albo ustalcie regułę deduplikacji. Klasa rzadsza niż ok. 5% → zanotujcie ryzyko i metrykę inną niż accuracy. Wiele braków w kolumnie → zapiszcie decyzję: usunąć, uzupełnić albo pominąć cechę.

**Pipeline (minuty 90–115):**

```bash
dvc add data/raw/customers.parquet
dvc repro
dvc dag
```

Oczekiwane: `Running stage 'preprocess'`, `'validate'`, `'train'`, `'evaluate'` (każdy z `Updating lock file 'dvc.lock'`). W `dvc dag` `validate` i `train` wychodzą z `preprocess`, a `evaluate` jest na dole.

**Testy i kontrola (minuty 115–145):**

```bash
pytest tests/unit tests/data -q      # 10 passed
pytest tests/model -q                # 2 passed
python scripts/check_phase1.py       # 10 × [OK] i na końcu:
                                     # ROC-AUC 0.7939... / minimum 0.7: PASS
cat reports/evaluation_metrics.json
dvc metrics show
```

```json
{"accuracy": 0.8885, "f1": 0.0591, "roc_auc": 0.7940, "min_roc_auc": 0.7, "quality_gate_passed": true}
```

`dvc metrics show` pokazuje dodatkowo `train_auc` ≈ 0,904, `train_f1` ≈ 0,215, `train_rows` 8000, `test_rows` 2000, `churn_rate_train` i `churn_rate_test` = 0,1095.

**Commit (minuty 130–145):**

```bash
git status --short     # przejrzyjcie listę PRZED commitem (sekcja 4, kolumna „W Gicie?”)
git add .
git commit -m "Faza 1: baseline"
dvc status             # oczekiwane: Data and pipelines are up to date.
```

## 6. Jak to opowiedzieć prowadzącemu (2 minuty na koniec)

Przygotujcie jedno zdanie do każdego punktu — to jest „wynik” bloku:

1. **Problem:** kto jest użytkownikiem i jaką decyzję wspiera model.
2. **Dane:** skąd, ile rekordów, jaki target, jaki rozkład klas.
3. **Pipeline:** 4 etapy DVC; co robi `validate` i dlaczego jest przed treningiem.
4. **Wynik:** test ROC-AUC 0,79 przy progu 0,70 → PASS; train AUC 0,90 jest wyższe, bo model widział te dane.
5. **Słabość:** F1 0,06 i accuracy 0,89 — accuracy prawie równa odsetkowi klasy 0, więc niewiele mówi; model rzadko przewiduje churn przy progu 0,5.
6. **Następny krok i właściciel:** np. „Ania: dobór progu / `class_weight` do bloku 2”.

## 7. Typowe błędy w projekcie

| Komunikat / objaw | Przyczyna | Naprawa |
|---|---|---|
| `cp: cannot stat '/labs-PL/...': No such file or directory` | pusta zmienna `COURSE_REPO` (nowa karta terminala) | `COURSE_REPO=~/mlops/materials-about-MLOps` i powtórzcie `cp` |
| `cp: cannot stat ...` mimo ustawionej zmiennej | repo kursu jest w innym miejscu | `ls ~/mlops` i poprawcie ścieżkę w `COURSE_REPO` |
| `check_phase1.py`: `Missing 10 required deliverable(s)` | uruchomienie poza katalogiem projektu | `cd ~/mlops/projekt-ZESPOL` i ponownie |
| `check_phase1.py`: `[ ] …` przy 1–2 plikach i `Missing 1 required deliverable(s)` | brak kopii pliku fazy 1 albo nie było `dvc repro` | powtórzcie „Krok 1” z `BLOK_1_projekt.md`, potem `dvc repro` |
| `dvc dag` pokazuje tylko `preprocess` i `train` | nie skopiowano `dvc.yaml` z fazy 1 | `cp "$COURSE_REPO/project/faza_1/dvc.yaml" dvc.yaml` i `dvc repro` |
| `ERROR: … 'scripts/evaluate_model.py' does not exist` w `dvc repro` | brak skryptu fazy 1 | skopiujcie `evaluate_model.py` i `check_phase1.py` do `scripts/` |
| `ERROR: file not found` / `pytest` nie znajduje `tests/model` | brak `mkdir -p tests/model` i kopii testu | powtórzcie dwa ostatnie polecenia „Kroku 1” |
| `dvc add`: `ERROR: output 'data/raw/customers.parquet' does not exist` | brak danych | `python scripts/generate_data.py` |
| `dvc repro` przerwany na `validate` | dane nie przechodzą walidacji (typowe w ścieżce B) | `cat reports/validation.json` → znajdźcie `"passed": false`; w B wróćcie do churnu na ten blok |
| `PASS`, a ROC-AUC różni się od 0,794 | inne dane lub zmienione `params.yaml` | `git diff params.yaml`; regeneracja `python scripts/generate_data.py` i `dvc repro` |
| `FAIL` / `quality_gate_passed: false` | ROC-AUC < 0,70 (np. po zmianie cech) | cofnijcie ostatnią zmianę (`git diff`), nie obniżajcie progu bez wpisu uzasadnienia w karcie |
| `git status` pokazuje `data/raw/customers.parquet` albo `models/*.pkl` | plik dodany do Gita przed `dvc add` | `git rm --cached <plik>`, potem `dvc add …` |
| `fatal: not a git repository` / `not inside of a DVC repository` | pominięto `git init` / `dvc init` w kopii projektu | wykonajcie brakujące `init` w `~/mlops/projekt-ZESPOL` |
| Kopia projektu jest „popsuta” i nikt nie wie dlaczego | — | utwórzcie nową kopię (`projekt-ZESPOL-2`) wg „Kroku 0”; skopiujcie do niej tylko wypełnioną kartę |

**Reguła 10 minut:** jeśli jeden błąd blokuje Was dłużej niż 10 minut, podnieście rękę i pokażcie pełny komunikat, `pwd` oraz `git status --short`. Ścieżka B, która nie działa po 10 minutach, wraca na churn.

## 8. Czego nie robić na tym bloku

- Nie przepisywać `preprocess.py`, `validation.py`, `train.py` pod własne dane — to praca między blokami ([`ADAPTACJA_WLASNYCH_DANYCH.md`](ADAPTACJA_WLASNYCH_DANYCH.md)).
- Nie obniżać progu `--min-roc-auc` w `dvc.yaml`, żeby uzyskać PASS.
- Nie stroić hiperparametrów — baseline ma być prosty i odtwarzalny; eksperymenty są w MLflow (opcjonalnie, minuty 145–160).
- Nie commitować danych, modeli, `mlflow.db`, `mlruns/`, `mlartifacts/`, haseł i tokenów.
- Nie pracować w repozytorium kursu ani w kopii z laboratorium.

## 9. Pytania kontrolne dla zespołu

1. Dlaczego `validate` jest osobnym etapem przed `train`, a nie częścią treningu?
2. Co by się stało z wynikiem testowym, gdyby skalowanie/czyszczenie liczyć na całym zbiorze przed podziałem?
3. Dlaczego `evaluation_metrics.json` nie jest w Gicie, a i tak wiemy, która wersja go wyprodukowała?
4. Dlaczego accuracy 0,89 to słaby wynik przy 10,95% churnu?
5. Czy próg ROC-AUC 0,70 jest uzasadniony dla Waszego użytkownika? Co należałoby wiedzieć, żeby go ustalić?
6. Co musi zrobić nowa osoba w zespole, żeby odtworzyć Wasz wynik od zera?
