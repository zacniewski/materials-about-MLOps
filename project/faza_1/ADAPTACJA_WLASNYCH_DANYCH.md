# Ścieżka B — adaptacja starter kitu do własnych danych (praca między blokami)

**Czas:** 2–4 godziny pracy zespołu, **po** bloku 1. Na samym bloku 1 wystarczy inspekcja danych i mapowanie kolumn w karcie projektu.
**Warunek wejścia:** baseline na danych demo churn działa w waszym repo (`dvc repro`, testy, `check_phase1.py` → `PASS`). Dzięki temu każdy błąd po zmianach wynika z adaptacji, a nie ze środowiska.
**Zakres:** klasyfikacja binarna na jednym pliku tabelarycznym (CSV/Parquet). Regresja, wiele plików, tekst lub obrazy — skonsultujcie najpierw z prowadzącym.

## Zasady bezpiecznej adaptacji

1. **Pracujcie na gałęzi:** `git switch -c dane-wlasne`. Gdy coś się zepsuje, `git switch main` przywraca działający baseline.
2. **Jeden krok = jedna kontrola = jeden commit.** Nie zmieniajcie wszystkich plików naraz.
3. **Nie zmieniajcie nazw plików wyjściowych** (`data/processed/train.parquet`, `test.parquet`, `models/churn_model.pkl`). Nazwa modelu jest „brzydka”, ale dzięki temu `check_phase1.py`, `dvc.yaml` i testy modelu działają bez zmian. Zmianę nazwy zostawcie na później.
4. **Target jako 0/1.** Zamieńcie etykiety (`yes/no`, `>50K/<=50K`) na liczby 0/1 już przy konwersji danych — wtedy walidacja targetu i metryki działają bez zmian.
5. **Bez danych wrażliwych.** Usuńcie kolumny identyfikujące osoby (imię, e-mail, PESEL, adres) przed dodaniem danych do DVC.

## Mapa zmian (wypełnijcie przed kodowaniem)

| W starterze (churn) | U was | Gdzie występuje |
|---|---|---|
| plik `data/raw/customers.parquet` | `data/raw/dataset.parquet` | `dvc.yaml` (etap `preprocess`), `src/data/preprocess.py` (`--input`) |
| klucz `customer_id` | `…` (albo brak → utworzymy `row_id`) | `preprocess.py` (`drop_duplicates`), `validation.py` (`unique_customer_id`), `scripts/evaluate_model.py` (wykluczenie z cech) |
| target `churn` | `…` | `preprocess.py` (`stratify`, statystyki), `validation.py` (`target`), `src/models/train.py` (`y_train`), etap `evaluate` w `dvc.yaml` (`--target`) |
| cechy `age`, `income`, … | `…` | `params.yaml` (`features`), `validation.py` (`required`, reguły domenowe), `preprocess.py` (filtry i feature engineering) |
| próg `--min-roc-auc 0.70` | `…` (uzasadnijcie) | etap `evaluate` w `dvc.yaml` |

## Krok po kroku

### Krok 1 — konwersja danych do `data/raw/dataset.parquet` (ok. 30 min)

Utwórzcie `scripts/prepare_data.py` na podstawie szablonu i uzupełnijcie stałe na górze:

```python
"""Convert the team's source file into data/raw/dataset.parquet."""

from pathlib import Path

import pandas as pd

SOURCE = Path("data/source/bank.csv")      # wasz plik źródłowy (poza Git!)
OUTPUT = Path("data/raw/dataset.parquet")
TARGET = "y"                               # nazwa kolumny targetu w źródle
POSITIVE = "yes"                           # wartość oznaczająca klasę 1
ID_COLUMN = None                           # nazwa klucza albo None
DROP = []                                  # kolumny wrażliwe / zbędne

frame = pd.read_csv(SOURCE, sep=",")       # dla UCI Bank Marketing: sep=";"
frame = frame.drop(columns=DROP)
frame[TARGET] = (frame[TARGET] == POSITIVE).astype(int)
if ID_COLUMN is None:
    frame.insert(0, "row_id", range(1, len(frame) + 1))
frame = pd.get_dummies(frame, drop_first=True, dtype=int)  # cechy kategoryczne → 0/1
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(OUTPUT, index=False)
print(f"Saved {len(frame):,} rows and {frame.shape[1]} columns to {OUTPUT}")
```

`get_dummies` zamienia kolumny tekstowe na liczbowe, bo model w starterze (`StandardScaler` + Random Forest) przyjmuje tylko liczby. Dopiszcie `data/source/` do `.gitignore`.

**Kontrola:**

```bash
python scripts/prepare_data.py
python scripts/inspect_dataset.py --input data/raw/dataset.parquet --target y --id row_id
```

Oczekiwane: target ma tylko wartości `0` i `1`, klucz nie ma duplikatów, braki są znane (zapiszcie decyzję: usuwamy wiersze czy uzupełniamy). Commit: `git add scripts/prepare_data.py .gitignore` + `git commit`.

### Krok 2 — dane w DVC i pipeline wskazuje nowy plik (ok. 15 min)

```bash
dvc add data/raw/dataset.parquet
```

W `dvc.yaml`, etap `preprocess`: zamieńcie zależność `data/raw/customers.parquet` na `data/raw/dataset.parquet`, a polecenie na `python src/data/preprocess.py --input data/raw/dataset.parquet`.

**Kontrola:** `dvc dag` pokazuje `data/raw/dataset.parquet.dvc` na początku grafu.

### Krok 3 — `src/data/preprocess.py` (ok. 30 min)

- `drop_duplicates(subset=["customer_id"])` → wasz klucz (np. `row_id`).
- Usuńcie lub zastąpcie filtry `age` / `income` regułami sensownymi dla waszych danych (albo usuńcie wiersze z brakami: `df = df.dropna()`).
- Usuńcie trzy linie feature engineeringu (`income_per_age`, `balance_per_product`, `is_long_tenure`) albo dodajcie własne cechy.
- `stratify=df["churn"]` oraz `churn_rate_*` → wasz target (zostawcie nazwy kluczy w statystykach albo zmieńcie je konsekwentnie).
- `n_features`: `df.shape[1] - 2` zakłada dokładnie dwie kolumny niebędące cechami (klucz i target).

**Kontrola:** `dvc repro preprocess` — etap kończy się bez błędu, w `reports/metrics/data_stats.json` liczby wierszy i odsetek klasy 1 zgadzają się z raportem z kroku 1.

### Krok 4 — `src/data/validation.py` (ok. 30 min)

- Zbiór `required`: wpiszcie swój klucz, najważniejsze cechy i `target`.
- Domyślny `target: str = "churn"` → wasz target (albo przekazujcie go z `scripts/validate_data.py`).
- Reguła `unique_customer_id` → wasz klucz.
- Reguły `age_range`, `positive_income` i podobne: zastąpcie 1–2 regułami domenowymi dla waszych danych (np. `duration >= 0`), resztę usuńcie.

**Kontrola:** `dvc repro validate` i `cat reports/validation.json` — `"passed": true`, a każda reguła ma sensowny opis.

### Krok 5 — cechy i trening (ok. 20 min)

- `params.yaml` → lista `features`: kolumny z pliku po kroku 3 **bez klucza i targetu**. Listę wypiszecie poleceniem:
  `python -c "import pandas as pd; print(list(pd.read_parquet('data/processed/train.parquet').columns))"`
- `src/models/train.py`: `y_train = train_df["churn"]` → wasz target.

**Kontrola:** `dvc repro train` — w logu `Trening na … próbkach, N cechach` z poprawnym `N`.

### Krok 6 — ewaluacja i próg jakości (ok. 20 min)

- W `scripts/evaluate_model.py` zbiór `{target, "customer_id"}` → wasz klucz zamiast `customer_id`.
- W `dvc.yaml`, etap `evaluate`: dopiszcie `--target WASZ_TARGET` i ustawcie `--min-roc-auc` na uzasadnioną wartość (np. wynik prostego modelu referencyjnego z małym zapasem). Uzasadnienie wpiszcie do `KARTA_PROJEKTU.md`.

**Kontrola:** `dvc repro` (cały pipeline), potem `python scripts/check_phase1.py` → `PASS`.

### Krok 7 — testy (ok. 30–45 min)

- `tests/data/test_validation.py` i `tests/unit/test_preprocess.py` budują małe ramki danych z kolumnami churn. Zamieńcie je na kilka wierszy z waszym schematem — zachowajcie **sens** testów (np. „duplikat klucza jest wykrywany”, „podział zachowuje proporcję klas”).
- `tests/model/test_model_quality.py` czyta tylko raport ewaluacji — nie wymaga zmian.

**Kontrola:**

```bash
pytest tests/unit tests/data -q
pytest tests/model -q
```

Oczekiwane: same `passed`. Nie usuwajcie testów, które nie przechodzą — poprawcie je albo kod.

### Krok 8 — domknięcie

- [ ] README projektu opisuje źródło danych, licencję i polecenia odtworzenia (`prepare_data.py` → `dvc repro`).
- [ ] `KARTA_PROJEKTU.md`: target, metryka, próg, ograniczenia danych.
- [ ] `git status` nie pokazuje danych (`data/source/`, `data/raw/*.parquet`) ani modelu.
- [ ] Merge gałęzi: `git switch main` → `git merge dane-wlasne`.

## Typowe problemy

| Objaw | Przyczyna | Naprawa |
|---|---|---|
| `KeyError: 'churn'` | pozostała nazwa targetu w którymś pliku | wyszukajcie `churn` w `src/`, `scripts/`, `dvc.yaml` |
| `KeyError: 'customer_id'` | pozostała nazwa klucza | jw. dla `customer_id` (m.in. `evaluate_model.py`) |
| `ValueError: could not convert string to float` | kolumna tekstowa w cechach | `get_dummies` w kroku 1 albo usunięcie kolumny z `features` |
| `Input contains NaN` | braki w cechach | decyzja z kroku 1: `dropna()` w `preprocess.py` albo uzupełnianie |
| `ValueError: The least populated class…` przy `stratify` | target nie jest 0/1 albo klasa ma 1 rekord | sprawdźcie mapowanie `POSITIVE` w kroku 1 |
| walidacja `FAIL: required_columns` | `required` zawiera kolumnę usuniętą w preprocessingu | zaktualizujcie `required` |
| ROC-AUC ≈ 0,5 | brak sygnału w cechach albo pomylony target | sprawdźcie rozkład targetu i listę `features` |
| ROC-AUC ≈ 1,0 | wyciek targetu (cecha zawiera odpowiedź, np. `duration` w Bank Marketing) | usuńcie podejrzaną cechę i opiszcie to w karcie |

Jeśli po 2 godzinach pipeline nadal nie przechodzi, wróćcie na `main` (baseline churn) i przynieście listę błędów na blok 2 — to wciąż spełnia kryteria fazy 1.
