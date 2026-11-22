# Blok 1 — projekt (180 min)

## Cel

Po tej części zespół ma uzgodnioną kartę projektu, wykonalny zakres fazy 1 oraz odtwarzalny baseline. **Domyślnie wybierzcie ścieżkę A — lokalny, syntetyczny churn ze starter kitu.** Wybierajcie ścieżkę B z własnym zbiorem tylko wtedy, gdy już macie plik CSV/Parquet, prawo do użycia danych i jasny target; w przeciwnym razie nie traćcie spotkania na pobieranie i czyszczenie nieznanych danych.

## Jak pracujemy

- **Środowisko:** ta sama VM Ubuntu 24.04 i ten sam venv co na laboratorium (`source ~/mlops/materials-about-MLOps/.venv/bin/activate`). Polecenia, słowniczek i typowe błędy: [`../../labs-PL/BLOK_1_SCIAGA.md`](../../labs-PL/BLOK_1_SCIAGA.md).
- **Ściągawka projektu:** [`BLOK_1_SCIAGA_projekt.md`](BLOK_1_SCIAGA_projekt.md) — role, wzór wypełnionej karty, mapa plików, prawdziwe wyniki każdego polecenia, typowe błędy i 2-minutowe podsumowanie dla prowadzącego.
- **Projekt = nowa, osobna kopia starter kitu w `~/mlops/projekt-ZESPOL`** (zamieńcie `ZESPOL` na nazwę zespołu, bez spacji i polskich znaków). Nie pracujcie w kopii z laboratorium (`~/mlops/mlops-project`) ani w repozytorium kursu — plik `dvc.yaml` fazy 1 nadpisałby wersję z laboratorium.
- **Repo prowadzi jedna osoba** (właściciel pipeline'u) na swojej VM; pozostałe osoby czytają wyniki, wypełniają kartę i sprawdzają oczekiwane wyniki w tabeli. Do wspólnego repo na GitHub/GitLab wrzućcie projekt po zajęciach (bez danych i modelu — pilnuje tego `.gitignore`).
- **Nie programujecie** — kopiujecie gotowe pliki fazy 1 i uruchamiacie polecenia. Wyjątkiem jest ścieżka B, ale jej adaptacja jest zadaniem **po** bloku: [`ADAPTACJA_WLASNYCH_DANYCH.md`](ADAPTACJA_WLASNYCH_DANYCH.md).
- **Zmienna `COURSE_REPO`** (ścieżka do repozytorium kursu) obowiązuje tylko w terminalu, w którym ją ustawiliście. W nowej karcie terminala ustawcie ją ponownie: `COURSE_REPO=~/mlops/materials-about-MLOps`.

## Wybór danych — ograniczcie ryzyko

- **Ścieżka A (rekomendowana):** syntetyczny churn ze `labs-PL/starter-kit/mlops-project/scripts/generate_data.py`. Działa lokalnie, bez pobierania danych i bez danych osobowych; używajcie go, jeśli chcecie skupić się na MLOps, a nie na walce ze schematem.
- **Ścieżka B (tylko gotowe dane):** własny zatwierdzony CSV/Parquet z jednoznacznym targetem i znanym kluczem. Wstępną listą kandydatów do późniejszego przygotowania mogą być UCI Bank Marketing (target `y`), UCI Adult/Census Income albo Telco Customer Churn z Kaggle. Przed użyciem sprawdźcie licencję/warunki dostępu; Kaggle może wymagać konta. Nie pobierajcie danych wrażliwych na zajęcia.
- **Reguła 10 minut:** jeśli własne dane nie dają się otworzyć albo target/zgoda są niejasne, przełączcie się na ścieżkę A i wpiszcie ryzyko jako następne zadanie.

## Checklista na 3 godziny

| Minuty | Zadanie / checklista | Kod / komenda — kiedy i jak użyć | Oczekiwany wynik |
|---:|---|---|---|
| 0–10 | Ustalcie role: właściciel pipeline'u, osoba od danych i osoba notująca decyzje. Utwórzcie kopię projektu i skopiujcie kartę. | Właściciel pipeline'u wykonuje blok **„Krok 0 — nowa kopia projektu”** spod tabeli. | `~/mlops/projekt-ZESPOL` istnieje, ma zainicjowane Git i DVC; są w nim `KARTA_PROJEKTU.md` i `scripts/inspect_dataset.py`. |
| 10–25 | Wybierzcie A lub B, nazwijcie problem i użytkownika oraz jedną decyzję, którą ma wspierać model. | Jeśli nie ma gotowego, zatwierdzonego pliku i targetu — wybierz A. Kandydatów B sprawdzajcie tylko po potwierdzeniu licencji i dostępności. | Jednozdaniowy problem, użytkownik i out-of-scope są zapisane. |
| 25–45 | Wypełnijcie pola karty: źródło/zgoda, jednostka obserwacji, klucz, target, typ zadania, metryka i ryzyko. | Dla B uruchom `python scripts/inspect_dataset.py --input PATH.csv --target TARGET --id ID`; dla Parquet podaj ścieżkę `.parquet`. Użyj raportu do sprawdzenia rozmiaru, braków, duplikatów klucza i rozkładu targetu. | Karta ma wskazane kolumny oraz wstępne ryzyka; jeśli target lub uprawnienia są niejasne, wybierz A. |
| 45–60 | Zdecydujcie o ścieżce i splitcie; dla klasyfikacji zanotujcie niezbalansowanie targetu. W ścieżce A wygenerujcie dane. | A: `python scripts/generate_data.py`; `python scripts/inspect_dataset.py --input data/raw/customers.parquet --target churn --id customer_id`. B: nie podmieniajcie jeszcze kolumn startera — zapiszcie mapowanie i zakres adaptacji. | A: lokalny zbiór 10 000 rekordów; raport pokazuje rozkład klas i brak powielonych `customer_id`. B: istnieje raport inspekcji i jawna decyzja, czy dane nadają się do dalszej pracy. |
| 60–75 | Sfinalizujcie kartę projektu v1 i skopiujcie pozostałe pliki fazy 1 do kopii projektu. | Blok **„Krok 1 — pliki fazy 1”** spod tabeli (w `~/mlops/projekt-ZESPOL`). Otwórzcie skopiowany `dvc.yaml` i znajdźcie nowe etapy `validate` i `evaluate`. | Repo zawiera uzgodnioną kartę projektu, plik pipeline'u fazy 1, skrypty `evaluate_model.py`, `check_phase1.py` i test jakości modelu. |
| 75–90 | Dodajcie dane raw do DVC. | `dvc add data/raw/customers.parquet` (jeśli dane nie istnieją: najpierw `python scripts/generate_data.py`). | Powstaje `data/raw/customers.parquet.dvc`; dane są ignorowane przez Git. |
| 90–115 | Odtwórzcie ścieżkę fazy 1. | `dvc repro` — wykonuje preprocess → validate → train → evaluate; rozwiązujcie pierwszy błąd, nie uruchamiajcie kolejnych testów na nieaktualnych artefaktach. Potem `dvc dag`. | `Running stage` dla czterech etapów; DAG zawiera `preprocess`, `validate`, `train` i `evaluate`. Powstają `reports/validation.json` i `reports/evaluation_metrics.json`. |
| 115–130 | Uruchomcie testy. | `pytest tests/unit tests/data -q`; następnie `pytest tests/model -q`. | Testy przechodzą (`passed`, bez `failed`); test modelu sprawdza próg jakości na odłożonym zbiorze. |
| 130–145 | Uruchomcie kontrolę kompletności, odczytajcie kryterium jakości i zróbcie commit. | `python scripts/check_phase1.py`; `cat reports/evaluation_metrics.json`; potem `git add .` i `git commit -m "Faza 1: baseline"`. | `PASS`; test ROC-AUC ok. 0,79 (próg 0,70). W commicie są kod, `*.dvc`, `dvc.lock`, karta i `reports/metrics/*.json` — bez danych i modelu (`reports/validation.json` i `evaluation_metrics.json` są wyjściami DVC, ich hash jest w `dvc.lock`). |
| 145–160 | Jeśli baseline jest gotowy, uruchomcie dwa szybkie runy MLflow; w przeciwnym razie wykorzystajcie czas na naprawę DVC/testów. | Opcjonalnie: lokalny serwer jak w `labs-PL/starter-kit/mlops-project/README.md`, potem `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000` i `python -m scripts.run_experiments --quick`. Nie jest to warunek ukończenia baseline'u. | Gdy wykonano: dwa runy są widoczne w UI; zapisana jest metryka porównania. |
| 160–175 | Zapiszcie ograniczenia, wynik baseline'u, najbliższe zadanie i jego właściciela. | Aktualizujcie `KARTA_PROJEKTU.md` oraz README projektu; Git ma zawierać kod, konfigurację i wskaźniki, nie surowe dane/model. | Każdy członek zespołu zna następny krok i ryzyko. |
| 175–180 | Sprawdźcie kryteria gotowości i zatrzymajcie lokalny serwer MLflow, jeśli był uruchomiony. | `Ctrl+C` w terminalu serwera. | Karta i wynik są zapisane, serwer zatrzymany. |

**Uwaga do wiersza „10–25”:** „Wybierzcie A lub B” to decyzja ze ścieżki danych opisanej wyżej w sekcji „Wybór danych — ograniczcie ryzyko” (A — syntetyczny churn, domyślnie; B — własne, gotowe dane). „Nazwijcie problem i użytkownika” oznacza krótkie, jednozdaniowe sformułowania: problem — jaki konkretny problem rozwiązuje projekt (np. „przewidzieć, którzy klienci odejdą w ciągu 30 dni”), użytkownik — kto skorzysta z wyników modelu (np. „zespół utrzymania klienta”). „Jedna decyzja, którą ma wspierać model” to konkretne zastosowanie predykcji w jednej akcji (np. „czy wysłać klientowi ofertę retencyjną”) — model nie ma istnieć „w próżni” bez jasnego celu, co ułatwia później dobór metryki i zakresu (out-of-scope). Te ustalenia zapisujecie w `KARTA_PROJEKTU.md`, zgodnie z „Oczekiwanym wynikiem” w tym wierszu tabeli.

### Krok 0 — nowa kopia projektu (minuty 0–10)

Wykonuje właściciel pipeline'u w terminalu VM, z aktywnym venv. Zamieńcie `projekt-ZESPOL` na nazwę zespołu. Jeśli katalog już istnieje, **nie kopiujcie ponownie** — przejdźcie tylko do niego (`cd`).

```bash
COURSE_REPO=~/mlops/materials-about-MLOps
cp -r "$COURSE_REPO/labs-PL/starter-kit/mlops-project" ~/mlops/projekt-ZESPOL
cd ~/mlops/projekt-ZESPOL
git init
dvc init
cp "$COURSE_REPO/project/faza_1/KARTA_PROJEKTU.md" ./KARTA_PROJEKTU.md
cp "$COURSE_REPO/project/faza_1/scripts/inspect_dataset.py" scripts/
python scripts/check_env.py
```

Oczekiwany wynik: `Initialized empty Git repository…`, brak błędów `cp`, a `check_env.py` kończy się `Środowisko gotowe na zajęcia.` Kartę otwórzcie w edytorze (np. `gedit KARTA_PROJEKTU.md &` albo w IDE) — notujący wpisuje do niej decyzje na bieżąco.

### Krok 1 — pliki fazy 1 (minuty 60–75)

W katalogu `~/mlops/projekt-ZESPOL`. Jeśli to nowa karta terminala, najpierw ponownie ustawcie `COURSE_REPO`.

```bash
COURSE_REPO=~/mlops/materials-about-MLOps
cp "$COURSE_REPO/project/faza_1/dvc.yaml" dvc.yaml
cp "$COURSE_REPO/project/faza_1/scripts/evaluate_model.py" "$COURSE_REPO/project/faza_1/scripts/check_phase1.py" scripts/
mkdir -p tests/model
cp "$COURSE_REPO/project/faza_1/tests/model/test_model_quality.py" tests/model/
ls data/raw/customers.parquet || python scripts/generate_data.py
```

### Krok 2 — baseline (minuty 75–145)

```bash
dvc add data/raw/customers.parquet
dvc repro
pytest tests/unit tests/data -q
pytest tests/model -q
python scripts/check_phase1.py
git add .
git commit -m "Faza 1: baseline"
```

**Gdy coś pójdzie nie tak:** najpierw [ściągawka projektu, sekcja 7](BLOK_1_SCIAGA_projekt.md), potem [ściągawka laboratorium, sekcja 5](../../labs-PL/BLOK_1_SCIAGA.md). Najczęstsze przyczyny w projekcie: praca w złym katalogu (`pwd` musi pokazywać `~/mlops/projekt-ZESPOL`), brak `COURSE_REPO` w nowej karcie terminala (`cp` zgłasza `No such file or directory`), brak danych przed `dvc add`. Jeśli `check_phase1.py` zgłasza brakujący plik, uruchomcie ponownie `dvc repro` i sprawdźcie, czy skopiowaliście wszystkie pliki z kroku 1.

## Adaptacja do własnych danych — ścieżka B

W 3 godziny nie przebudowujcie całego startera. Najpierw wykonajcie inspekcję i przygotujcie małą kopię danych bez informacji wrażliwych. Następnie zapiszcie mapowanie kolumn; zmiany są potrzebne w następujących miejscach:

- `src/data/preprocess.py`: ścieżka/format odczytu, klucz w `drop_duplicates`, reguły czyszczenia i feature engineering, nazwa targetu oraz strategia podziału.
- `src/data/validation.py` i `scripts/validate_data.py`: kolumny wymagane, klucz, dozwolone wartości targetu, reguły domenowe i progi.
- `params.yaml` oraz `src/models/train.py`: nazwy cech; trening obecnie czyta target `churn` na stałe.
- `scripts/evaluate_model.py` i `project/faza_1/dvc.yaml`: target, format/metoda splitu i uzasadniony próg metryki. `evaluate_model.py` przyjmuje `--target`, ale dla innego targetu zmień też komendę etapu `evaluate`.
- Testy `tests/data`, `tests/unit`, `tests/model` i README: zaktualizuj oczekiwany schemat, warunki jakości i instrukcję odtworzenia.

Na tym spotkaniu wystarczy karta mapowania i inspekcja danych. Jeśli nie zdążycie bezpiecznie zmienić kodu i testów, zachowajcie demo churn jako baseline, a adaptację potraktujcie jako zadanie na następny etap. Krok po kroku (z kolejnością zmian i kontrolą po każdym kroku) opisuje [`ADAPTACJA_WLASNYCH_DANYCH.md`](ADAPTACJA_WLASNYCH_DANYCH.md); planujcie na to 2–4 godziny pracy między blokami.

## Kryteria gotowości po bloku

- [ ] `KARTA_PROJEKTU.md` zawiera problem, dane/zgodę, target, split, metrykę, zakres i ryzyko.
- [ ] Repo odtwarza minimum pipeline'u fazy 1 przez `dvc repro`; testy i `check_phase1.py` przechodzą (dla ścieżki A).
- [ ] Wynik baseline'u i ograniczenia są zapisane; dane wrażliwe nie trafiają do Git/DVC remote/logów.
- [ ] Zespół ma właściciela i konkretny następny krok; API, chmura i produkcyjny monitoring pozostają poza zakresem.

## Przed blokiem 2 (praca między spotkaniami)

- [ ] Repo projektu jest wypchnięte na GitHub/GitLab (prywatne lub z dostępem dla prowadzącego); wszyscy członkowie zespołu mogą je sklonować.
- [ ] Każda osoba w zespole odtworzyła baseline na swojej VM: `git clone …`, `python scripts/generate_data.py`, `dvc repro` (dane demo generują się deterministycznie, więc nie potrzebujecie DVC remote).
- [ ] Ścieżka B: adaptacja wg [`ADAPTACJA_WLASNYCH_DANYCH.md`](ADAPTACJA_WLASNYCH_DANYCH.md) albo świadoma decyzja, że zostajecie przy churnie.
- [ ] Karta projektu ma zaktualizowane ryzyka i następny krok z właścicielem.

**Uwaga o metrykach demo:** wynik treningowy może być wyższy od testowego; ROC-AUC testowe ok. 0,79–0,80 jest rozsądnym wynikiem dla przygotowanego syntetycznego przykładu. F1 może być niskie przy niezbalansowanych klasach. Próg 0,70 jest bramką demonstracyjną, nie uniwersalnym kryterium dobrego modelu.