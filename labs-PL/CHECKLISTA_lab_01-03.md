# Checklista zajęć: laboratoria 1–3

Plan zakłada **120 minut na laboratorium** i gotowy starter kit. Studenci pracują w parach, a nie przepisują kodu. Każdy punkt oznaczony **minimum na zajęciach** daje działający, sprawdzalny rezultat; rozszerzenia są pracą domową lub dodatkiem dla szybszych zespołów.

## Przed pierwszymi zajęciami (prowadzący / studenci, poza czasem laboratorium)

- [ ] Pobrać repozytorium kursu i skopiować `labs-PL/starter-kit/mlops-project/` do własnego katalogu roboczego / repozytorium.
- [ ] Mieć Python 3.10+ (zalecany 3.12), Git i możliwość utworzenia venv; zainstalować `requirements.txt`.
- [ ] Uruchomić `python scripts/generate_data.py`; upewnić się, że plik `data/raw/customers.parquet` istnieje.
- [ ] Na początku laboratorium mieć dostęp do README i checklisty; jeśli instalacja nie zdążyła się zakończyć, użyć przygotowanego przez prowadzącego środowiska.
- [ ] Prowadzący przygotowuje awaryjne archiwum starter kitu oraz kopię danych, ale studenci nadal samodzielnie wykonują `dvc add` i `dvc repro`.

---

## Laboratorium 1 — środowisko, Git, DVC i pierwszy pipeline (120 min)

**Cel minimum:** z wersjonowanego kodu i wskaźnika DVC odtworzyć podział danych i model; rozumieć, które pliki należą do Git, a które do DVC.

| Minuty | Zadanie / checklista | Dowód ukończenia |
|---:|---|---|
| 0–10 | **Minimum na zajęciach:** w parach sprawdzić katalog główny, aktywować `.venv`, `python --version`, `git --version`, `dvc --version`. Uzgodnić kto wykonuje polecenia, a kto zapisuje decyzje. | Narzędzia działają; terminal znajduje się w projekcie. |
| 10–20 | Sprawdzić `README.md`, `params.yaml` i `dvc.yaml`; wskazać wejście raw, etapy preprocess/train oraz oczekiwane wyjścia. | Każdy potrafi nazwać jeden artefakt danych i jeden modelowy. |
| 20–35 | **Minimum na zajęciach:** wygenerować dane (`python scripts/generate_data.py`), sprawdzić liczbę wierszy i kolumn w Parquet; wykonać `git init` i `dvc init`. | `data/raw/customers.parquet` istnieje; repozytorium i katalog `.dvc/` utworzone. |
| 35–50 | Wykonać `dvc add data/raw/customers.parquet`; odczytać utworzony plik `data/raw/customers.parquet.dvc` i wpis w `.gitignore`. Dodać do Git wyłącznie kod, konfigurację, wskaźniki i ignore rules. | Student umie wyjaśnić: Git trzyma wskaźnik, DVC — zawartość w cache. |
| 50–70 | **Minimum na zajęciach:** wykonać `dvc repro`; sprawdzić `data/processed/train.parquet`, `test.parquet`, `models/churn_model.pkl`, `reports/metrics/data_stats.json` i `reports/metrics/train_metrics.json`. | Pipeline kończy się powodzeniem; widać oba splity, model i metryki. |
| 70–85 | Wykonać `dvc dag`, `dvc status`, `dvc metrics show`; opisać zależności i metryki. | Zrzut / notatka z DAG oraz metryką ROC-AUC. |
| 85–100 | Zmienić jeden parametr treningu w `params.yaml` (np. `model.n_estimators`), ponowić `dvc repro`, porównać metryki; odróżnić zmianę parametru od zmiany kodu. | Nowy wynik jest policzony, a `dvc status` pokazuje co wymaga odtworzenia. |
| 100–112 | **Minimum na zajęciach:** uruchomić `pytest tests/unit -q`; sprawdzić testy i `git status`. Wybrać pliki, które powinny wejść do commita. | Testy przechodzą; dane i model nie trafiają bezpośrednio do Git. |
| 112–120 | Podsumowanie w parach: co jest wejściem/wyjściem? Co należy zatwierdzić w Git? Co zrobi ponownie `dvc repro`? | Każda para zapisuje 3 odpowiedzi i jeden wniosek. |

**Jeśli brakuje czasu:** ukończyć wygenerowanie/wersjonowanie raw, `dvc repro`, sprawdzenie metryk i testów; krok zmiany parametru dokończyć w domu. **Nie pomijać:** odtworzenia pipeline'u i sprawdzenia, co Git śledzi.

**W domu (15–30 min):** wykonać logiczny commit kodu i wskaźników DVC; zmienić `data.test_size` lub `data.seed`, uruchomić `dvc repro`, porównać `dvc metrics show`; dodać do README 2–3 zdania o DAG i reprodukowalności.

---

## Laboratorium 2 — MLflow Tracking i porównanie runów (120 min)

**Warunek wejścia:** Lab 1 działa; istnieją pliki `data/processed/train.parquet` i `test.parquet`.

| Minuty | Zadanie / checklista | Dowód ukończenia |
|---:|---|---|
| 0–10 | **Minimum na zajęciach:** sprawdzić dane i parametry modelu; omówić run vs experiment, parametry vs metryki vs artefakty. | Para zna 3 elementy, które chce porównywać. |
| 10–25 | Uruchomić lokalny serwer MLflow zgodnie z README w osobnym terminalu; sprawdzić `/health` i otworzyć UI. Użyć `127.0.0.1`, nie wystawiać serwera do sieci. | UI odpowiada na `http://127.0.0.1:5000`. |
| 25–35 | W drugim terminalu aktywować venv i ustawić `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; sprawdzić uruchomienie eksperymentu `--quick`. | Powstaje eksperyment i dwa runy. |
| 35–55 | **Minimum na zajęciach:** wykonać `python -m scripts.run_experiments --quick`; odczytać identyfikatory runów, parametry, metryki i artefakt modelu w UI. | Dwa runy z różnymi parametrami i metryką ROC-AUC są widoczne. |
| 55–70 | Wykonać `python -m scripts.compare_experiments`; wskazać run z najlepszym ROC-AUC, obejrzeć również F1 i accuracy. | Ranking tekstowy zgadza się z UI. |
| 70–90 | Uruchomić pełną siatkę tylko gdy pozostaje czas; sprawdzić wpływ `n_estimators` i `max_depth`, bez strojenia na danych testowych jako metody wyboru. | Para zapisuje porównanie dwóch konfiguracji i ograniczenie użytej metryki. |
| 90–105 | **Minimum na zajęciach:** zarejestrować jeden run: `python -m scripts.registry_aliases RUN_ID --name churn-random-forest --alias candidate`; obejrzeć wpis w Model Registry. | Istnieje wersja modelu i alias `candidate`. |
| 105–115 | Otworzyć metryki, tagi i artefakt w UI; zapisać run ID, parametry, URI trackingu i metrykę zwycięzcy. | Notatka pozwala wskazać konkretny eksperyment. |
| 115–120 | Podsumowanie: co pozwala odtworzyć run? Dlaczego nie wybieramy modelu przez wielokrotne podglądanie testu? | Każda para odpowiada na oba pytania. |

**Jeśli brakuje czasu:** wykonać tylko `--quick`, ranking i rejestrację jednego runu; pominięta pełna siatka nie blokuje zaliczenia. **W domu:** krótki porównawczy opis dwóch runów; serwer zatrzymać `Ctrl+C`.

---

## Laboratorium 3 — walidacja, anomalia i prosty feature store (120 min)

**Warunek wejścia:** Lab 1 ukończony. Lab 2 jest przydatny, lecz uruchomienie serwera MLflow nie jest wymagane do ćwiczeń walidacji.

| Minuty | Zadanie / checklista | Dowód ukończenia |
|---:|---|---|
| 0–10 | **Minimum na zajęciach:** wczytać podział danych, nazwać błąd schematu, błąd zakresu i anomalię rozkładu. Uzgodnić różnicę między error a warning. | Para podaje przykład każdej kategorii. |
| 10–30 | **Minimum na zajęciach:** uruchomić `pytest tests/data -q` i `python -m scripts.validate_data`; sprawdzić reguły kolumn, braków, unikalnego ID, zakresu wieku i minimalnej liczby wierszy. | Testy przechodzą, raport JSON zawiera wynik dla każdej reguły. |
| 30–45 | Otworzyć `src/data/validation.py`; dodać (lub opisać) jedną regułę dziedzinową; wykonać test negatywny na kopii danych, nie psuć właściwego pliku raw. | Test wykazuje niepoprawne dane; raport rozróżnia severity. |
| 45–60 | **Minimum na zajęciach:** scalić etap z `dvc_lab03_validate.yaml` do `dvc.yaml` i dodać raport walidacji jako zależność etapu `train`; wykonać `dvc repro`. | Walidacja znajduje się przed treningiem i błąd zatrzyma pipeline. |
| 60–75 | Omówić `IQRAnomalyDetector`: dopasować go wyłącznie do train; wykryć wstrzyknięty skrajny punkt testowy / kopię danych. | Para wyjaśnia, dlaczego nie wolno dopasować progu na teście. |
| 75–95 | **Minimum na zajęciach:** utworzyć `FileFeatureStore`, zapisać grupę cech train do Parquet, przeczytać rekordy po `customer_id`; obejrzeć `registry.json`. | Grupa feature jest zapisana, zarejestrowana i odczytywalna po kluczu. |
| 95–105 | Uruchomić `python scripts/validate_gx.py` (opcjonalnie, jeśli Great Expectations zainstalowane); porównać oczekiwanie GX z własną regułą. | Widać wynik jednej expectation; brak zależności od GX w głównym pipeline. |
| 105–115 | **Minimum na zajęciach:** uruchomić `python scripts/check_progress.py`; wskazać brakujące artefakty i zaplanować ich uzupełnienie. | Checklista podaje status Labs 1–3. |
| 115–120 | Zapis refleksji: która kontrola blokuje pipeline, a która może tylko ostrzegać? Co należy monitorować w przyszłości? | Jedno zdanie do raportu. |

**Jeśli brakuje czasu:** priorytet to testy i raport walidacji, integracja DVC oraz jeden zapis/odczyt feature store; GX i dodatkowa reguła są pracą domową. **W domu (20–40 min):** dodać test negatywny wybranej reguły i opisać ograniczenia IQR (progi statystyczne nie są regułami biznesowymi).

---

## Szybkie kryteria zaliczenia minimum

- [ ] **Lab 1:** `dvc repro` kończy się poprawnie; w Git są pliki kontrolne, nie surowe dane/model; testy jednostkowe przechodzą.
- [ ] **Lab 2:** dwa runy w MLflow, metryka/parametry widoczne, jeden model zarejestrowany.
- [ ] **Lab 3:** walidacja przed treningiem, raport i test negatywny; feature store zapisuje i odczytuje feature group.