# Checklista projektu — faza 1: problem, dane i baseline

> **Układ 3 × (3 h laboratorium + 3 h projektu):** na pierwszy blok obowiązuje skondensowana checklista [`BLOK_1_projekt.md`](BLOK_1_projekt.md) (180 min) i ściągawka [`BLOK_1_SCIAGA_projekt.md`](BLOK_1_SCIAGA_projekt.md). Poniższy kickoff 120 min jest materiałem szczegółowym i referencyjnym, a nie harmonogramem pierwszego bloku.

**Cel:** w ograniczonym czasie uzgodnić mały, wykonalny problem i uruchomić powtarzalny baseline z kontrolą danych. Faza 1 **nie** wymaga jeszcze API, kontenera, chmury, orkiestracji ani monitoringu produkcyjnego.

## Przed kickoffem (praca własna, 20–30 min)

- [ ] Każdy członek zespołu przygotowuje 1 propozycję problemu i źródło danych z licencją / zgodą.
- [ ] Skopiować starter kit z `labs-PL/starter-kit/mlops-project/` do katalogu repozytorium projektu.
- [ ] Uruchomić środowisko, wygenerować dane demo i przejść `dvc repro` na starter kicie (dla własnego datasetu — przygotować próbkę i mapowanie kolumn).
- [ ] Wypełnić wstępnie `KARTA_PROJEKTU.md`; decyzje wymagające danych zostawić jako otwarte, nie zgadywać metryki.
- [ ] Podzielić role na kickoff: osoba prowadząca repo/pipeline, osoba danych, osoba notująca decyzje; role można zmieniać.

## Kickoff — 120 min

| Minuty | Checklista / decyzja | Rezultat |
|---:|---|---|
| 0–10 | **Minimum na zajęciach:** przypomnieć cel fazy 1 i ograniczenia zakresu. Uzgodnić, że pierwszym celem jest odtwarzalny baseline, nie produkcyjna platforma. | Wszyscy rozumieją kryterium ukończenia. |
| 10–25 | Zebrać propozycje, porównać dostępność danych, rozmiar zadania, wartość edukacyjną i ryzyko prawne / prywatności. | Lista 1–2 kandydatów z danymi. |
| 25–40 | **Minimum na zajęciach:** wybrać jeden problem, użytkownika i decyzję wspieraną predykcją; zapisać także zakres, którego na razie nie realizujemy. | Jednozdaniowy opis problemu i jawny out-of-scope. |
| 40–55 | Ustalić obserwację, jednostkę, klucz, target, typ zadania, źródło i licencję danych. Dla danych zewnętrznych sprawdzić możliwość pobrania małej próbki. | Uzupełnione pola „Problem” i „Dane” w karcie. |
| 55–70 | **Minimum na zajęciach:** sprawdzić brakujące wartości, duplikaty, target i możliwe przecieki. Zdecydować, jaki jest uczciwy train/test split (czasowy, grupowy lub stratyfikowany). | Krótka notatka o ryzykach danych i podziale. |
| 70–85 | Uzgodnić metrykę główną, pomocniczą i uzasadniony próg jakości. Dla klasyfikacji uwzględnić niezbalansowanie klas; nie wybierać progu „bo baseline przeszedł”. | Kryterium sukcesu zapisane w karcie. |
| 85–100 | **Minimum na zajęciach:** zmapować dane do `customer_id`, cech i targetu lub zdecydować o małej zmianie startera; wskazać moduły preprocess/train do adaptacji. | Plan adaptacji kodu, bez projektowania całej platformy. |
| 100–110 | Wybrać model bazowy i minimalny zakres jakości danych; wskazać jak uruchomić walidację, `dvc repro` i testy. | Prosty eksperyment bazowy i warunek powtarzalności. |
| 110–120 | Uzupełnić kartę projektu, przydzielić najbliższe zadania i ustalić datę kontroli. Każdy członek zespołu wskazuje ryzyko oraz właściciela. | Karta projektu v1, zadania i właściciele. |

**Jeśli brakuje czasu:** zamknąć wybór problemu, dostępność/licencję danych, target, split, metrykę oraz właścicieli; zostawić alternatywne architektury na później. Nie rozpoczynać implementacji API.

## Minimalny rezultat fazy 1 (po kickoffie i krótkiej pracy własnej)

- [ ] `KARTA_PROJEKTU.md` jest uzupełniona i uzgodniona przez zespół.
- [ ] Mała, legalna próbka danych jest dostępna i opisana; dane wrażliwe nie są commitowane.
- [ ] Repozytorium zawiera kod, `params.yaml`, `dvc.yaml` i wskaźniki DVC; dane są śledzone przez DVC.
- [ ] Pipeline wykonuje preprocess → validate → train → evaluate.
- [ ] Jest baseline i raport z accuracy, F1, ROC-AUC (lub właściwymi metrykami zadania).
- [ ] `pytest tests/unit tests/data -q`, `pytest tests/model -q` oraz `python scripts/check_phase1.py` przechodzą.
- [ ] README projektu mówi, jak odtworzyć środowisko i wynik; zapisane są ograniczenia i następny krok.

## Komendy dla demonstracyjnego projektu churn

Z katalogu głównego kopii starter kitu (wymaga skopiowania plików fazy 1 do projektu):

```bash
COURSE_REPO=/ścieżka/do/materials-about-MLOps
python scripts/generate_data.py
dvc add data/raw/customers.parquet
cp "$COURSE_REPO/project/faza_1/dvc.yaml" dvc.yaml
cp "$COURSE_REPO/project/faza_1/scripts/evaluate_model.py" "$COURSE_REPO/project/faza_1/scripts/check_phase1.py" scripts/
mkdir -p tests/model
cp "$COURSE_REPO/project/faza_1/tests/model/test_model_quality.py" tests/model/
dvc repro
pytest tests/unit tests/data -q
pytest tests/model -q
python scripts/check_phase1.py
```

Przy projekcie z własnym repozytorium najlepiej skopiować `scripts/evaluate_model.py`, `scripts/check_phase1.py`, `tests/model/test_model_quality.py` oraz `dvc.yaml` z `project/faza_1/` do własnej struktury, a `params.yaml`, preprocess i walidację dostosować do swoich kolumn i metryki. Dla przykładowego zbioru churn próg ROC-AUC jest ustawiony w etapie `evaluate` w `dvc.yaml`; dla innego problemu zmień metrykę i uzasadniony próg **przed** oceną wyniku.

## Poza zakresem fazy 1

- Nie trenuj wielkich modeli ani nie stroisz dziesiątek konfiguracji.
- Nie buduj API, Kubernetes/Kubeflow, CI/CD ani monitoringu, zanim dane i baseline nie są reprodukowalne.
- Nie umieszczaj sekretów ani danych osobowych w Git, DVC remote lub logach.