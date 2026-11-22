# Blok 1 — przygotowanie przed zajęciami (VM Ubuntu 24.04)

> **Kolejność czytania materiałów bloku 1:** [`BLOK_1_00_START.md`](BLOK_1_00_START.md).

**Czas:** ok. 20–30 min, jeśli maszyna wirtualna z [Laboratorium 0](00_przygotowanie_maszyny_wirtualnej.md) jest już gotowa (inaczej doliczcie 60–90 min na Lab 0).
**Cel:** na zajęciach od pierwszej minuty uruchamiacie pipeline, zamiast instalować pakiety. Każdy problem z instalacją zgłoszony **przed** zajęciami oszczędza 20–40 minut bloku.

Wszystkie polecenia wykonujecie **w terminalu wewnątrz VM** (Ubuntu 24.04, Python 3.12). Na hoście (Windows/macOS) nic nie instalujecie poza VirtualBoxem.

## Docelowy układ katalogów w VM

```text
~/mlops/
├── materials-about-MLOps/   # repozytorium kursu (tylko do odczytu + wspólny .venv)
│   └── .venv/               # jedno środowisko Pythona dla laboratoriów i projektu
├── mlops-project/           # kopia starter kitu na LABORATORIUM
└── projekt-ZESPOL/          # osobna kopia starter kitu na PROJEKT (własne repo zespołu)
```

Zasady:

- **Nie pracujcie wewnątrz repozytorium kursu** — tylko kopiujecie z niego pliki. Dzięki temu `git pull` w repozytorium kursu nigdy nie zrobi konfliktu z waszą pracą.
- **Laboratorium i projekt to dwie osobne kopie.** Na projekcie skopiujecie do swojej kopii plik `dvc.yaml` z fazy 1; gdyby to była ta sama kopia, nadpisałby on wersję z laboratorium.
- **Nie trzymajcie projektów w folderze współdzielonym VirtualBoxa** (`/media/sf_*`) — jest wolny i źle obsługuje symlinki, co psuje `git` i `dvc`. Folder współdzielony służy tylko do wymiany plików.

## Checklista przygotowania

- [ ] **1. VM działa i ma migawkę.** Jeśli nie macie migawki `01-srodowisko-mlops` z Lab 0, zróbcie ją teraz (VM wyłączona → Migawki → Zrób migawkę). Gdy coś się zepsuje na zajęciach, przywrócenie migawki trwa kilkanaście sekund.
- [ ] **2. Schowek działa w obie strony** (Urządzenia → Współdzielony schowek → Dwukierunkowy). Będziecie kopiować komendy z materiałów. W terminalu Ubuntu wklejanie to `Ctrl+Shift+V`, a kopiowanie `Ctrl+Shift+C` (samo `Ctrl+C` przerywa program).
- [ ] **3. Repozytorium kursu jest aktualne, a venv zainstalowany:**

  ```bash
  cd ~/mlops/materials-about-MLOps
  git pull
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  ```

  Oczekiwany wynik: ostatnia linia instalacji nie zawiera `ERROR`; na początku linii poleceń widać `(.venv)`.
- [ ] **4. Kopia starter kitu na laboratorium:**

  ```bash
  cp -r ~/mlops/materials-about-MLOps/labs-PL/starter-kit/mlops-project ~/mlops/mlops-project
  cd ~/mlops/mlops-project
  ```

  Jeśli katalog `~/mlops/mlops-project` już istnieje, **nie kopiujcie ponownie** (powstałby zagnieżdżony katalog `mlops-project/mlops-project`).
- [ ] **5. Kontrola środowiska** (z katalogu `~/mlops/mlops-project`, z aktywnym venv):

  ```bash
  python scripts/check_env.py
  ```

  Oczekiwany wynik: same `[OK]` (ewentualnie `[UWAGA]`) i na końcu `Środowisko gotowe na zajęcia.` Każde `[BŁĄD]` ma pod spodem podpowiedź `->` z poleceniem naprawy.
- [ ] **6. Test dymny — dane generują się poprawnie:**

  ```bash
  python scripts/generate_data.py
  ```

  Oczekiwany wynik: komunikat o zapisaniu `data/raw/customers.parquet` (10 000 wierszy). Nie róbcie jeszcze `git init` ani `dvc init` — to część zajęć.
- [ ] **7. (Opcjonalnie) wygodne otwieranie terminala z venv.** Dopiszcie aktywację do `~/.bashrc`:

  ```bash
  echo 'source ~/mlops/materials-about-MLOps/.venv/bin/activate' >> ~/.bashrc
  ```

  Na zajęciach używacie **dwóch terminali** (serwer MLflow + praca); dzięki temu oba będą miały aktywny venv.
- [ ] **8. Zespół i pomysł.** Ustalcie skład zespołu projektowego (2–3 osoby) i nazwę katalogu `projekt-ZESPOL`. Przeczytajcie [`../project/faza_1/BLOK_1_projekt.md`](../project/faza_1/BLOK_1_projekt.md) — sekcję „Wybór danych”. Jeśli myślicie o własnych danych (ścieżka B), sprawdźcie wcześniej, że macie plik CSV/Parquet i prawo do jego użycia.

## Gdy coś nie działa przed zajęciami

| Objaw | Co zrobić |
|---|---|
| `error: externally-managed-environment` | Nie macie aktywnego venv: `source ~/mlops/materials-about-MLOps/.venv/bin/activate`. Nigdy nie używajcie `--break-system-packages`. |
| `[BŁĄD] sqlalchemy 2.1.x — wymagane <2.1` | `python -m pip install 'sqlalchemy<2.1'` (MLflow 2.x nie działa z SQLAlchemy 2.1). |
| `dvc: command not found` | venv nieaktywny albo instalacja przerwana: aktywujcie venv i powtórzcie `pip install -r requirements.txt`. |
| `[UWAGA] Git nie ma ustawionego: user.name` | `git config --global user.name "Imię Nazwisko"` i `git config --global user.email "adres@uczelnia.pl"`. |
| `[UWAGA] Zmienna SHELL wskazuje nieistniejący program` | `export SHELL=/bin/bash` (i dopiszcie tę linię do `~/.bashrc`). |
| Instalacja pakietów trwa bardzo długo lub się zawiesza | Sprawdźcie internet w VM (`ping -c 3 pypi.org`) i zegar (`sudo timedatectl set-ntp true`). |
| Za mało miejsca / pamięci | Na blok 1 wystarczą 4 GB RAM i ok. 5 GB wolnego dysku (`df -h ~`). |

Jeśli problem zostaje, wyślijcie prowadzącemu **przed zajęciami** pełny wynik `python scripts/check_env.py` (skopiowany tekst, nie zrzut ekranu z telefonu) oraz ostatnie ok. 20 linii komunikatu błędu.

## Co dalej

- Na laboratorium: [`BLOK_1_laboratorium.md`](BLOK_1_laboratorium.md).
- Na projekt: [`../project/faza_1/BLOK_1_projekt.md`](../project/faza_1/BLOK_1_projekt.md).
- Ściągawka z komendami, słowniczkiem i typowymi błędami: [`BLOK_1_SCIAGA.md`](BLOK_1_SCIAGA.md).
