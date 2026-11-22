# Blok 1 — od czego zacząć i w jakiej kolejności czytać

Blok 1 to jedno spotkanie: **3 h laboratorium, potem 3 h projektu**. Materiały są w kilku plikach `BLOK_1_…`. Czytajcie je w kolejności z tabeli poniżej. Nie trzeba czytać wszystkiego od razu: każdy plik ma swój moment.

## Kolejność

| # | Kiedy | Plik | Po co | Jak czytać |
|---|---|---|---|---|
| 0 | **Przed zajęciami** (jeśli VM nie jest gotowa) | [`00_przygotowanie_maszyny_wirtualnej.md`](00_przygotowanie_maszyny_wirtualnej.md) | Instalacja VM Ubuntu 24.04 w VirtualBoxie (Lab 0) | Krok po kroku, ok. 60–90 min. Pomińcie, jeśli VM już działa. |
| 1 | **Przed zajęciami** | [`BLOK_1_PRZED_ZAJECIAMI.md`](BLOK_1_PRZED_ZAJECIAMI.md) | Układ katalogów, venv, `check_env.py` | Wykonajcie całą checklistę (20–30 min). Problemy zgłoście prowadzącemu **przed** zajęciami. |
| 2 | **Laboratorium, minuta 0** | [`BLOK_1_laboratorium.md`](BLOK_1_laboratorium.md) | Plan 180 min: co robić i jaki wynik ma wyjść | Główny plik laboratorium. Idźcie wiersz po wierszu i odhaczajcie. |
| 3 | **Laboratorium, cały czas** | [`BLOK_1_SCIAGA.md`](BLOK_1_SCIAGA.md) | Komendy z wynikami, słowniczek, typowe błędy | Trzymajcie otwartą obok terminala. Zaglądajcie, gdy coś jest niejasne albo nie działa. |
| 4 | **Projekt, minuta 0** | [`../project/faza_1/BLOK_1_projekt.md`](../project/faza_1/BLOK_1_projekt.md) | Plan 180 min projektu: karta, baseline, podsumowanie | Główny plik projektu. Idźcie krok po kroku i odhaczajcie. |
| 5 | **Projekt, cały czas** | [`../project/faza_1/BLOK_1_SCIAGA_projekt.md`](../project/faza_1/BLOK_1_SCIAGA_projekt.md) | Role, wzór karty, mapa plików, wyniki, błędy, podsumowanie dla prowadzącego | Trzymajcie otwartą obok terminala. Sekcję 6 przeczytajcie ok. 15 min przed końcem. |
| 6 | **Po bloku, tylko ścieżka B** | [`../project/faza_1/ADAPTACJA_WLASNYCH_DANYCH.md`](../project/faza_1/ADAPTACJA_WLASNYCH_DANYCH.md) | Przeniesienie pipeline'u na własne dane | Praca między blokami (2–4 h). **Nie** na zajęciach. |

W skrócie: **przygotowanie → laboratorium + ściąga → projekt + ściąga → (ścieżka B) adaptacja**.

## Zasada „plan + ściąga”

W obu częściach bloku macie otwarte **dwa** pliki:

- **plan** (`BLOK_1_laboratorium.md` / `BLOK_1_projekt.md`) mówi, **co** robić i w której minucie;
- **ściąga** (`BLOK_1_SCIAGA.md` / `BLOK_1_SCIAGA_projekt.md`) mówi, **jak** to zrobić, jaki wynik jest poprawny i co zrobić przy błędzie.

Jeśli plan i ściąga wydają się różnić, obowiązuje **plan**. Zgłoście to prowadzącemu.

## Pliki pomocnicze (nie czytajcie ich od początku do końca)

| Plik | Kiedy sięgnąć |
|---|---|
| [`lab_01_srodowisko_i_git.md`](lab_01_srodowisko_i_git.md), [`lab_02_mlflow_tracking.md`](lab_02_mlflow_tracking.md) | Gdy plan laboratorium odsyła do konkretnego kroku albo chcecie poznać szczegóły. |
| [`CHECKLISTA_lab_01-03.md`](CHECKLISTA_lab_01-03.md), [`../project/faza_1/CHECKLISTA_faza_1.md`](../project/faza_1/CHECKLISTA_faza_1.md) | Wersje referencyjne z innym podziałem czasu. Na blok 1 obowiązują pliki `BLOK_1_…`. |
| [`../project/faza_1/KARTA_PROJEKTU.md`](../project/faza_1/KARTA_PROJEKTU.md) | Szablon kopiowany do repo zespołu w projekcie (krok opisany w `BLOK_1_projekt.md`). Wypełniacie go, nie czytacie wcześniej. |
| [`starter-kit/mlops-project/README.md`](starter-kit/mlops-project/README.md) | Opis struktury startera, gdy chcecie wiedzieć, co robi dany plik. |

## Minimalna wersja (gdy macie tylko 10 minut przed zajęciami)

1. Wykonajcie `BLOK_1_PRZED_ZAJECIAMI.md` do końca: ma przejść `python scripts/check_env.py`.
2. Na zajęciach otwórzcie `BLOK_1_laboratorium.md` i `BLOK_1_SCIAGA.md`.
3. Po przerwie otwórzcie `BLOK_1_projekt.md` i `BLOK_1_SCIAGA_projekt.md`.
