# Wykład 2: Inżynieria Danych dla ML

## Cel wykładu
Po tym wykładzie student:
- rozumie rolę inżynierii danych w systemach ML,
- zna wzorce architektoniczne Data Lake i Data Warehouse,
- potrafi zaprojektować pipeline przetwarzania danych,
- rozumie koncepcję Feature Store i jego znaczenie.

## Skrypty do wykładu

Skrypty demonstracyjne znajdują się w `lectures/scripts/lecture_02/`. Pełną instrukcję znajdziesz w [`lectures/scripts/README.md`](scripts/README.md).

| Kiedy (sekcja) | Skrypt | Cel demonstracji |
|---|---|---|
| 3. Formaty danych | `01_csv_vs_parquet.py` | rozmiar, szybkość, column pruning i typy danych |
| 5. Pipeline przetwarzania | `02_data_pipeline.py` | kompletny pipeline pandas + sklearn zapisany jako jeden artefakt |
| 6. Walidacja danych | `03_data_validation.py [--corrupt]` | walidacja jako bramka jakości (kod wyjścia 1 przy błędach) |
| 8. DVC | `05_dvc_demo.sh` | wersjonowanie danych i powrót do poprzedniej wersji |
| 10. Pułapki | `04_data_leakage.py` | data leakage daje złudnie wysokie wyniki |

**Jak:** z katalogu głównego repozytorium uruchom `python lectures/scripts/lecture_02/<skrypt>`. Skrypt DVC uruchamia się przez `bash` i wymaga `git` oraz `dvc`. Działa w katalogu tymczasowym, więc nie zmienia repozytorium.

---

## 1. Dane jako fundament ML

> „Garbage in, garbage out" – jakość modelu jest ograniczona jakością danych.

W systemach produkcyjnych dane są najważniejszym zasobem. Inżynieria danych (Data Engineering) zajmuje się:
- **zbieraniem** danych z różnych źródeł,
- **przechowywaniem** w odpowiednich formatach i strukturach,
- **przetwarzaniem** (czyszczenie, transformacja, agregacja),
- **udostępnianiem** dla modeli ML.

```mermaid
flowchart LR
    subgraph Źródła
        S1[Bazy danych\nOLTP]
        S2[Logi\naplikacji]
        S3[API\nzewnętrzne]
        S4[Pliki\nCSV/JSON]
        S5[Strumienie\nKafka]
    end

    subgraph Przetwarzanie
        ETL[ETL / ELT\nPipeline]
    end

    subgraph Przechowywanie
        DL[Data Lake\nsurowe dane]
        DW[Data Warehouse\nprzetworzone]
        FS[Feature Store\ncechy ML]
    end

    S1 & S2 & S3 & S4 & S5 --> ETL
    ETL --> DL --> DW --> FS
```

---

## 2. Architektura danych: Data Lake vs Data Warehouse

### Data Lake
- Przechowuje **surowe dane** w oryginalnym formacie (pliki, JSON, Parquet, Avro).
- Schemat definiowany przy odczycie (*schema-on-read*).
- Tanie przechowywanie dużych wolumenów.
- Przykłady: AWS S3, Google Cloud Storage, Azure Data Lake.

### Data Warehouse
- Przechowuje **przetworzone, ustrukturyzowane dane**.
- Schemat definiowany przy zapisie (*schema-on-write*).
- Zoptymalizowany pod kątem zapytań analitycznych (OLAP).
- Przykłady: BigQuery, Snowflake, Redshift.

### Data Lakehouse
Nowoczesne podejście łączące zalety obu:

```mermaid
graph TB
    DL[Data Lake\nsurow dane, tanie] --> DLH[Data Lakehouse\nDelta Lake / Iceberg]
    DW[Data Warehouse\nstruktura, drogie] --> DLH
    DLH --> ML[ML Training]
    DLH --> BI[BI / Analytics]
    DLH --> RT[Real-time\nQueries]
```


### Architektura medalionowa (Bronze / Silver / Gold)

Popularny wzorzec organizacji danych w Lakehouse – dane przechodzą przez kolejne warstwy jakości:

```mermaid
flowchart LR
    SRC[Źródła] --> B[(Bronze\nsurowe, niezmienione\nappend-only)]
    B -->|czyszczenie,\ndeduplikacja, typy| S[(Silver\noczyszczone,\nzintegrowane)]
    S -->|agregacje,\ncechy| G[(Gold\ngotowe do użycia:\nraporty, cechy ML)]
    G --> ML[Trening ML]
    G --> BI[Dashboardy]
```

| Warstwa | Zawartość | Kto używa | Zasada |
|---|---|---|---|
| Bronze | kopia danych źródłowych | inżynierowie danych | nigdy nie nadpisujemy – możliwy pełny replay |
| Silver | dane oczyszczone, ujednolicone typy | analitycy, DS | jedna „wersja prawdy” dla encji |
| Gold | agregaty i cechy biznesowe | modele ML, BI | zoptymalizowane pod konkretne użycie |

Zaleta dla ML: gdy zmienimy logikę feature engineeringu, możemy **odtworzyć** warstwy Silver/Gold z Bronze i wytrenować model na historycznych danych z nowymi cechami.

### OLTP vs OLAP

| | OLTP (np. PostgreSQL) | OLAP (np. BigQuery) |
|---|---|---|
| Operacje | wiele małych zapisów/odczytów | duże skany i agregacje |
| Układ danych | wierszowy | kolumnowy |
| Rola w ML | źródło danych, online serving | trening, analityka, feature engineering |

---

## 3. Formaty danych dla ML

### Porównanie formatów

| Format | Typ | Kompresja | Schemat | Użycie |
|--------|-----|-----------|---------|--------|
| CSV | Wierszowy | Brak | Brak | Małe zbiory, prototypy |
| JSON | Wierszowy | Opcjonalna | Brak | API, logi |
| Parquet | Kolumnowy | Snappy/ZSTD | Tak | Analityka, ML |
| Avro | Wierszowy | Deflate | Tak | Kafka, streaming |
| TFRecord | Binarny | Opcjonalna | Tak | TensorFlow |
| HDF5 | Hierarchiczny | Tak | Tak | Duże macierze, DL |

### Dlaczego Parquet dla ML?

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os

# Generowanie przykładowych danych
df = pd.DataFrame({
    'user_id': range(1_000_000),
    'age': pd.Series(range(1_000_000)) % 80 + 18,
    'income': pd.Series(range(1_000_000)) * 0.05 + 20000,
    'churn': pd.Series(range(1_000_000)) % 2
})

# Zapis CSV
t0 = time.time()
df.to_csv('/tmp/data.csv', index=False)
csv_time = time.time() - t0
csv_size = os.path.getsize('/tmp/data.csv') / 1024 / 1024

# Zapis Parquet
t0 = time.time()
df.to_parquet('/tmp/data.parquet', compression='snappy', index=False)
parquet_time = time.time() - t0
parquet_size = os.path.getsize('/tmp/data.parquet') / 1024 / 1024

print(f"CSV:     {csv_size:.1f} MB, zapis: {csv_time:.2f}s")
print(f"Parquet: {parquet_size:.1f} MB, zapis: {parquet_time:.2f}s")
print(f"Oszczędność miejsca: {(1 - parquet_size/csv_size)*100:.0f}%")

# Odczyt tylko wybranych kolumn (column pruning)
t0 = time.time()
df_partial = pd.read_parquet('/tmp/data.parquet', columns=['user_id', 'churn'])
print(f"\nOdczyt 2 kolumn z Parquet: {time.time()-t0:.3f}s")
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_02/01_csv_vs_parquet.py --rows 1000000`  
> **Kiedy:** po omówieniu tabeli porównawczej formatów.  
> **Co pokazać:** Parquet jest kilkukrotnie mniejszy, odczyt dwóch kolumn jest wielokrotnie szybszy, a CSV gubi typy `datetime` i `category`.

---

## 4. ETL vs ELT

### ETL (Extract → Transform → Load)
Tradycyjne podejście: transformacja przed załadowaniem do magazynu.

```mermaid
sequenceDiagram
    participant S as Źródło
    participant E as ETL Engine
    participant DW as Data Warehouse

    S->>E: Extract (surowe dane)
    E->>E: Transform (czyszczenie, agregacja)
    E->>DW: Load (przetworzone dane)
```

### ELT (Extract → Load → Transform)
Nowoczesne podejście: ładowanie surowych danych, transformacja w miejscu.

```mermaid
sequenceDiagram
    participant S as Źródło
    participant DL as Data Lake
    participant DW as Data Warehouse

    S->>DL: Extract + Load (surowe dane)
    DL->>DW: Load (surowe dane)
    DW->>DW: Transform (SQL, dbt)
```

**Zalety ELT:**
- Zachowanie surowych danych (możliwość ponownego przetworzenia).
- Wykorzystanie mocy obliczeniowej magazynu danych.
- Łatwiejsze debugowanie.


### Orkiestracja pipeline'ów: DAG, idempotentność, backfill

Pipeline danych to **skierowany graf acykliczny (DAG)** zadań. Orkiestrator (Airflow, Prefect, Dagster) pilnuje kolejności, ponowień i harmonogramu.

```mermaid
flowchart LR
    E1[extract_db] --> V[validate]
    E2[extract_api] --> V
    V --> T[transform]
    T --> F[build_features]
    F --> TR[train_model]
    F --> RP[report]
```

Kluczowe właściwości dobrego zadania w pipeline:
- **Idempotentność** – wielokrotne uruchomienie dla tego samego okresu daje ten sam wynik (np. `INSERT OVERWRITE partition=2024-01-15` zamiast `INSERT`).
- **Determinizm** – wynik zależy tylko od wejścia i parametrów (np. daty logicznej), a nie od „teraz”.
- **Atomowość** – zapis do lokalizacji tymczasowej, potem zamiana (rename); brak „połówek” danych.
- **Backfill** – możliwość przeliczenia historii dla nowej logiki.

Przykład szkieletu DAG w Airflow (TaskFlow API):

```python
from datetime import datetime
from airflow.decorators import dag, task

@dag(schedule="@daily", start_date=datetime(2024, 1, 1), catchup=True)
def churn_features():
    @task
    def extract(ds=None) -> str:            # ds = data logiczna, np. "2024-01-15"
        path = f"s3://lake/bronze/events/date={ds}/"
        # ... kopiowanie danych dla JEDNEGO dnia -> idempotentne
        return path

    @task
    def validate(path: str) -> str:
        # ... uruchom oczekiwania; wyjątek zatrzymuje pipeline
        return path

    @task
    def build_features(path: str, ds=None) -> None:
        # ... nadpisz partycję gold/features/date={ds}
        ...

    build_features(validate(extract()))

churn_features()
```

---

## 5. Pipeline przetwarzania danych w Pythonie

### Przykład: Pipeline z użyciem pandas i scikit-learn

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# --- Krok 1: Wczytanie danych ---
def load_data(path: str) -> pd.DataFrame:
    """Wczytuje dane i wykonuje podstawową walidację."""
    df = pd.read_parquet(path)
    
    required_cols = ['age', 'income', 'category', 'churn']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Brakujące kolumny: {missing}")
    
    print(f"Wczytano {len(df):,} wierszy, {df.shape[1]} kolumn")
    return df

# --- Krok 2: Czyszczenie danych ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa duplikaty i oczywiste błędy."""
    initial_size = len(df)
    
    # Usuń duplikaty
    df = df.drop_duplicates()
    
    # Usuń wiersze z ujemnym wiekiem
    df = df[df['age'] > 0]
    
    # Usuń wartości odstające (IQR method)
    Q1 = df['income'].quantile(0.25)
    Q3 = df['income'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df['income'].between(Q1 - 3*IQR, Q3 + 3*IQR)]
    
    removed = initial_size - len(df)
    print(f"Usunięto {removed:,} wierszy ({removed/initial_size*100:.1f}%)")
    return df

# --- Krok 3: Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy nowe cechy."""
    df = df.copy()
    
    # Cechy pochodne
    df['income_per_age'] = df['income'] / df['age']
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 25, 35, 50, 65, 100],
                              labels=['<25', '25-35', '35-50', '50-65', '65+'])
    
    return df

# --- Krok 4: Preprocessing Pipeline ---
def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """Buduje pipeline preprocessingu."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

# --- Główny pipeline ---
def run_data_pipeline(input_path: str) -> tuple:
    """Uruchamia pełny pipeline przetwarzania danych."""
    
    # Wczytaj i oczyść
    df = load_data(input_path)
    df = clean_data(df)
    df = engineer_features(df)
    
    # Podział na cechy i etykietę
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Podział na zbiory
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
    numeric_features = ['age', 'income', 'income_per_age']
    categorical_features = ['category', 'age_group']
    
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nWymiary po preprocessingu:")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Test:  {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_02/02_data_pipeline.py`  
> **Kiedy:** po omówieniu kodu z sekcji 5.  
> **Co pokazać:** log czyszczenia (duplikaty, błędne dochody), raport klasyfikacji i predykcja z zapisanego pipeline'u. Podkreśl, że imputacja i skalowanie są dopasowywane tylko na zbiorze treningowym.

---

## 6. Walidacja danych – Great Expectations

Walidacja danych to kluczowy element MLOps. Biblioteka **Great Expectations** pozwala definiować i weryfikować oczekiwania dotyczące danych.

```python
import great_expectations as gx
import pandas as pd

# Przykładowe dane
df = pd.DataFrame({
    'age': [25, 30, -5, 200, 45],       # -5 i 200 to błędy
    'income': [50000, 60000, None, 80000, 55000],
    'churn': [0, 1, 0, 1, 2]            # 2 to błąd (powinno być 0 lub 1)
})

# Tworzenie kontekstu GE
context = gx.get_context()

# Definicja oczekiwań (expectations)
validator = context.sources.pandas_default.read_dataframe(df)

# Oczekiwania dotyczące kolumny 'age'
validator.expect_column_values_to_be_between('age', min_value=0, max_value=120)
validator.expect_column_values_to_not_be_null('age')

# Oczekiwania dotyczące kolumny 'income'
validator.expect_column_values_to_not_be_null('income')
validator.expect_column_values_to_be_between('income', min_value=0)

# Oczekiwania dotyczące kolumny 'churn'
validator.expect_column_values_to_be_in_set('churn', value_set=[0, 1])

# Uruchomienie walidacji
results = validator.validate()
print(f"Walidacja: {'✅ OK' if results.success else '❌ FAILED'}")
for result in results.results:
    status = "✅" if result.success else "❌"
    print(f"  {status} {result.expectation_config.expectation_type}: "
          f"{result.expectation_config.kwargs}")
```

### Schemat walidacji danych w pipeline

```mermaid
flowchart TD
    A[Nowe dane] --> B{Walidacja\nschema}
    B -->|Błąd schematu| E1[Alert:\nBłąd krytyczny]
    B -->|OK| C{Walidacja\nstatystyczna}
    C -->|Anomalie| E2[Alert:\nSprawdź dane]
    C -->|OK| D{Walidacja\nbiznesowa}
    D -->|Naruszenie reguł| E3[Alert:\nReguły biznesowe]
    D -->|OK| F[Dane zatwierdzone\ndo treningu]
```


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_02/03_data_validation.py && python lectures/scripts/lecture_02/03_data_validation.py --corrupt; echo $?`  
> **Kiedy:** po przykładzie Great Expectations.  
> **Co pokazać:** te same oczekiwania (w tym parametr `mostly`) zaimplementowane w czystym pandas. Na danych z błędami walidacja nie przechodzi, a skrypt zwraca kod 1, co w CI/CD zatrzymuje pipeline.


### Wymiary jakości danych

| Wymiar | Pytanie | Przykładowa reguła |
|---|---|---|
| Kompletność | Czy brakuje wartości/wierszy? | `age` nie-null w ≥ 99% |
| Poprawność | Czy wartości są w dozwolonym zakresie? | `0 ≤ age ≤ 120` |
| Spójność | Czy dane są zgodne między systemami? | suma zamówień = suma w księgowości |
| Unikalność | Czy nie ma duplikatów? | `customer_id` unikalny |
| Aktualność | Czy dane są świeże? | ostatni rekord < 24 h |
| Stabilność rozkładu | Czy rozkład się nie zmienił? | PSI(`income`) < 0.2 |

### Training-serving skew

Najczęstsze źródło „model działał offline, a na produkcji nie”: **cechy liczone inaczej podczas treningu i serwowania**.

```mermaid
flowchart LR
    subgraph Trening
        H[(Dane historyczne)] --> P1["feature code\n(Python / pandas)"] --> M[Model]
    end
    subgraph Produkcja
        R[Request] --> P2["feature code\n(Java / SQL – przepisany!)"] --> M2[Model]
    end
    P1 -. "różnice: zaokrąglenia,\nstrefy czasowe, braki" .-> P2
```

Rozwiązania: jedna implementacja cech (Feature Store, wspólna biblioteka), zapis cech użytych przy predykcji (*log & wait*) i porównywanie ich z cechami treningowymi.

---

## 7. Feature Store – centralne repozytorium cech

**Feature Store** to system do przechowywania, zarządzania i serwowania cech ML.

### Problem bez Feature Store

```mermaid
graph LR
    subgraph "Bez Feature Store"
        DS1[Data Scientist 1] -->|oblicza| F1[cechy A, B, C]
        DS2[Data Scientist 2] -->|oblicza| F2[cechy A, B, D]
        DS3[Data Scientist 3] -->|oblicza| F3[cechy A, E, F]
        F1 & F2 & F3 -->|duplikacja\nniezgodność| PROB[Problemy!]
    end
```

### Rozwiązanie z Feature Store

```mermaid
graph LR
    subgraph "Z Feature Store"
        ETL[ETL Pipeline] --> FS[Feature Store]
        FS --> DS1[Data Scientist 1]
        FS --> DS2[Data Scientist 2]
        FS --> DS3[Data Scientist 3]
        FS --> PROD[Serwis produkcyjny]
    end
```

### Implementacja prostego Feature Store z Feast

```python
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64
from datetime import timedelta

# Definicja encji
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Identyfikator użytkownika"
)

# Źródło danych
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp"
)

# Definicja Feature View
user_stats_fv = FeatureView(
    name="user_statistics",
    entities=["user_id"],
    ttl=timedelta(days=7),  # czas życia cechy
    features=[
        Feature(name="age", dtype=Float32),
        Feature(name="income", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int64),
        Feature(name="total_purchases_30d", dtype=Int64),
        Feature(name="avg_order_value", dtype=Float32),
    ],
    source=user_stats_source,
)

# Pobieranie cech dla treningu (historical features)
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo/")

# Pobierz historyczne cechy dla zbioru treningowego
entity_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "event_timestamp": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-15"])
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_statistics:age",
        "user_statistics:income",
        "user_statistics:total_purchases_30d",
    ]
).to_df()

print(training_df)

# Pobieranie cech online (dla predykcji w czasie rzeczywistym)
online_features = store.get_online_features(
    features=["user_statistics:age", "user_statistics:income"],
    entity_rows=[{"user_id": 1001}]
).to_dict()

print(online_features)
```


### Point-in-time correctness – poprawność w czasie

Budując zbiór treningowy, dla każdego zdarzenia wolno użyć **tylko cech znanych w chwili tego zdarzenia**. Zwykły `JOIN` po `user_id` wziąłby najnowszą wartość cechy – czyli informację z przyszłości (data leakage).

#### Dlaczego to ważne?

Model na produkcji zawsze dostaje cechy „ze stanu na teraz”, czyli z przeszłości względem chwili, dla której przewiduje. Zbiór treningowy musi odtwarzać dokładnie tę sytuację. Jeśli w treningu do zdarzenia z 15 stycznia dołączymy cechę policzoną 21 stycznia, model uczy się z informacji, której w momencie decyzji nikt jeszcze nie miał. Skutki:

- **zawyżone metryki offline** – walidacja wygląda świetnie, bo model „zna przyszłość”;
- **spadek jakości po wdrożeniu** – na produkcji tej informacji nie ma, więc model działa gorzej, niż obiecywały testy;
- **trudna diagnoza** – kod jest poprawny, dane też, błąd tkwi tylko w sposobie złączenia w czasie.

To szczególny przypadek data leakage (sekcja 10) i jedna z przyczyn training-serving skew (sekcja 6).

#### Kluczowe pojęcia

| Pojęcie | Znaczenie | Przykład |
|---|---|---|
| **Encja** (entity) | obiekt, którego dotyczą cechy; klucz złączenia | `user_id = 1001` |
| **Czas zdarzenia** (event timestamp) | chwila, dla której robimy predykcję i której dotyczy etykieta | wniosek złożony 15.01 |
| **Czas cechy** (feature timestamp) | chwila, od której dana wartość cechy obowiązuje | `avg_order = 80` od 11.01 |
| **Czas dostępności** (created / ingestion timestamp) | chwila, w której wartość faktycznie trafiła do systemu | wyliczona nocnym jobem 12.01 o 02:00 |
| **TTL** (time-to-live) | jak długo wartość cechy uznajemy za aktualną | cecha starsza niż 30 dni = brak (`NaN`) |

Reguła point-in-time join: dla każdego wiersza etykiety `(encja, t)` wybierz **ostatnią wartość cechy z `czas_cechy ≤ t`** (a w wersji ostrożnej `czas_dostępności ≤ t`), o ile nie jest starsza niż TTL. Jest to tzw. złączenie *as-of* („według stanu na dzień”).

```mermaid
flowchart LR
    L["Etykieta\nuser 1001, t = 15.01"] --> Q{"Wartości cechy\nz czasem ≤ 15.01"}
    F1["01.01: 50"] --> Q
    F2["11.01: 80"] --> Q
    F3["21.01: 200\n(przyszłość – odrzucona)"] -.->|"✗"| Q
    Q -->|"najnowsza = 80\n(w granicach TTL)"| R["Wiersz treningowy\navg_order = 80"]
```

```mermaid
gantt
    dateFormat YYYY-MM-DD
    title Cechy użytkownika 1001 vs moment etykiety
    section Wartości cechy
    avg_order=50   :a1, 2024-01-01, 10d
    avg_order=80   :a2, 2024-01-11, 10d
    avg_order=200  :a3, 2024-01-21, 10d
    section Zdarzenie
    etykieta (15.01) → poprawnie avg_order=80 :milestone, 2024-01-15, 0d
```

Feature Store realizuje to przez *point-in-time join*. W pandas ten sam efekt daje `merge_asof`:

```python
import pandas as pd

features = pd.DataFrame({
    "user_id": [1001, 1001, 1001],
    "ts": pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"]),
    "avg_order": [50, 80, 200],
})
labels = pd.DataFrame({"user_id": [1001], "ts": pd.to_datetime(["2024-01-15"]), "churn": [1]})

wrong = labels.merge(features.sort_values("ts").groupby("user_id").tail(1), on="user_id")
right = pd.merge_asof(labels.sort_values("ts"), features.sort_values("ts"),
                      on="ts", by="user_id", direction="backward")
print(wrong[["avg_order"]].iloc[0].item(), "← wyciek z przyszłości")   # 200
print(right[["avg_order"]].iloc[0].item(), "← poprawnie")              # 80
```

Co robi ten kod:
- `wrong` – bierze **ostatni** znany wiersz cechy dla użytkownika (21.01, wartość 200) i dokleja go do etykiety z 15.01. Tak działa naiwny `JOIN` z tabelą „aktualnych” cech.
- `right` – `merge_asof` dla każdej etykiety szuka po `user_id` (`by=`) wiersza cechy o największym `ts`, które jest **≤** czasowi etykiety (`direction="backward"`). Obie tabele muszą być posortowane po kolumnie czasu.

#### Pułapka: opóźnienie dostępności i nieaktualne cechy

Cecha z czasem 11.01 mogła zostać policzona dopiero następnego dnia (np. dzienny job wsadowy). Jeśli zdarzenie było 11.01 o 10:00, na produkcji tej wartości jeszcze by nie było. Dlatego złączenie robi się po **czasie dostępności**, a nie po czasie cechy. Drugi problem to wartości bardzo stare – parametr `tolerance` odpowiada TTL:

```python
import pandas as pd

features = pd.DataFrame({
    "user_id": [1001, 1001],
    "feature_ts": pd.to_datetime(["2024-01-01", "2024-01-11"]),
    "avg_order": [50, 80],
})
# cechy liczone nocnym jobem – dostępne dzień później o 02:00
features["available_ts"] = features["feature_ts"] + pd.Timedelta(days=1, hours=2)

labels = pd.DataFrame({
    "user_id": [1001, 1001, 1001],
    "ts": pd.to_datetime(["2024-01-11 10:00", "2024-01-15 00:00", "2024-03-01 00:00"]),
})

result = pd.merge_asof(
    labels.sort_values("ts"),
    features.sort_values("available_ts"),
    left_on="ts", right_on="available_ts", by="user_id",
    direction="backward",
    tolerance=pd.Timedelta(days=30),   # TTL: starsze wartości → NaN
)
print(result[["ts", "avg_order"]])
# 2024-01-11 10:00 → 50   (wartość 80 jeszcze niedostępna)
# 2024-01-15       → 80
# 2024-03-01       → NaN  (ostatnia wartość starsza niż 30 dni)
```

Odpowiednik w SQL (np. BigQuery, Snowflake), gdy brak `ASOF JOIN`:

```sql
SELECT l.user_id, l.ts, f.avg_order
FROM labels l
LEFT JOIN features f
  ON f.user_id = l.user_id
 AND f.available_ts <= l.ts
 AND f.available_ts >  l.ts - INTERVAL 30 DAY
QUALIFY ROW_NUMBER() OVER (PARTITION BY l.user_id, l.ts ORDER BY f.available_ts DESC) = 1;
```

#### Typowe błędy i dobre praktyki

- **Tabela cech bez historii** (tylko „stan bieżący”, nadpisywany każdej nocy) – nie da się z niej zbudować poprawnego zbioru treningowego. Cechy przechowuj jako historię (append-only, warstwa Bronze/Silver) z kolumną czasu.
- **Agregaty liczone na całym zbiorze**, np. „średnia wartość zamówień klienta” policzona ze wszystkich lat – zawiera przyszłość. Okno agregacji musi kończyć się przed czasem zdarzenia (`rolling` z `closed="left"`).
- **Strefy czasowe** – porównuj znaczniki czasu w jednej strefie (najlepiej UTC), inaczej przesunięcie o kilka godzin daje wyciek albo brak cech.
- **Losowy podział train/test** przy danych czasowych – nawet przy poprawnym złączeniu test „miesza się” z treningiem. Dziel po czasie (trening: przeszłość, test: przyszłość).
- **Test kontrolny** – w pipeline sprawdzaj asercją, że dla każdego wiersza `feature_ts ≤ event_ts`.

Feature Store (np. Feast `get_historical_features`) wykonuje point-in-time join automatycznie na podstawie `timestamp_field` źródła i `ttl` widoku cech, dlatego ta sama definicja cechy daje spójne dane dla treningu (offline) i serwowania (online).

### Online vs offline store

| | Offline store | Online store |
|---|---|---|
| Cel | zbiory treningowe, batch scoring | predykcja w czasie rzeczywistym |
| Technologia | Parquet, BigQuery | Redis, DynamoDB |
| Latencja | minuty | milisekundy |
| Dane | pełna historia | tylko najnowsza wartość cechy |

---

## 8. Wersjonowanie danych z DVC

**DVC** (Data Version Control) to narzędzie do wersjonowania danych i modeli, działające razem z Git.

```bash
# Inicjalizacja DVC w projekcie
git init
dvc init

# Dodanie zdalnego storage (np. Google Cloud Storage)
dvc remote add -d myremote gs://my-bucket/dvc-storage

# Śledzenie pliku danych
dvc add data/train.parquet
git add data/train.parquet.dvc .gitignore
git commit -m "Add training data v1"

# Przesłanie danych do zdalnego storage
dvc push

# Przełączenie na inną wersję danych
git checkout v1.0
dvc checkout
```

### Struktura pliku .dvc

```yaml
# data/train.parquet.dvc
outs:
- md5: a1b2c3d4e5f6...
  size: 52428800
  path: train.parquet
```

### Pipeline DVC

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/
    outs:
      - data/prepared/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/prepared/
    params:
      - params.yaml:
          - train.n_estimators
          - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false
```

```python
# params.yaml
train:
  n_estimators: 100
  max_depth: 5
  random_state: 42
test_size: 0.2
```


> **▶️ Demo na żywo:** `bash lectures/scripts/lecture_02/05_dvc_demo.sh`  
> **Kiedy:** po omówieniu poleceń DVC i pliku `.dvc`.  
> **Co pokazać:** zawartość pliku `train.csv.dvc` (hash md5), zmianę liczby wierszy z 5000 na 1000 po `git checkout data-v1 && dvc checkout` oraz historię commitów.

---

## 9. Streaming vs Batch Processing

### Batch Processing
- Przetwarzanie dużych zbiorów danych w regularnych interwałach.
- Narzędzia: Apache Spark, dbt, pandas.
- Użycie: trening modeli, raporty dzienne.

### Stream Processing
- Przetwarzanie danych w czasie rzeczywistym.
- Narzędzia: Apache Kafka, Apache Flink, Spark Streaming.
- Użycie: wykrywanie fraudów, rekomendacje real-time.

```mermaid
graph LR
    subgraph "Batch Processing"
        B1[Dane historyczne] -->|co godzinę/dzień| B2[Spark/dbt] --> B3[Wyniki]
    end

    subgraph "Stream Processing"
        S1[Kafka Topic] -->|ms-sekundy| S2[Flink/Spark\nStreaming] --> S3[Wyniki\nreal-time]
    end
```

### Przykład: Kafka Producer/Consumer dla ML

```python
from kafka import KafkaProducer, KafkaConsumer
import json
import numpy as np

# Producer – wysyłanie zdarzeń do Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_prediction_request(user_id: int, features: dict):
    """Wysyła żądanie predykcji do Kafka."""
    message = {
        "user_id": user_id,
        "features": features,
        "timestamp": "2024-01-15T10:30:00Z"
    }
    producer.send('ml-predictions', value=message)
    producer.flush()

# Consumer – odbieranie i przetwarzanie zdarzeń
consumer = KafkaConsumer(
    'ml-predictions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='ml-service-group',
    auto_offset_reset='earliest'
)

def process_predictions(model):
    """Przetwarza żądania predykcji ze strumienia."""
    for message in consumer:
        data = message.value
        features = np.array(list(data['features'].values())).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        print(f"User {data['user_id']}: "
              f"churn={'TAK' if prediction else 'NIE'} "
              f"(p={probability:.2%})")
```


### Architektury Lambda i Kappa

```mermaid
flowchart LR
    subgraph Lambda
        S1[Strumień] --> B1[Batch layer\nSpark, codziennie]
        S1 --> SP1[Speed layer\nFlink, na bieżąco]
        B1 --> SV1[Serving: scalanie wyników]
        SP1 --> SV1
    end
    subgraph Kappa
        S2[Log zdarzeń\nKafka] --> SP2[Jeden silnik\nstreamingowy]
        SP2 --> SV2[Serving]
        S2 -. replay historii .-> SP2
    end
```

- **Lambda** – dwie ścieżki (dokładna batch + szybka streaming); wada: dwie implementacje tej samej logiki.
- **Kappa** – tylko streaming; historię przelicza się, odtwarzając log zdarzeń.

### Okna czasowe w przetwarzaniu strumieniowym

Cechy strumieniowe (np. „liczba transakcji w ostatnich 10 min”) liczy się w oknach:
- **tumbling** – rozłączne okna stałej długości (00:00–00:10, 00:10–00:20),
- **sliding/hopping** – okna nakładające się (10 min co 1 min),
- **session** – okno zamykane po okresie bezczynności użytkownika.

Ważne rozróżnienie: **event time** (kiedy zdarzenie zaszło) vs **processing time** (kiedy je przetworzono). Spóźnione zdarzenia obsługuje się *watermarkami*.

```python
import pandas as pd

tx = pd.DataFrame({
    "ts": pd.to_datetime(["10:00", "10:03", "10:07", "10:12", "10:14"], format="%H:%M"),
    "amount": [20, 35, 500, 15, 40],
}).set_index("ts")

print(tx["amount"].resample("10min").sum())          # okna tumbling
print(tx["amount"].rolling("10min").count())         # okno przesuwne: liczba tx w ost. 10 min
```

---

## 10. Typowe pułapki w inżynierii danych dla ML

> ⚠️ **Pułapka 1: Data Leakage (wyciek danych)**
> Najczęstszy i najgroźniejszy błąd — informacje z przyszłości „wyciekają" do danych treningowych. Przykład: normalizacja cech na całym zbiorze przed podziałem train/test powoduje, że statystyki zbioru testowego wpływają na trening.

```python
# ❌ ŹLE: normalizacja przed podziałem (data leakage!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit na CAŁYM zbiorze
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ DOBRZE: normalizacja po podziale
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit tylko na train
X_test_scaled = scaler.transform(X_test)         # transform na test
```

> ⚠️ **Pułapka 2: Training-Serving Skew**
> Preprocessing danych w treningu różni się od preprocessingu w produkcji. Rozwiązanie: używaj tego samego kodu (np. sklearn Pipeline) w obu środowiskach.

> ⚠️ **Pułapka 3: Ignorowanie brakujących wartości**
> Model wytrenowany na danych bez braków może się zachowywać nieprzewidywalnie, gdy w produkcji pojawią się wartości null. Zawsze testuj model z brakującymi danymi.


> **▶️ Demo na żywo:** `python lectures/scripts/lecture_02/04_data_leakage.py`  
> **Kiedy:** przy omawianiu pułapki „normalizacja przed podziałem”.  
> **Co pokazać:** na czystym szumie selekcja cech przed CV daje accuracy ~0.85, a poprawny `Pipeline` daje ~0.5. Cecha „z przyszłości” podnosi wynik z ~0.65 do ~0.95.

### Case Study: Zillow — katastrofa danych

**Zillow** (portal nieruchomości) stracił **304 miliony dolarów** w 2021 roku z powodu błędów w modelu wyceny nieruchomości. Główne przyczyny:
- Model był trenowany na danych historycznych, które nie uwzględniały gwałtownych zmian rynkowych (pandemia COVID-19).
- Brak odpowiedniego monitoringu dryfu danych — model kontynuował predykcje mimo fundamentalnej zmiany rozkładu cen.
- Niewystarczająca walidacja danych wejściowych — model akceptował nierealistyczne wartości cech.

---

## Pytania kontrolne i do dyskusji

1. Wyjaśnij różnicę między Data Lake a Data Warehouse. Kiedy użyjesz każdego z nich?
2. Dlaczego Parquet jest lepszym formatem niż CSV dla dużych zbiorów danych ML?
3. Czym różni się ETL od ELT? Jakie są zalety podejścia ELT?
4. Co to jest data leakage i jak go uniknąć?
5. Jakie korzyści daje Feature Store w porównaniu z ręcznym obliczaniem cech?
6. Dlaczego wersjonowanie danych (DVC) jest ważne w projektach ML?
7. **Dyskusja:** Kiedy warto użyć stream processingu zamiast batch processingu w kontekście ML?


### Proponowane odpowiedzi

1. Data Lake przechowuje surowe, różnorodne dane „as-is”, a Data Warehouse dane już oczyszczone i ustrukturyzowane pod analitykę. Lake wybieram do eksploracji i długiego przechowywania, Warehouse do raportowania i stabilnych zapytań biznesowych.
2. Parquet jest kolumnowy, kompresuje dane i pozwala czytać tylko potrzebne kolumny, więc jest szybszy i tańszy I/O przy dużych zbiorach. CSV jest tekstowy, cięższy i mniej wydajny przy analityce ML.
3. ETL: transformujesz przed załadowaniem do magazynu; ELT: najpierw ładujesz, potem transformujesz w silniku docelowym. ELT lepiej skaluje się w chmurze, daje większą elastyczność i łatwiejsze odtwarzanie transformacji.
4. Data leakage to sytuacja, gdy model „widzi” informacje niedostępne w momencie predykcji (np. statystyki z testu podczas treningu). Unikamy tego przez poprawny split czasowy/train-test, pipeline preprocessingu fitowany tylko na train i rygor walidacji cech.
5. Feature Store zapewnia spójność cech online/offline, wersjonowanie, reużywalność i centralne zarządzanie definicjami cech. Ręczne liczenie cech zwiększa ryzyko niespójności i duplikacji.
6. DVC daje odtwarzalność eksperymentów: wiemy, na jakiej wersji danych trenowano dany model. Ułatwia audyt, porównania i rollback po pogorszeniu jakości.
7. Stream processing warto wybrać, gdy decyzje muszą być podejmowane blisko czasu rzeczywistego (np. fraud, rekomendacje live). Batch wystarcza, gdy dopuszczalne jest opóźnienie i ważniejsza jest prostota/koszt.


---

## Podsumowanie

- Dane są fundamentem systemów ML – ich jakość determinuje jakość modelu.
- **Data Lake** przechowuje surowe dane, **Data Warehouse** przetworzone, **Data Lakehouse** łączy zalety obu.
- **Parquet** to preferowany format dla ML (kolumnowy, skompresowany, z schematem).
- **Walidacja danych** (Great Expectations) powinna być integralną częścią pipeline'u — nie opcjonalnym dodatkiem.
- **Feature Store** eliminuje duplikację i zapewnia spójność cech między treningiem a produkcją.
- **DVC** umożliwia wersjonowanie danych razem z kodem.
- Unikaj typowych pułapek: data leakage, training-serving skew, ignorowanie brakujących wartości.

## Literatura i zasoby

- [Fundamentals of Data Engineering (O'Reilly)](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [DVC Documentation](https://dvc.org/doc)
- [Feast Feature Store](https://feast.dev/)
- [Apache Parquet Format](https://parquet.apache.org/)
- [Zillow's iBuying Debacle – What Went Wrong](https://insidebigdata.com/2021/12/13/the-500-million-lesson-from-zillow/)
- [Data Leakage in Machine Learning (Kaggle)](https://www.kaggle.com/code/alexisbcook/data-leakage)
