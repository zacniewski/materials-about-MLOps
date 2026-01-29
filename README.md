# MLOps i inżynieria systemów

## Opis przedmiotu
* **Laboratoria:** 30 godzin
* **Projekt:** 30 godzin

## Wstęp do MLOps

Celem MLOps jest zapewnienie niezawodności, wydajności i realnej wartości modeli machine learning. Pomaga firmom w płynnym wdrażaniu modeli machine learning w środowisku produkcyjnym i zapewnianiu jego prawidłowego działania.

### Co to jest Machine Learning?

Machine Learning to dziedzina sztucznej inteligencji, która umożliwia komputerom uczenie się i ulepszanie na podstawie doświadczenia, bez konieczności wyraźnego programowania. Polega ona na opracowywaniu algorytmów i modeli statystycznych, które umożliwiają systemom skuteczną realizację określonych zadań poprzez analizę danych, identyfikację wzorców, a także podejmowanie przewidywań lub decyzji.

Kluczową ideą uczenia maszynowego jako części rozwiązań AI jest tworzenie programów, które mają dostęp do danych, uczyć się z nich, a następnie wykorzystywać je do podejmowania świadomych decyzji lub prognoz bez polegania na programowaniu opartym na regułach.

Machine Learning to różnorodne podejścia, z których każda ma swoje mocne strony i zastosowania. Oto kilka typowych typów:

#### Uczenie nadzorowane
W ramach tej metody dane wykorzystywane do treningu są etykietowane. Wyobraźmy sobie algorytm uczenia maszynowego, tysiące zdjęć kotów i psów, z jasno oznakowanym zdjęciem. Pozwala to algorytmowi na poznanie cech charakterystycznych pozwalających odróżnić koty od psów, a następnie na wykorzystanie tej wiedzy do identyfikacji nowych, niewidocznych obrazów.

#### Uczenie nienadzorowane
W tym przypadku dane nie są etykietowane. Algorytm uczenia maszynowego musi niezależnie wyszukiwać wzorce i relacje w danych. Jest to przydatne podczas wykonywania zadań, takich jak wykrywanie anomalii lub klaster danych.

#### Uczenie się przez wzmocnienie
Metoda ta polega na trenowaniu algorytmu za pomocą procesu prób i błędów. Algorytm wchodzi w interakcję z symulowanym środowiskiem i otrzymuje nagrody za pożądane zachowania, dzięki czemu z czasem uczy się optymalnych strategii.

### Machine Learning w kontekście AI

U podstaw sztucznej inteligencji leży nauka tworzenia maszyn zdolnych do wykonywania zadań, które zazwyczaj wymagają ludzkiej inteligencji. Począwszy od rozwiązywania problemów i podejmowania decyzji aż po rozpoznawanie mowy i tłumaczenie językowe. AI obejmuje wiele technik i metodologii, wśród których szczególnie potężnym i wszechstronnym narzędziem jest Machine Learning. Fundamentalną ideą ML jest to, że systemy mogą uczyć się na podstawie danych, identyfikować wzorce i podejmować decyzje przy minimalnej interwencji człowieka.

Machine Learning przyspiesza rozwój AI, oferując bardziej dynamiczne podejście do analizy danych. Dzięki tej zdolności systemy sztucznej inteligencji mogą dostosowywać się do nowych okoliczności i doskonalić się w czasie, co jest kluczowe dla aplikacji wymagających częstych aktualizacji lub przetwarzających złożone, zmienne zbiory danych.

Dzięki algorytmom uczenia, modele ML mogą przetwarzać duże ilości danych z prędkością i skalą nieosiągalną dla analityków. Dlatego właśnie ML stał się szkieletem wielu współczesnych systemów AI, napędzając postęp w tak różnych dziedzinach, jak opieka zdrowotna, finanse, autonomiczne pojazdy i inteligentne miasta.

ML jest istotnym obszarem zainteresowania w zakresie sztucznej inteligencji i stanowi część szerszego ekosystemu technologii AI, w tym deep learning, przetwarzania języka naturalnego (NLP) i robotyki. Deep Learning, podzbiór uczenia maszynowego, wykonuje złożone zadania, takie jak rozpoznawanie obrazu i mowy, za pośrednictwem sieci neuronowych, które naśladują funkcje ludzkiego mózgu.

Protokół NLP, który umożliwia maszynom rozumienie i interpretację języka ludzkiego, często wykorzystuje ML do ulepszania algorytmów. W robotyce algorytmy ML pomagają robotom uczyć się na podstawie ich środowiska i doświadczeń, zwiększając ich autonomię. Ta współzależność pokazuje, jak ML nie tylko korzysta z, ale również przyczynia się do rozwoju innych domen AI.

### Jak działa MLOps?

Cykl życia MLOps składa się z czterech cyklów głównych. Każdy cykl określa etap, na którym mają zostać przeprowadzone udane operacje uczenia maszynowego. Cztery cykle lub etapy to:

1. **Cykl danych:** Wymaga to gromadzenia i przygotowania danych do treningu modeli ML. Surowe dane są gromadzone z różnych źródeł, a techniki, takie jak inżynieria funkcji, przekształcają je i organizują w dane etykietowane, gotowe do treningu modeli.
2. **Cykl modelu:** W tym cyklu model ML jest trenowany z wykorzystaniem przygotowanych danych. Kluczowe jest śledzenie różnych wersji modelu w miarę jego rozwoju w cyklu życia, co można zrobić za pomocą narzędzi, takich jak MLflow.
3. **Cykl rozwoju:** Wytrenowany model jest dalej rozwijany, testowany i zatwierdzany w celu zapewnienia jego gotowości do wdrożenia w środowisku produkcyjnym. Zautomatyzowane ciągłe procesy integracji i ciągłego dostarczania (CI/CD) pozwalają ograniczyć liczbę zadań ręcznych.
4. **Cykl operacyjny:** Proces monitorowania zapewnia dobrą wydajność modelu produkcyjnego i jest trenowany w miarę potrzeb, aby z czasem go ulepszać. MLOps mogą automatycznie przetrenować model zgodnie z harmonogramem lub gdy metryki wydajności spadną poniżej progu.

### Podstawowe zasady MLOps

Platforma MLOps została zbudowana w oparciu o podstawowe zasady, które zapewniają niezawodność, wydajność i skalowalność modeli machine learning w realnym świecie. Oto szczegóły kilku podstawowych zasad:

* **Automatyzacja:** Głównym założeniem MLOps jest automatyzacja powtarzalnych zadań w całym cyklu życia machine learning. Obejmuje to zarządzanie data pipeline, trenowanie modeli, testy, wdrażanie i monitoring. Automatyzacja minimalizuje błędy ludzkie i uwalnia data scientists do koncentrowania się na zadaniach wyższego poziomu, takich jak tworzenie i ulepszanie modeli.
* **Kontrola wersji i odtwarzalność:** MLOps kładzie nacisk na monitorowanie każdej zmiany wprowadzonej w danych, kodzie i modelach. W ten sposób można w prosty sposób przywrócić poprzednie wersje, jeśli jest to konieczne, i zapewnić powtarzalność eksperymentów. Każdy członek zespołu potrafi zrozumieć pochodzenie modelu i sposób jego rozwoju.
* **Ciągła integracja i dostarczanie (CI/CD):** Nowoczesne MLOps integruje się z narzędzi do programowania wykorzystywania przez data scientists. Wprowadzając zmiany, automatyzacja testów sprawdza, czy wszystko działa zgodnie z oczekiwaniami. Dzięki temu błędy są wychwytywane na wczesnym etapie cyklu rozwojowego i problemy nie opóźniają wdrożenia.
* **Współpraca:** MLOps sprzyja współpracy między zespołami data science, inżynierii i operacji. Usprawniając procesy i zapewniając wspólną widoczność cyklu życia modelu, MLOps rozbija silosy i gwarantuje, że wszyscy pracują nad tym samym celem.
* **Pętle monitorowania i sprzężenia zwrotnego:** Proces MLOps stale monitoruje wydajność wdrażanych modeli. Śledzi metryki, takie jak dokładność, sprawiedliwość i potencjalne uprzedzenia. W przypadku błędu uruchamiane są alerty, które skłaniają do wszczęcia dochodzenia i umożliwiają podjęcie działań naprawczych. Ta pętla sprzężenia zwrotnego jest kluczowa dla utrzymania wydajności modelu i dostosowania się do zmian w świecie rzeczywistym.
* **Zarządzanie i zgodność z przepisami:** Bardzo ważne jest również, aby MLOps egzekwował polityki i procedury związane z rozwojem i wdrażaniem modeli. Zachowasz w ten sposób zgodność z przepisami dotyczącymi uczciwości, zrozumiałości i prywatności danych. Narzędzia MLOps pozwalają na śledzenie pochodzenia danych używanych do trenowania modeli i dokumentowanie procesu podejmowania decyzji na potrzeby audytów.
* **Skalowalność / efektywność:** Praktyki MLOps sprawiają, że cały potok uczenia maszynowego jest w stanie poradzić sobie z rosnącą ilością danych i coraz większą złożonością modeli. Wymaga to wykorzystania infrastruktury i technologii konteneryzacji w chmurze do efektywnego wykorzystania zasobów i wdrażania modeli w różnych środowiskach.

### Zalety MLOps

MLOps oferuje szereg korzyści, które usprawniają cykl życia machine learning i pozwalają uwolnić prawdziwy potencjał Twoich modeli. Automatyzacja powtarzalnych zadań, takich jak przygotowywanie danych, trenowanie i wdrażanie, uwalnia data scientists do bardziej strategicznej pracy. Praktyki CI/CD przyspieszają rozwój poprzez wczesne wychwytywanie błędów i zapewnienie płynnego wdrażania. Przekłada się to na szybszy zwrot z inwestycji w projekty machine learning.

Wspiera również współpracę między zespołami data science, inżynierów i operatorów. Współdzielone narzędzia i procesy zapewniają każdemu widoczność w cyklu życia modelu, co prowadzi do lepszej komunikacji i usprawnionych przepływów pracy.

Praktyki MLOps, takie jak konteneryzacja i infrastruktury oparte na chmurze, pozwalają obsłużyć rosnącą ilość danych i rosnącą złożoność modeli. Dzięki temu możesz skutecznie skalować uczenie maszynowe w miarę jak zmieniają się Twoje potrzeby.

Zespołom, które wykorzystują MLOps zamiast innej drogi do sukcesu uczenia maszynowego podoba się również to, że MLOps egzekwuje polityki i procedury dotyczące rozwoju i wdrażania modeli. W związku z tym Twoje modele są zgodne z przepisami dotyczącymi uczciwości, zrozumiałości i ochrony danych. Narzędzia MLOps umożliwiają śledzenie procesów tworzenia linii danych i podejmowania decyzji w sprawie dokumentów na potrzeby audytów.

Zespoły korzystające z MLOps widzą, że automatyzacja zadań i optymalizacja wykorzystania zasobów przekłada się na znaczne oszczędności. Co więcej, dzięki wczesnemu wychwytywaniu błędów i wdrażaniu wysokiej jakości modeli można uniknąć kosztownych zmian i problemów z wydajnością.

MLOps ułatwia tworzenie pętli sprzężenia zwrotnego, w ramach której wdrażane modele są stale monitorowane. Umożliwia to identyfikację pogorszenia wydajności, dryfu danych lub potencjalnych problemów. Aktywne rozwiązywanie tych problemów gwarantuje, że modele pozostaną aktualne i z czasem przyniosą optymalne rezultaty.

### Jak wdrożyć MLOps?

Należy rozpocząć od utworzenia infrastruktury niezbędnej do wdrożenia procesu MLOps. Obejmuje to wykorzystanie systemu kontroli wersji do zarządzania kodem, danymi i artefaktami modeli, wdrożenie potoku CI/CD w celu automatyzacji budowy, testowania i wdrażania modeli, wdrożenie rejestru modeli w celu przechowywania i trenowania modeli oraz ustanowienie monitorowania i alertów w celu śledzenia wydajności modelu w produkcji.

Następnie organizacje powinny zdefiniować swoje procesy MLOps. Wiąże się to z ustanowieniem iteracyjnego procesu przyrostowego do projektowania, opracowywania i obsługi aplikacji ML, automatyzacji kompleksowego potoku ML (w tym przygotowywania danych, trenowania modeli, oceny i wdrażania) oraz wdrożeniem ciągłych procesów przekwalifikowania i aktualizacji modeli w oparciu o monitoring produkcji.

Wreszcie, organizacje powinny przyjąć najlepsze praktyki MLOps, takie jak wykorzystanie konteneryzacji w celu zapewnienia spójnych środowisk programistycznych i wdrożeniowych, wdrożenie rygorystycznych testów na każdym etapie potoku ML, utrzymanie magazynu funkcji do zarządzania i wprowadzania danych do wersji, wykorzystanie platform MLOps lub narzędzi do uproszczenia wdrażania oraz wspieranie współpracy między data scientists, inżynierami ML i zespołami DevOps.

### Wyzwania związane z MLOps

Wdrożenie MLOps, choć niezwykle korzystne, wiąże się z wyjątkowymi wyzwaniami. Problemy związane z danymi stanowią poważną przeszkodę. Zapewnienie jakości danych w całym procesie ma kluczowe znaczenie, ponieważ słabe dane prowadzą do niskiej wydajności i potencjalnie szkodliwych modeli. Dodatkowo, zarządzanie wersjonowaniem danych w celu uzyskania odtwarzalności modeli i procesów zarządzania, aby zapewnić bezpieczeństwo, prywatność i kwestie etyczne, to wszystkie złożone elementy układanki MLOps.

Brak wykwalifikowanego personelu może również stanowić przeszkodę na drodze. MLOps wymaga wielofunkcyjnych ekspertów w dziedzinie data science, inżynierii oprogramowania i zasad DevOps - znalezienie takich osób może być dość trudne. Poza tym sukces MLOps zależy od wspierania współpracy między zespołami, które w przeszłości pozostawały odizolowane.

Przełamywanie barier między data scientists, programistami i personelem operacyjnym przy jednoczesnym dostosowywaniu się do celów i procesów wymaga świadomej zmiany kulturowej w organizacjach.

Często zaniedbywanym obszarem jest monitorowanie modeli po wdrożeniu. Świat rzeczywisty jest dynamiczny, a wydajność modelu będzie się zmniejszać z czasem z powodu dryfu koncepcji.

Do gromadzenia informacji zwrotnych od użytkowników potrzebne są proaktywne systemy i mechanizmy monitorowania, dzięki którym modele są stale ulepszane i dostosowywane do potrzeb biznesowych użytkowników. Eksperymentowanie i odtwarzalność mogą być skomplikowane. Śledzenie mnogości eksperymentów, zmian w danych i powiązanych z nimi wyników jest niezbędne do zrozumienia procesu tworzenia modelu i usprawnienia przyszłych aktualizacji.

Nie należy lekceważyć tych wyzwań, ale można sobie z nimi poradzić. Inwestowanie w specjalistyczne platformy MLOps, zapewnianie możliwości szkoleniowych dla istniejącego personelu oraz priorytet jasnej komunikacji i współpracy między zespołami pomaga utorować drogę do płynniejszego wdrożenia MLOps.

### Różnice między MLOps i DevOps

Kluczową różnica między DevOps a MLOps polega na tym, że MLOps koncentruje się w szczególności na unikalnych wyzwaniach związanych z wdrażaniem modeli machine learning i zarządzaniem nimi w produkcji. Natomiast DevOps to szerszy zestaw praktyk usprawniających cykl rozwoju oprogramowania.

Zarówno DevOps, jak i MLOps dążą do zniwelowania luki między rozwojem a operacjami, natomiast MLOps dodaje dodatkowe zagadnienia specyficzne dla uczenia maszynowego. Obejmują one zarządzanie danymi używanymi do trenowania modeli, walidację wydajności modeli oraz monitorowanie modeli pod kątem spadku wydajności w czasie, gdy dane w świecie rzeczywistym są narażone na zmiany.

W przypadku potoku DevOps nacisk kładziony jest na automatyzację tworzenia, testowania i wdrażania aplikacji. W potoku MLOps dodatkowe etapy przygotowania danych, treningu modeli i oceny modeli muszą być zautomatyzowane i zintegrowane z procesem wdrożenia.

Kolejną kluczową różnicą jest potrzeba włączenia przez MLOps zasad odpowiedzialnej i etycznej sztucznej inteligencji. Zapewnienie, aby modele machine learning zachowywały się bezstronnie i w sposób transparentny, jest kwestią kluczową, która nie jest tak istotna w tradycyjnym rozwoju oprogramowania.

Chociaż DevOps i MLOps dzielą wiele wspólnych zasad dotyczących współpracy, automatyzacji i ciągłego doskonalenia, MLOps wprowadza dodatkowe złożoności dotyczące danych, modeli i zarządzania modelami, które wymagają specjalistycznych narzędzi i praktyk, innych niż zwykle spotykane w środowisku DevOps.

### Przykłady MLOps

MLOps jest wykorzystywany przez duże i małe firmy. Wiodący bank wdrożył na przykład MLOps, aby usprawnić proces onboardingu klientów. Bank wykorzystywał modele ML do automatyzacji weryfikacji informacji o kliencie oraz wykrywania oszustw w czasie rzeczywistym. Poprawiło to doświadczenie klienta, ponieważ proces onboardingu stał się szybszy i bardziej wydajny. Bank ograniczył również ryzyko oszustw, co zwiększyło zaufanie klientów.

Duża firma handlu detalicznego wykorzystywała MLOps do usprawnienia zarządzania łańcuchem dostaw. Firma wykorzystuje modele ML do przewidywania zapotrzebowania na produkty i optymalizacji alokacji zasobów w magazynach. Pozwoliło to na zwiększenie dokładności prognozowania popytu, zmniejszenie ilości odpadów i zwiększenie wydajności łańcucha dostaw.

MLOps pomagają również w opiece zdrowotnej. Dostawca usług medycznych wykorzystywał MLOps do poprawy wyników leczenia pacjentów. Dostawca korzystał z modeli ML do analizy danych pacjentów i identyfikacji pacjentów zagrożonych zdarzeniami niepożądanymi. Informacje te zostały wykorzystane do interwencji i zapobiegania zdarzeniom niepożądanym, w związku z czym dostawca usług odnotował znaczącą poprawę wyników leczenia pacjentów.

Duża firma logistyczna wykorzystała MLOps wraz z Google Cloud AI Platform do optymalizacji procesów łańcuchem dostaw. Firma opracowała i wdrożyła modele ML, które mogą dokładnie przewidywać zapotrzebowanie, optymalizować trasy i skrócić czas dostawy. Poprawiło to ogólną wydajność łańcucha dostaw i obniżyło koszty.

Przykłady te pokazują, w jaki sposób organizacje z różnych branż wykorzystują MLOps do usprawnienia przepływów pracy machine learning, poprawy wydajności operacyjnej i zapewnienia rzeczywistej wartości biznesowej.

## Poziomy dojrzałości MLOps (wg Google Cloud)

W procesie wdrażania MLOps można wyróżnić trzy poziomy dojrzałości, zależne od stopnia automatyzacji procesów:

### Poziom 0: Proces manualny
Charakteryzuje się brakiem automatyzacji. Każdy krok (analiza danych, przygotowanie danych, trenowanie modelu i walidacja) jest wykonywany ręcznie. Przejście od treningu do produkcji odbywa się poprzez przekazanie artefaktu (modelu) zespołowi operacyjnemu.
* **Wyzwania:** Brak monitorowania wydajności modelu w produkcji, rzadkie aktualizacje, ryzyko "dryfu" modelu.

### Poziom 1: Automatyzacja potoku ML
Celem jest ciągły trening modelu poprzez automatyzację potoku ML. Umożliwia to systemowi automatyczne douczanie modelu w produkcji na nowych danych.
* **Kluczowe cechy:** Szybkie eksperymentowanie, automatyczna walidacja danych i modelu, wyzwalacze potoków (triggers).

### Poziom 2: Automatyzacja potoku CI/CD
Najwyższy poziom dojrzałości, w którym zautomatyzowany jest nie tylko trening modelu, ale również proces budowania, testowania i wdrażania całego potoku ML.
* **Zalety:** Szybka iteracja, pełna odtwarzalność, wysoka niezawodność dzięki rygorystycznym testom jednostkowym i integracyjnym komponentów ML.

## Dokumentacja dodatkowa
* [Harmonogram przedmiotu (Laboratoria i Projekt)](SYLLABUS.md)
* [Dokumentacja Google Cloud MLOps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)