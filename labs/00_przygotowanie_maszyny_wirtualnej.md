# Laboratorium 0: Przygotowanie maszyny wirtualnej (Ubuntu + VirtualBox)

## Informacje ogólne
- **Czas:** 60–90 min (jednorazowo, najlepiej przed pierwszymi zajęciami)
- **Cel:** jednakowe, odizolowane środowisko dla wszystkich laboratoriów 1–7
- **Wynik:** maszyna wirtualna Ubuntu 24.04 LTS z Pythonem 3.12, Gitem, DVC, Dockerem, (opcjonalnie) `kind` + `kubectl`, sklonowanym repozytorium i działającym `venv`

## Dlaczego maszyna wirtualna?

| Problem bez VM | Jak rozwiązuje to VM |
|---|---|
| „U mnie działa” – różne systemy, wersje Pythona, biblioteki | Wszyscy mają to samo Ubuntu 24.04 i Python 3.12 |
| Docker / Kubernetes na Windows/macOS bywa kłopotliwy | Natywny Linux – Docker i `kind` działają bez warstw pośrednich |
| Instalacje mogą „zaśmiecić” system hosta | Wszystko w VM; snapshot pozwala wrócić do czystego stanu |
| Eksperyment zepsuł środowisko | Przywracamy snapshot w kilkanaście sekund |

```
┌──────────────────────── HOST (Windows / macOS / Linux) ───────────────────────┐
│  przeglądarka: http://localhost:5000 (MLflow), :8000 (API), :8080 (KFP UI) ... │
│        │  port forwarding (NAT)                                               │
│  ┌─────▼──────────────────── VirtualBox: VM Ubuntu 24.04 ───────────────────┐ │
│  │  venv (Python 3.12) ── MLflow, DVC, FastAPI, Evidently, pytest, kfp      │ │
│  │  Docker Engine ── kontenery API, Prometheus, Grafana                     │ │
│  │  kind (Kubernetes w Dockerze) ── Kubeflow Pipelines (lab 6, opcjonalnie) │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

> Alternatywy dla VirtualBox: **VMware Workstation Player / Fusion**, **UTM** (Mac z Apple Silicon – użyj obrazu Ubuntu **ARM64**), **Hyper-V**, **WSL2** (Windows, bez GUI – też działa, części 2–3 pomijamy), **Multipass**. Polecenia wewnątrz Ubuntu są identyczne.

---

## Część 1: Wymagania sprzętowe

| Zasób | Minimum (lab 1–5, 7) | Zalecane (z labem 6: Kubernetes + KFP) |
|---|---|---|
| RAM przydzielony VM | 4 GB | 8 GB (host ≥ 16 GB) |
| vCPU | 2 | 4 |
| Dysk (dynamiczny VDI) | 30 GB | 50 GB |
| Wirtualizacja w BIOS/UEFI | VT-x / AMD-V włączone | jw. |

Sprawdź na hoście, czy wirtualizacja jest włączona (Windows: Menedżer zadań → Wydajność → CPU → „Wirtualizacja: Włączone”). Na Windows z włączonym Hyper-V/WSL2 VirtualBox 7 działa, ale wolniej – w razie problemów rozważ WSL2.

---

## Część 2: Instalacja VirtualBox i utworzenie VM

1. Pobierz i zainstaluj **VirtualBox 7.x** oraz **Extension Pack** z https://www.virtualbox.org/wiki/Downloads.
2. Pobierz obraz **Ubuntu 24.04 LTS Desktop** (ISO, ~6 GB) z https://ubuntu.com/download/desktop (lub *Server*, jeśli wolisz pracować tylko przez SSH – zużywa mniej RAM).
3. VirtualBox → **Nowa**:
   - Nazwa: `mlops-lab`, Typ: Linux, Wersja: Ubuntu (64-bit), obraz ISO: pobrany plik,
   - zaznacz **„Pomiń instalację nienadzorowaną”** (Skip Unattended Installation), jeśli chcesz kontrolować instalację,
   - Pamięć: 8192 MB, Procesory: 4, Dysk: 50 GB (VDI, dynamicznie przydzielany).
4. **Ustawienia → System → Procesor:** włącz PAE/NX; **Ekran:** pamięć wideo 128 MB, kontroler VMSVGA.
5. **Ustawienia → Sieć → Karta 1:** NAT (domyślnie) – patrz część 4 (przekierowanie portów).
6. Uruchom VM i zainstaluj Ubuntu („Normalna instalacja”, zaznacz aktualizacje i sterowniki firm trzecich). Użytkownik np. `student`.
7. Po instalacji: menu VM **Urządzenia → Wstaw obraz płyty z dodatkami gościa** i w terminalu:

```bash
sudo apt update && sudo apt install -y build-essential dkms linux-headers-$(uname -r) bzip2
sudo /media/$USER/VBox_GAs_*/VBoxLinuxAdditions.run
sudo usermod -aG vboxsf $USER       # dostęp do folderów współdzielonych
sudo reboot
```

Dodatki gościa dają: dopasowanie rozdzielczości, współdzielony schowek (Urządzenia → Współdzielony schowek → Dwukierunkowy) i foldery współdzielone.

---

## Część 3: Snapshot „czysty system”

Zanim zaczniesz instalować narzędzia: VM wyłączona → **Maszyna → Narzędzia → Migawki → Zrób migawkę** `00-czyste-ubuntu`. Kolejną migawkę zrób po części 5 (`01-srodowisko-mlops`). Gdy coś zepsujesz – przywróć migawkę zamiast debugować godzinami.

---

## Część 4: Sieć – dostęp do usług z przeglądarki hosta

Usługi uruchamiane w laboratoriach muszą nasłuchiwać na **`0.0.0.0`** (nie tylko `127.0.0.1`), jeśli chcesz je otwierać z hosta. Masz dwie możliwości:

**A) NAT + przekierowanie portów** (Ustawienia → Sieć → Zaawansowane → Przekierowanie portów):

| Nazwa | Port hosta | Port gościa | Do czego (lab) |
|---|---|---|---|
| ssh | 2222 | 22 | praca z IDE/terminala hosta |
| mlflow | 5000 | 5000 | MLflow UI (lab 2) |
| api | 8000 | 8000 | FastAPI (lab 4, 5) |
| kfp | 8080 | 8080 | Kubeflow Pipelines UI (lab 6) |
| prometheus | 9090 | 9090 | Prometheus (lab 5) |
| grafana | 3000 | 3000 | Grafana (lab 5) |
| jupyter | 8888 | 8888 | Jupyter (opcjonalnie) |

**B) Druga karta „Sieć izolowana hosta” (Host-only)** – VM dostaje adres np. `192.168.56.101` (`ip a`), a z hosta otwierasz `http://192.168.56.101:5000` bez konfigurowania każdego portu.

Przykłady uruchamiania z dostępem z hosta:

```bash
mlflow ui --host 0.0.0.0 --port 5000
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
kubectl port-forward --address 0.0.0.0 -n kubeflow svc/ml-pipeline-ui 8080:80
```

> Najprościej jednak pracować bezpośrednio w przeglądarce **wewnątrz** VM (Ubuntu Desktop ma Firefoksa) – wtedy wystarczy `localhost`.

### Praca przez SSH (opcjonalnie, polecane)

```bash
# w VM
sudo apt install -y openssh-server && sudo systemctl enable --now ssh
# na hoście (przy przekierowaniu 2222 → 22)
ssh -p 2222 student@localhost
```

PyCharm (Professional) potrafi użyć interpretera przez SSH (*Settings → Python Interpreter → Add → On SSH*), a wersja Community – pracować na folderze współdzielonym. Dodatkowo tunel SSH zastępuje przekierowanie portów: `ssh -p 2222 -L 5000:localhost:5000 student@localhost`.

---

## Część 5: Instalacja narzędzi w Ubuntu

Wszystkie polecenia w terminalu VM. Instrukcja jest idempotentna – można ją powtórzyć.

### Krok 5.1: Pakiety systemowe, Python, Git

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget ca-certificates gnupg make unzip tree htop jq \
    python3 python3-venv python3-pip python3-dev build-essential
python3 --version     # Ubuntu 24.04 → Python 3.12.x
git --version

git config --global user.name  "Imię Nazwisko"
git config --global user.email "student@uczelnia.pl"
git config --global init.defaultBranch main
```

> Ubuntu 24.04 blokuje `pip install` do systemowego Pythona (PEP 668, błąd *externally-managed-environment*). **Zawsze używaj `venv`** – nie stosuj `--break-system-packages`.

### Krok 5.2: Repozytorium kursu i środowisko wirtualne

```bash
mkdir -p ~/mlops && cd ~/mlops
git clone <URL-repozytorium-kursu> materials-about-MLOps
cd materials-about-MLOps

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt          # kilka minut
python -c "import sklearn, mlflow, dvc, fastapi, kfp; print('OK')"
```

Aktywację można dopisać do `~/.bashrc`: `echo 'source ~/mlops/materials-about-MLOps/.venv/bin/activate' >> ~/.bashrc`.
Własne projekty z laboratoriów (np. `mlops-project/` z lab 1) twórz w `~/mlops/`, **poza** repozytorium kursu.

### Krok 5.3: Docker Engine (lab 4–7)

```bash
# oficjalne repozytorium Dockera
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# praca bez sudo (wymaga ponownego zalogowania lub `newgrp docker`)
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
docker compose version
```

### Krok 5.4: Kubernetes lokalnie – kind + kubectl (lab 6, opcjonalnie)

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl

# kind (sprawdź najnowszą wersję na https://kind.sigs.k8s.io)
curl -Lo kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
sudo install -m 0755 kind /usr/local/bin/kind && rm kind

kind create cluster --name mlops
kubectl cluster-info --context kind-mlops
kubectl get nodes
```

Na Macu z Apple Silicon (UTM, Ubuntu ARM64) zamień `amd64` na `arm64`. Instalację Kubeflow Pipelines na klastrze opisuje lab 6. Jeżeli VM ma < 8 GB RAM, w lab 6 poprzestań na **kompilacji pipeline'u do YAML** i lokalnym uruchomieniu – nie instaluj KFP na klastrze.

### Krok 5.5: Narzędzia dodatkowe (opcjonalnie)

```bash
# act – lokalne uruchamianie GitHub Actions (lab 7)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash -s -- -b /usr/local/bin
# GitHub CLI
sudo apt install -y gh
# PyCharm Community / VS Code w VM (jeśli wolisz IDE w środku)
sudo snap install pycharm-community --classic
```

### Krok 5.6: Weryfikacja środowiska

Zapisz jako `~/mlops/check_env.sh` i uruchom `bash ~/mlops/check_env.sh`:

```bash
#!/usr/bin/env bash
set -o pipefail
ok()   { printf "  ✅ %-12s %s\n" "$1" "$2"; }
fail() { printf "  ❌ %-12s %s\n" "$1" "$2"; }
check() { if out=$("${@:2}" 2>&1 | head -n1); then ok "$1" "$out"; else fail "$1" "brak / błąd"; fi; }

echo "Środowisko MLOps:"
check python   python --version
check pip      pip --version
check git      git --version
check dvc      dvc --version
check mlflow   mlflow --version
check pytest   pytest --version
check docker   docker version --format '{{.Server.Version}}'
check compose  docker compose version --short
check kubectl  kubectl version --client
check kind     kind version
echo "RAM: $(free -h | awk '/Mem:/{print $2}'), CPU: $(nproc), wolny dysk: $(df -h ~ | awk 'NR==2{print $4}')"
[[ -n "$VIRTUAL_ENV" ]] && ok venv "$VIRTUAL_ENV" || fail venv "nieaktywny – source .venv/bin/activate"
```

Następnie szybki test na skryptach z wykładów:

```bash
cd ~/mlops/materials-about-MLOps
python lectures/scripts/lecture_01/01_reproducibility.py
```

Na koniec: wyłącz VM i zrób migawkę **`01-srodowisko-mlops`**.

---

## Część 6: Mapa laboratoriów i wymagań

| Lab | Temat | Python/venv | Docker | Kubernetes (kind) | Porty |
|---|---|---|---|---|---|
| 1 | Środowisko, Git, DVC | ✅ | – | – | – |
| 2 | MLflow tracking | ✅ | – | – | 5000 |
| 3 | Walidacja danych, feature store | ✅ | – | – | – |
| 4 | REST API, serwowanie | ✅ | ✅ | – | 8000 |
| 5 | Monitoring i dryft | ✅ | ✅ (Prometheus/Grafana) | – | 8000, 9090, 3000 |
| 6 | Pipeline KFP | ✅ | ✅ | opcjonalnie | 8080 |
| 7 | CI/CD GitHub Actions | ✅ | opcjonalnie (`act`) | – | – |

Dobra praktyka: przed każdym laboratorium `cd ~/mlops/materials-about-MLOps && git pull && source .venv/bin/activate && pip install -r requirements.txt`, a po zajęciach `docker compose down` / `kind delete cluster --name mlops`, żeby zwolnić RAM.

---

## Typowe problemy i rozwiązania

| Problem | Przyczyna | Rozwiązanie |
|---|---|---|
| VirtualBox: *VT-x is disabled* / VM bardzo wolna | Wirtualizacja wyłączona w BIOS lub konflikt z Hyper-V | Włącz VT-x/AMD-V w UEFI; na Windows `bcdedit /set hypervisorlaunchtype off` lub użyj WSL2 |
| Mała rozdzielczość, brak schowka | Brak Guest Additions | Część 2, krok 7; schowek: Urządzenia → Współdzielony schowek |
| `error: externally-managed-environment` | `pip` poza venv (PEP 668) | `source .venv/bin/activate` |
| `ensurepip is not available` | Brak pakietu venv | `sudo apt install python3-venv` |
| `permission denied ... docker.sock` | Użytkownik nie w grupie `docker` | `sudo usermod -aG docker $USER`, wyloguj się/`newgrp docker` |
| Z hosta nie działa `localhost:5000` | Usługa na 127.0.0.1 lub brak przekierowania | `--host 0.0.0.0` + reguła NAT (część 4) |
| `kind create cluster` się zawiesza / pody `Pending` | Za mało RAM/CPU | Zwiększ VM do 8 GB / 4 vCPU, zamknij inne kontenery |
| Brak miejsca na dysku | Obrazy Dockera, cache pip | `docker system prune -a`, `pip cache purge`; powiększ VDI w Menedżerze nośników + `sudo growpart`/`resize2fs` (lub GParted) |
| Błędy uprawnień w folderze współdzielonym, wolne `git`/`dvc` | `vboxsf` nie wspiera dobrze symlinków i jest wolny | Trzymaj projekt na dysku VM (`~/mlops`), folder współdzielony tylko do wymiany plików |
| Zły czas w VM → błędy TLS/`git pull` | Rozjechany zegar po wstrzymaniu VM | `sudo timedatectl set-ntp true` |

## Pytania kontrolne
1. Dlaczego usługi w VM powinny nasłuchiwać na `0.0.0.0`, jeśli chcemy je otwierać z hosta?
2. Czym różni się sieć NAT z przekierowaniem portów od karty Host-only?
3. Po co tworzyć migawki VM i kiedy warto je robić?
4. Dlaczego w Ubuntu 24.04 instalujemy pakiety Pythona w `venv`, a nie globalnie?
5. Jaką rolę pełni `kind` i dlaczego wymaga Dockera?
