# Lab 0: Preparing the Virtual Machine (Ubuntu + VirtualBox)

## General information
- **Duration:** 60–90 min (one-time, preferably before the first class)
- **Goal:** a uniform, isolated environment for all labs 1–7
- **Outcome:** an Ubuntu 24.04 LTS virtual machine with Python 3.12, Git, DVC, Docker, (optionally) `kind` + `kubectl`, the cloned repository and a working `venv`

## Why a virtual machine?

| Problem without a VM | How the VM solves it |
|---|---|
| "Works on my machine" – different systems, Python versions, libraries | Everyone has the same Ubuntu 24.04 and Python 3.12 |
| Docker / Kubernetes on Windows/macOS can be troublesome | Native Linux – Docker and `kind` run without intermediate layers |
| Installations can "clutter" the host system | Everything lives in the VM; a snapshot lets you return to a clean state |
| An experiment broke the environment | We restore a snapshot in a dozen or so seconds |

```
┌──────────────────────── HOST (Windows / macOS / Linux) ───────────────────────┐
│  browser: http://localhost:5000 (MLflow), :8000 (API), :8080 (KFP UI) ...      │
│        │  port forwarding (NAT)                                               │
│  ┌─────▼──────────────────── VirtualBox: VM Ubuntu 24.04 ───────────────────┐ │
│  │  venv (Python 3.12) ── MLflow, DVC, FastAPI, Evidently, pytest, kfp      │ │
│  │  Docker Engine ── API containers, Prometheus, Grafana                    │ │
│  │  kind (Kubernetes in Docker) ── Kubeflow Pipelines (lab 6, optional)     │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

> Alternatives to VirtualBox: **VMware Workstation Player / Fusion**, **UTM** (Mac with Apple Silicon – use the Ubuntu **ARM64** image), **Hyper-V**, **WSL2** (Windows, no GUI – also works, we skip parts 2–3), **Multipass**. The commands inside Ubuntu are identical.

---

## Part 1: Hardware requirements

| Resource | Minimum (labs 1–5, 7) | Recommended (with lab 6: Kubernetes + KFP) |
|---|---|---|
| RAM assigned to the VM | 4 GB | 8 GB (host ≥ 16 GB) |
| vCPU | 2 | 4 |
| Disk (dynamic VDI) | 30 GB | 50 GB |
| Virtualization in BIOS/UEFI | VT-x / AMD-V enabled | same as left |

Check on the host whether virtualization is enabled (Windows: Task Manager → Performance → CPU → "Virtualization: Enabled"). On Windows with Hyper-V/WSL2 enabled, VirtualBox 7 works, but more slowly – if you run into problems, consider WSL2.

---

## Part 2: Installing VirtualBox and creating the VM

1. Download and install **VirtualBox 7.x** and the **Extension Pack** from https://www.virtualbox.org/wiki/Downloads.
2. Download the **Ubuntu 24.04 LTS Desktop** image (ISO, ~6 GB) from https://ubuntu.com/download/desktop (or *Server*, if you prefer to work only over SSH – it uses less RAM).
3. VirtualBox → **New**:
   - Name: `mlops-lab`, Type: Linux, Version: Ubuntu (64-bit), ISO image: the downloaded file,
   - check **"Skip Unattended Installation"** if you want to control the installation,
   - Memory: 8192 MB, Processors: 4, Disk: 50 GB (VDI, dynamically allocated).
4. **Settings → System → Processor:** enable PAE/NX; **Display:** video memory 128 MB, VMSVGA controller.
5. **Settings → Network → Adapter 1:** NAT (default) – see part 4 (port forwarding).
6. Start the VM and install Ubuntu ("Normal installation", check updates and third-party drivers). User e.g. `student`.
7. After installation: VM menu **Devices → Insert Guest Additions CD image** and in the terminal:

```bash
sudo apt update && sudo apt install -y build-essential dkms linux-headers-$(uname -r) bzip2
sudo /media/$USER/VBox_GAs_*/VBoxLinuxAdditions.run
sudo usermod -aG vboxsf $USER       # access to shared folders
sudo reboot
```

Guest Additions provide: resolution auto-resize, shared clipboard (Devices → Shared Clipboard → Bidirectional) and shared folders.

---

## Part 3: "Clean system" snapshot

Before you start installing tools: with the VM powered off → **Machine → Tools → Snapshots → Take snapshot** `00-clean-ubuntu`. Take another snapshot after part 5 (`01-mlops-environment`). When you break something – restore a snapshot instead of debugging for hours.

---

## Part 4: Network – accessing services from the host browser

Services started in the labs must listen on **`0.0.0.0`** (not only `127.0.0.1`) if you want to open them from the host. You have two options:

**A) NAT + port forwarding** (Settings → Network → Advanced → Port Forwarding):

| Name | Host port | Guest port | Used for (lab) |
|---|---|---|---|
| ssh | 2222 | 22 | working from the host IDE/terminal |
| mlflow | 5000 | 5000 | MLflow UI (lab 2) |
| api | 8000 | 8000 | FastAPI (labs 4, 5) |
| kfp | 8080 | 8080 | Kubeflow Pipelines UI (lab 6) |
| prometheus | 9090 | 9090 | Prometheus (lab 5) |
| grafana | 3000 | 3000 | Grafana (lab 5) |
| jupyter | 8888 | 8888 | Jupyter (optional) |

**B) A second "Host-only Adapter"** – the VM gets an address such as `192.168.56.101` (`ip a`), and from the host you open `http://192.168.56.101:5000` without configuring each port.

Examples of starting services accessible from the host:

```bash
mlflow ui --host 0.0.0.0 --port 5000
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
kubectl port-forward --address 0.0.0.0 -n kubeflow svc/ml-pipeline-ui 8080:80
```

> The simplest option, however, is to work directly in a browser **inside** the VM (Ubuntu Desktop has Firefox) – then `localhost` is enough.

### Working over SSH (optional, recommended)

```bash
# in the VM
sudo apt install -y openssh-server && sudo systemctl enable --now ssh
# on the host (with forwarding 2222 → 22)
ssh -p 2222 student@localhost
```

PyCharm (Professional) can use an interpreter over SSH (*Settings → Python Interpreter → Add → On SSH*), while the Community edition can work on a shared folder. Additionally, an SSH tunnel replaces port forwarding: `ssh -p 2222 -L 5000:localhost:5000 student@localhost`.

---

## Part 5: Installing tools in Ubuntu

All commands are run in the VM terminal. The instructions are idempotent – they can be repeated.

### Step 5.1: System packages, Python, Git

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget ca-certificates gnupg make unzip tree htop jq \
    python3 python3-venv python3-pip python3-dev build-essential
python3 --version     # Ubuntu 24.04 → Python 3.12.x
git --version

git config --global user.name  "First Last"
git config --global user.email "student@university.edu"
git config --global init.defaultBranch main
```

> Ubuntu 24.04 blocks `pip install` into the system Python (PEP 668, *externally-managed-environment* error). **Always use a `venv`** – do not use `--break-system-packages`.

### Step 5.2: Course repository and virtual environment

```bash
mkdir -p ~/mlops && cd ~/mlops
git clone <course-repository-URL> materials-about-MLOps
cd materials-about-MLOps

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt          # a few minutes
python -c "import sklearn, mlflow, dvc, fastapi, kfp; print('OK')"
```

Activation can be added to `~/.bashrc`: `echo 'source ~/mlops/materials-about-MLOps/.venv/bin/activate' >> ~/.bashrc`.
Create your own lab projects (e.g. `mlops-project/` from lab 1) in `~/mlops/`, **outside** the course repository.

### Step 5.3: Docker Engine (labs 4–7)

```bash
# official Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# working without sudo (requires logging in again or `newgrp docker`)
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
docker compose version
```

### Step 5.4: Local Kubernetes – kind + kubectl (lab 6, optional)

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl

# kind (check the latest version at https://kind.sigs.k8s.io)
curl -Lo kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
sudo install -m 0755 kind /usr/local/bin/kind && rm kind

kind create cluster --name mlops
kubectl cluster-info --context kind-mlops
kubectl get nodes
```

On a Mac with Apple Silicon (UTM, Ubuntu ARM64) replace `amd64` with `arm64`. Installing Kubeflow Pipelines on the cluster is described in lab 6. If the VM has < 8 GB RAM, in lab 6 stop at **compiling the pipeline to YAML** and running it locally – do not install KFP on the cluster.

### Step 5.5: Additional tools (optional)

```bash
# act – running GitHub Actions locally (lab 7)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash -s -- -b /usr/local/bin
# GitHub CLI
sudo apt install -y gh
# PyCharm Community / VS Code in the VM (if you prefer an IDE inside)
sudo snap install pycharm-community --classic
```

### Step 5.6: Verifying the environment

Save as `~/mlops/check_env.sh` and run `bash ~/mlops/check_env.sh`:

```bash
#!/usr/bin/env bash
set -o pipefail
ok()   { printf "  ✅ %-12s %s\n" "$1" "$2"; }
fail() { printf "  ❌ %-12s %s\n" "$1" "$2"; }
check() { if out=$("${@:2}" 2>&1 | head -n1); then ok "$1" "$out"; else fail "$1" "missing / error"; fi; }

echo "MLOps environment:"
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
echo "RAM: $(free -h | awk '/Mem:/{print $2}'), CPU: $(nproc), free disk: $(df -h ~ | awk 'NR==2{print $4}')"
[[ -n "$VIRTUAL_ENV" ]] && ok venv "$VIRTUAL_ENV" || fail venv "inactive – source .venv/bin/activate"
```

Then a quick test using the lecture scripts:

```bash
cd ~/mlops/materials-about-MLOps
python lectures/scripts/lecture_01/01_reproducibility.py
```

Finally: power off the VM and take the snapshot **`01-mlops-environment`**.

---

## Part 6: Map of labs and requirements

| Lab | Topic | Python/venv | Docker | Kubernetes (kind) | Ports |
|---|---|---|---|---|---|
| 1 | Environment, Git, DVC | ✅ | – | – | – |
| 2 | MLflow tracking | ✅ | – | – | 5000 |
| 3 | Data validation, feature store | ✅ | – | – | – |
| 4 | REST API, serving | ✅ | ✅ | – | 8000 |
| 5 | Monitoring and drift | ✅ | ✅ (Prometheus/Grafana) | – | 8000, 9090, 3000 |
| 6 | KFP pipeline | ✅ | ✅ | optional | 8080 |
| 7 | CI/CD GitHub Actions | ✅ | optional (`act`) | – | – |

Good practice: before each lab run `cd ~/mlops/materials-about-MLOps && git pull && source .venv/bin/activate && pip install -r requirements.txt`, and after class `docker compose down` / `kind delete cluster --name mlops` to free up RAM.

---

## Common problems and solutions

| Problem | Cause | Solution |
|---|---|---|
| VirtualBox: *VT-x is disabled* / VM very slow | Virtualization disabled in BIOS or conflict with Hyper-V | Enable VT-x/AMD-V in UEFI; on Windows `bcdedit /set hypervisorlaunchtype off` or use WSL2 |
| Low resolution, no clipboard | Guest Additions missing | Part 2, step 7; clipboard: Devices → Shared Clipboard |
| `error: externally-managed-environment` | `pip` outside a venv (PEP 668) | `source .venv/bin/activate` |
| `ensurepip is not available` | venv package missing | `sudo apt install python3-venv` |
| `permission denied ... docker.sock` | User not in the `docker` group | `sudo usermod -aG docker $USER`, log out/`newgrp docker` |
| `localhost:5000` does not work from the host | Service on 127.0.0.1 or no forwarding | `--host 0.0.0.0` + NAT rule (part 4) |
| `kind create cluster` hangs / pods `Pending` | Not enough RAM/CPU | Increase the VM to 8 GB / 4 vCPU, close other containers |
| Out of disk space | Docker images, pip cache | `docker system prune -a`, `pip cache purge`; enlarge the VDI in the Media Manager + `sudo growpart`/`resize2fs` (or GParted) |
| Permission errors in the shared folder, slow `git`/`dvc` | `vboxsf` does not handle symlinks well and is slow | Keep the project on the VM disk (`~/mlops`), use the shared folder only for exchanging files |
| Wrong time in the VM → TLS/`git pull` errors | Clock drift after pausing the VM | `sudo timedatectl set-ntp true` |

## Review questions
1. Why should services in the VM listen on `0.0.0.0` if we want to open them from the host?
2. How does a NAT network with port forwarding differ from a Host-only adapter?
3. Why create VM snapshots and when is it worth taking them?
4. Why do we install Python packages in a `venv` on Ubuntu 24.04 rather than globally?
5. What role does `kind` play and why does it require Docker?
