"""Sprawdzenie środowiska przed pierwszymi zajęciami.

Uruchom z katalogu głównego projektu, z aktywnym venv:

    python scripts/check_env.py

Skrypt niczego nie instaluje i niczego nie zmienia. Wypisuje [OK], [UWAGA]
albo [BŁĄD] oraz podpowiedź naprawy. Kod wyjścia 1 oznacza co najmniej jeden błąd.
"""

import importlib.metadata
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
PROJECT_FILES = ("dvc.yaml", "params.yaml", "requirements.txt")

results = {"OK": 0, "UWAGA": 0, "BŁĄD": 0}


def report(status: str, message: str, hint: str = "") -> None:
    results[status] += 1
    print(f"[{status}] {message}")
    if hint:
        print(f"        -> {hint}")


def check_python() -> None:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] >= MIN_PYTHON:
        report("OK", f"Python {version}")
    else:
        report(
            "BŁĄD",
            f"Python {version} — wymagany co najmniej {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            "Zainstaluj nowszy Python i utwórz venv od nowa.",
        )


def check_venv() -> None:
    if sys.prefix != sys.base_prefix:
        report("OK", f"Aktywny venv: {sys.prefix}")
    else:
        report(
            "UWAGA",
            "Nie wykryto aktywnego venv (pakiety mogą trafiać do systemowego Pythona).",
            "Linux/macOS: source .venv/bin/activate; Windows: .venv\\Scripts\\Activate.ps1",
        )


def check_project_dir() -> None:
    missing = [name for name in PROJECT_FILES if not Path(name).exists()]
    if missing:
        report(
            "BŁĄD",
            f"Brak plików projektu w bieżącym katalogu: {', '.join(missing)}",
            "Przejdź (cd) do katalogu mlops-project i uruchom skrypt ponownie.",
        )
    else:
        report("OK", f"Katalog projektu: {Path.cwd()}")


def read_requirements() -> list[str]:
    path = Path("requirements.txt")
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def check_packages() -> None:
    try:
        from packaging.requirements import Requirement
    except ImportError:
        report(
            "BŁĄD",
            "Brak pakietu 'packaging' — zależności nie zostały zainstalowane.",
            "python -m pip install -r requirements.txt",
        )
        return

    requirements = read_requirements()
    if not requirements:
        report("UWAGA", "Nie znaleziono requirements.txt — pomijam kontrolę pakietów.")
        return

    for line in requirements:
        requirement = Requirement(line)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            report(
                "BŁĄD",
                f"{requirement.name}: nie zainstalowano",
                "python -m pip install -r requirements.txt",
            )
            continue
        if requirement.specifier and not requirement.specifier.contains(
            installed, prereleases=True
        ):
            report(
                "BŁĄD",
                f"{requirement.name} {installed} — wymagane {requirement.specifier}",
                f"python -m pip install '{line}'",
            )
        else:
            report("OK", f"{requirement.name} {installed}")


def check_command(name: str, hint: str) -> None:
    path = shutil.which(name)
    if not path:
        report("BŁĄD", f"Polecenie '{name}' nie jest dostępne w PATH", hint)
        return
    try:
        output = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        report("BŁĄD", f"'{name} --version' nie działa: {error}", hint)
        return
    report("OK", f"{name}: {output.splitlines()[0] if output else path}")


def check_git_identity() -> None:
    if not shutil.which("git"):
        return
    missing = []
    for key in ("user.name", "user.email"):
        value = subprocess.run(
            ["git", "config", "--get", key],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not value:
            missing.append(key)
    if missing:
        report(
            "UWAGA",
            f"Git nie ma ustawionego: {', '.join(missing)} (commit się nie uda).",
            'git config --global user.name "Imię Nazwisko"; '
            'git config --global user.email "adres@example.com"',
        )
    else:
        report("OK", "Git: autor commitów skonfigurowany")


def check_shell() -> None:
    shell = os.environ.get("SHELL")
    if shell and not Path(shell).exists():
        report(
            "UWAGA",
            f"Zmienna SHELL wskazuje nieistniejący program: {shell} (DVC może zgłaszać błąd).",
            "export SHELL=/bin/bash  (albo ścieżka do istniejącej powłoki)",
        )


def check_port(port: int = 5000) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        busy = sock.connect_ex(("127.0.0.1", port)) == 0
    if busy:
        report(
            "UWAGA",
            f"Port {port} na 127.0.0.1 jest zajęty (najczęściej działa już inny serwer MLflow).",
            "Zatrzymaj stary serwer (Ctrl+C w jego terminalu) albo uruchom MLflow z --port 5001 "
            "i ustaw MLFLOW_TRACKING_URI=http://127.0.0.1:5001",
        )
    else:
        report("OK", f"Port {port} wolny dla serwera MLflow")


def main() -> int:
    print("=== Kontrola środowiska MLOps ===\n")
    check_python()
    check_venv()
    check_project_dir()
    check_packages()
    check_command("git", "Zainstaluj Git: https://git-scm.com/downloads")
    check_command("dvc", "python -m pip install -r requirements.txt (w aktywnym venv)")
    check_git_identity()
    check_shell()
    check_port()

    print(
        f"\nPodsumowanie: OK={results['OK']}, UWAGA={results['UWAGA']}, "
        f"BŁĄD={results['BŁĄD']}"
    )
    if results["BŁĄD"]:
        print("Środowisko NIE jest gotowe. Napraw pozycje [BŁĄD] i uruchom skrypt ponownie.")
        return 1
    print("Środowisko gotowe na zajęcia." + (" Przejrzyj pozycje [UWAGA]." if results["UWAGA"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
