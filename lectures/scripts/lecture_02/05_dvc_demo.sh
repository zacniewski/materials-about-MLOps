#!/usr/bin/env bash
# Wykład 2, sekcja 8 – Wersjonowanie danych z DVC.
#
# Tworzy tymczasowe repozytorium git+DVC, wersjonuje dwie wersje danych
# z lokalnym "remote" i pokazuje przełączanie między nimi.
#
# Uruchomienie (z katalogu głównego repozytorium):
#     bash lectures/scripts/lecture_02/05_dvc_demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
echo "📁 Katalog roboczy demo: $WORK"
cd "$WORK"

git init -q
git config user.email "demo@example.com"
git config user.name "DVC Demo"
dvc init -q
mkdir -p data /tmp/dvc-remote-demo
dvc remote add -d -f localremote /tmp/dvc-remote-demo
git add -A && git commit -qm "Init DVC"

echo "== v1 danych =="
python "$SCRIPT_DIR/_make_dataset.py" data/train.csv 1000
dvc add -q data/train.csv
git add data/train.csv.dvc data/.gitignore && git commit -qm "Dane v1 (1000 wierszy)"
git tag data-v1
dvc push -q
cat data/train.csv.dvc

echo "== v2 danych =="
python "$SCRIPT_DIR/_make_dataset.py" data/train.csv 5000
dvc add -q data/train.csv
git add data/train.csv.dvc && git commit -qm "Dane v2 (5000 wierszy)"
dvc push -q
echo "Liczba wierszy (v2): $(wc -l < data/train.csv)"

echo "== Powrót do v1 =="
git checkout -q data-v1
dvc checkout -q
echo "Liczba wierszy (v1): $(wc -l < data/train.csv)"

echo "== Historia =="
git log --oneline --all
echo "✅ Gotowe. Usuń katalogi: rm -rf $WORK /tmp/dvc-remote-demo"
