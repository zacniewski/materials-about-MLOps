#!/usr/bin/env bash
# Lecture 2, section 8 – Data versioning with DVC.
#
# Creates a temporary git+DVC repository, versions two versions of the data
# with a local "remote" and shows switching between them.
#
# Usage (from the repository root):
#     bash lectures-EN/scripts/lecture_02/05_dvc_demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
echo "📁 Demo working directory: $WORK"
cd "$WORK"

git init -q
git config user.email "demo@example.com"
git config user.name "DVC Demo"
dvc init -q
mkdir -p data /tmp/dvc-remote-demo
dvc remote add -d -f localremote /tmp/dvc-remote-demo
git add -A && git commit -qm "Init DVC"

echo "== data v1 =="
python "$SCRIPT_DIR/_make_dataset.py" data/train.csv 1000
dvc add -q data/train.csv
git add data/train.csv.dvc data/.gitignore && git commit -qm "Data v1 (1000 rows)"
git tag data-v1
dvc push -q
cat data/train.csv.dvc

echo "== data v2 =="
python "$SCRIPT_DIR/_make_dataset.py" data/train.csv 5000
dvc add -q data/train.csv
git add data/train.csv.dvc && git commit -qm "Data v2 (5000 rows)"
dvc push -q
echo "Row count (v2): $(wc -l < data/train.csv)"

echo "== Back to v1 =="
git checkout -q data-v1
dvc checkout -q
echo "Row count (v1): $(wc -l < data/train.csv)"

echo "== History =="
git log --oneline --all
echo "✅ Done. Remove the directories: rm -rf $WORK /tmp/dvc-remote-demo"
