#!/usr/bin/env bash
set -e

echo "== Step 1: confirm no forbidden identifiers in tracked content =="
if grep -rIn "1836076\|K21194535\|CS5811\|CS5710\|Brightspace" . \
    --exclude-dir=.git \
    --exclude=.gitignore \
    --exclude=GIT_SETUP_COMMANDS.sh 2>/dev/null; then
  echo ""
  echo "FOUND forbidden strings above — stop and fix before continuing."
  exit 1
else
  echo "Clean — no forbidden identifiers found."
fi

echo ""
echo "== Step 2: initialise git and stage files =="
git init
git add .

echo ""
echo "== Step 3: confirm no raw/large dataset files were actually staged =="
staged_raw=$(git diff --cached --name-only | grep -iE "train\.csv$|test\.csv$|HPCI_Cleaned_Data\.csv$" || true)
if [ -n "$staged_raw" ]; then
  echo "FOUND a raw dataset file staged for commit:"
  echo "$staged_raw"
  exit 1
else
  echo "Clean — no raw dataset files staged."
fi

echo ""
echo "== Step 4: review status before committing =="
git status
echo ""
read -p "Type 'yes' to continue with commit + push, anything else to stop: " confirm
if [ "$confirm" != "yes" ]; then
  echo "Stopped — nothing committed."
  exit 0
fi

echo ""
echo "== Step 5: commit =="
git commit -m "Add credit-risk-ml-hpc: hybrid R (caret) + PySpark (MLlib) credit-risk classification, AUC 0.8748 (R) / 0.8460 (Spark) on 100K financial records"

echo ""
echo "== Step 6: connect to GitHub and push =="
echo "  git remote add origin https://github.com/AshishMenghani27/credit-risk-ml-hpc.git"
echo "  git branch -M main"
echo "  git push -u origin main"
