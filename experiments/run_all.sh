#!/usr/bin/env bash
# PATHFINDER — Full Experiment Pipeline (no-cost stack)
# ======================================================
# Prerequisites:
#   pip install -r requirements.txt
#   python -m spacy download en_core_web_sm
#   export GROQ_API_KEY=your_key_here
#
# Usage:
#   bash run_all.sh              # full 7405-query HotpotQA dev run
#   bash run_all.sh --fast       # 100-query smoke test
#   bash run_all.sh --no_llm     # Recall@5 only (no Groq needed)

set -euo pipefail

FAST=false
NO_LLM=false

for arg in "$@"; do
  case $arg in
    --fast)   FAST=true ;;
    --no_llm) NO_LLM=true ;;
  esac
done

SAMPLES_FLAG=""
LLM_FLAG=""

if $FAST; then
  SAMPLES_FLAG="--max_samples 100"
  echo ">>> FAST MODE: 100 queries"
fi

if $NO_LLM; then
  LLM_FLAG="--no_llm"
  echo ">>> NO_LLM MODE: Recall@5 only"
fi

echo ""
echo "========================================"
echo " Step 1/2: Build KGs"
echo "========================================"
python 01_build_kg.py \
  --split validation \
  $SAMPLES_FLAG \
  --output data/hotpotqa_graphs.pkl

echo ""
echo "========================================"
echo " Step 2/2: Evaluate (all metrics)"
echo "========================================"
python 05_evaluate.py \
  --graphs data/hotpotqa_graphs.pkl \
  $SAMPLES_FLAG \
  $LLM_FLAG \
  --output results/results.json

echo ""
echo "========================================"
echo " Done. Results in results/results.json"
echo "========================================"
