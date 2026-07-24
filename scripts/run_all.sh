#!/usr/bin/env bash
# Runs the full pipeline: full fine-tune -> LoRA fine-tune -> evaluation.
# Usage: bash scripts/run_all.sh [path/to/config.yaml]
set -euo pipefail

CONFIG="${1:-config/config.yaml}"
cd "$(dirname "$0")/.."

echo "=== [1/3] Full fine-tuning ==="
python -m src.train_full --config "$CONFIG"

echo "=== [2/3] LoRA fine-tuning ==="
python -m src.train_lora --config "$CONFIG"

echo "=== [3/3] Evaluating base vs full vs LoRA ==="
python -m src.evaluate --config "$CONFIG"

echo "Done. See outputs/results/comparison_table.csv"
