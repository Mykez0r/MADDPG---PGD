#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-host_data/results/main_run}"

if [[ "$RESULTS_ROOT" == "/" || "$RESULTS_ROOT" == "." ]]; then
  echo "Refusing to clean unsafe path: $RESULTS_ROOT"
  exit 1
fi

mkdir -p "$RESULTS_ROOT"
find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "Cleaned: $RESULTS_ROOT"
