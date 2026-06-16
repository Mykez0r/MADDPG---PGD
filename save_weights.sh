#!/usr/bin/env bash
# save_weights.sh — Push locally trained weights (host_data/) to a remote server.
#
# Usage: bash save_weights.sh <destination_server>
# Example: bash save_weights.sh pedroamaral@10.26.110.14
#
# This lets you move trained models from one server to another without Git.
# The destination must already have the project directory.
set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "Usage: bash save_weights.sh <user@host>"
  echo "Example: bash save_weights.sh pedroamaral@10.26.110.14"
  exit 1
fi

SRC="host_data/results/main_run"

if [[ ! -f "$SRC/phase1_training_results.json" ]]; then
  echo "ERROR: $SRC/phase1_training_results.json not found. Has training completed?"
  exit 1
fi

echo "Syncing models → $DEST:~/maddpg-adversarial-experiments/host_data/results/main_run/"
rsync -av --progress \
  "$SRC/models/" \
  "$DEST:~/maddpg-adversarial-experiments/host_data/results/main_run/models/"

echo "Syncing phase1_training_results.json"
rsync -av \
  "$SRC/phase1_training_results.json" \
  "$DEST:~/maddpg-adversarial-experiments/host_data/results/main_run/"

echo ""
echo "Done. $DEST now has weights and can run Phase 2 or Phase 3."
