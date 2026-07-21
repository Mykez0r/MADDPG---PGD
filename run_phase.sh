#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-train}"
VARIANTS="${2:-}"

if [[ ! "$PHASE" =~ ^(train|paper1|paper2|hotspot|failure|all)$ ]]; then
  echo "Usage: $0 <train|paper1|paper2|hotspot|failure|all> [comma-separated-variants]"
  echo "Example: $0 train CC-Duelling,LC-Duelling-GNN"
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-experiment_config.json}"
RESULTS_DIR="${RESULTS_DIR:-data/results/main_run}"
GPU_ID="${GPU_ID:-0}"
IMAGE_NAME="${IMAGE_NAME:-maddpg-exp:latest}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-maddpg}"

CONTAINER_NAME="${CONTAINER_PREFIX}_${PHASE}"
mkdir -p host_data/results host_data/models host_logs

# Keep only one container per phase to simplify monitoring.
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

CMD=(
  python src/standalone_experiment_runner.py
  --config "$CONFIG_PATH"
  --gpu "$GPU_ID"
  --phase "$PHASE"
  --results-dir "$RESULTS_DIR"
)

if [[ -n "$VARIANTS" ]]; then
  CMD+=(--variants "$VARIANTS")
fi

docker run -d --gpus all --name "$CONTAINER_NAME" \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  -v "$(pwd)/experiment_config.json":/workspace/experiment_config.json \
  -v "$(pwd)/src":/workspace/src \
  "$IMAGE_NAME" "${CMD[@]}"

echo "Started container: $CONTAINER_NAME"
echo "Use: ./check_progress.sh $CONTAINER_NAME"
