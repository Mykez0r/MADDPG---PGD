#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-maddpg_train}"
MODE="${2:-snapshot}"

if [[ "$MODE" == "follow" ]]; then
  docker logs -f "$CONTAINER_NAME"
else
  docker logs "$CONTAINER_NAME" 2>&1 | tail -n 200
fi
