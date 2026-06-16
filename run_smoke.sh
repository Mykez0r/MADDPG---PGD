#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p host_data/results/smoke_test host_logs
exec .venv/bin/python3 src/standalone_experiment_runner.py \
    --smoke \
    --variants LC-Duelling \
    --phase all \
    --results-dir host_data/results/smoke_test
