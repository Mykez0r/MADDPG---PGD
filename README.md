# MADDPG Routing Experiments (Clean Workflow)

This repository evaluates MADDPG routing variants on a real service-provider topology and compares them against an EVPN shortest-path baseline (`k=0` always selected).

The project is now organized around one base configuration and one generic runner script so models can be trained in separate runs, persisted, and reused later for evaluation and FGSM testing.

## What Is In Scope

- Base training/evaluation code in `src/`
- One main config file: `experiment_config.json`
- Reusable checkpoints under `host_data/results/main_run/models/`
- Phase-based runner supporting subsets of variants

## Core Goal

Train policies that:

- Avoid congestion and overload hot spots
- Carry higher offered load on the same infrastructure
- Reduce loss/backlog versus traditional shortest-path routing (`k=0`)

## Minimal Script Interface

- `./run_phase.sh <train|paper1|paper2|all> [comma-separated-variants]`
- `./check_progress.sh <container-name> [snapshot|follow]`
- `./clean_outputs.sh [results-dir]`
- `./save_weights.sh <user@host>`
- `./load_weights.sh <user@host>`

## Build

```bash
docker build -t maddpg-exp:latest .
```

## Run Phases

Default output directory inside container is `data/results/main_run`, persisted to host at `host_data/results/main_run`.

### Phase 1 (Training)

Train all configured variants:

```bash
./run_phase.sh train
./check_progress.sh maddpg_train follow
```

Train only selected variants:

```bash
./run_phase.sh train CC-Duelling,LC-Duelling-GNN
./check_progress.sh maddpg_train follow
```

Training is resumable by results file presence (`phase1_training_results.json`) and checkpoint directories.

### Phase 2 (Clean Evaluation)

```bash
./run_phase.sh paper1
./check_progress.sh maddpg_paper1 follow
```

Phase 2 consumes already-trained checkpoints from:

- `host_data/results/main_run/models/<variant>/...`

### Phase 3 (FGSM Evaluation)

```bash
./run_phase.sh paper2
./check_progress.sh maddpg_paper2 follow
```

Phase 3 also consumes the same persisted Phase 1 checkpoints.

### Full Pipeline

```bash
./run_phase.sh all
./check_progress.sh maddpg_all follow
```

## Persisting Weights Across Machines

Push weights to another server:

```bash
./save_weights.sh user@server
```

Pull weights from a server:

```bash
./load_weights.sh user@server
```

After pulling, you can directly run Phase 2 and Phase 3 without retraining.

## Results Layout

```text
host_data/
  results/
    main_run/
      phase1_training_results.json
      phase2_maddpg_results.json
      phase2_rankings.json
      phase3_fgsm_results.json
      phase3_rankings.json
      models/
        <variant>/
      figures/
host_logs/
```

## Configuration Notes

All runtime behavior is controlled by `experiment_config.json`.

Important keys:

- `training` (epochs, episodes, exploration, early stopping)
- `traffic` (flow mode and load model)
- `reward` (shared flow-level objective across all variants)
- `variants` (architectures to include)

Variant subsets for a run are selected via CLI (`--variants`), exposed through `run_phase.sh` second argument.

## Cleanup

To clear previous outputs before a fresh run:

```bash
./clean_outputs.sh
```

Or target a custom results path:

```bash
./clean_outputs.sh host_data/results/my_experiment
```
