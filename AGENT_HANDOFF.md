# Agent Handoff — MADDPG Adversarial Experiments

> **Purpose**: This file gives a new AI agent all context needed to continue this
> research project without any prior conversation history.
> **Last updated**: 2026-06-11 by GitHub Copilot (Claude Sonnet 4.6).

---

## 1. Project Overview

This repository implements and evaluates **hop-by-hop MADDPG routing** for a
real-topology Service Provider network. The work has two papers:

- **Paper 1** — Architecture comparison: which of 7 MADDPG variants best learns
  to route traffic, and how does it compare to the OSPF/EVPN-SP baseline?
- **Paper 2** — Adversarial robustness: how do the best variants hold up under
  FGSM observation-space attacks?

### 7 MADDPG Variants

| Name | Critic | Network | GNN |
|---|---|---|---|
| CC-Simple | Central | Simple Q | No |
| CC-Duelling | Central | Duelling Q | No |
| LC-Duelling | Local | Duelling Q | No |
| CC-Simple-GNN | Central | Simple Q | Yes |
| CC-Duelling-GNN | Central | Duelling Q | Yes |
| LC-Duelling-GNN | Local | Duelling Q | Yes |
| LC-Simple | Local | Simple Q | No |

**Baseline**: `EVPN_SP` — deterministic OSPF shortest-path routing
(implemented in `ospf_step()` in `src/maddpg_clean/network_environment.py`).

### Topology

`service_provider_real` — 86 nodes: 65 switches (S1–S65) + 21 endpoints
(BS1–7, MECS1–7, CS1–7). Link capacities: 1000 (access), 700 (aggregation),
300–500 (distribution/core). **32 trainable dist-node agents**.

### Traffic Mode

`flow` — 21 concurrent flows, each lasting 12 steps, one 0.03-unit packet per
active flow per step. Episode length = 128 steps.
These are the training-time values (May 29 run, the valid checkpoint set).

---

## 2. Three-Phase Experiment Structure

```
Phase 1 (train)   → bash run_phase.sh train
Phase 2 (paper1)  → bash run_phase.sh paper1
Phase 3 (paper2)  → bash run_phase.sh paper2
```

All phases use `experiment_config.json` (volume-mounted into Docker at runtime)
and write results to `host_data/results/main_run/`.

---

## 3. Infrastructure

### Server
- **Host**: `pedroamaral@10.26.110.14`
- **Docker image**: `maddpg-exp:latest`
- **Container naming**: `maddpg_{phase}` e.g. `maddpg_paper1`
- **Project root on server**: `~/maddpg-adversarial-experiments/`
- **Results on server**: `~/maddpg-adversarial-experiments/host_data/results/main_run/`

### Volume mounts (see `run_phase.sh`)
```
host_data/          → /workspace/data
host_logs/          → /workspace/logs
experiment_config.json → /workspace/experiment_config.json
src/                → /workspace/src
```
`src/` is live-mounted, so any local code edits take effect immediately in the
next container run without rebuilding the image.

### Useful commands
```bash
# Check progress (last 200 log lines)
ssh pedroamaral@10.26.110.14 "docker logs maddpg_paper1 2>&1 | tail -30"

# Follow logs live
ssh pedroamaral@10.26.110.14 "docker logs -f maddpg_paper1"

# Check if container finished
ssh pedroamaral@10.26.110.14 "docker inspect maddpg_paper1 --format '{{.State.Status}} {{.State.ExitCode}} {{.State.FinishedAt}}'"

# Pull results after completion
scp pedroamaral@10.26.110.14:~/maddpg-adversarial-experiments/host_data/results/main_run/phase2_maddpg_results.json /tmp/
scp pedroamaral@10.26.110.14:~/maddpg-adversarial-experiments/host_data/results/main_run/phase2_load_sweep_results.json /tmp/
scp pedroamaral@10.26.110.14:~/maddpg-adversarial-experiments/host_data/results/main_run/phase2_rankings.json /tmp/

# Run Phase 2 (clean eval)
ssh pedroamaral@10.26.110.14 "cd maddpg-adversarial-experiments && bash run_phase.sh paper1"

# Run Phase 3 (adversarial eval)
ssh pedroamaral@10.26.110.14 "cd maddpg-adversarial-experiments && bash run_phase.sh paper2"
```

---

## 4. Current State (as of 2026-06-11 ~18:00 UTC)

### Phase 1 — COMPLETED ✅
- Trained with **May 29** weights (`flow_count=21, flow_duration_steps=12`).
- 934 `.pth` files in `host_data/results/main_run/models/` (local) and on server.
- Results JSON: `host_data/results/main_run/phase1_training_results.json`.
- Training showed meaningful learning: rewards ~1.89–29.17 (positive), all 7 variants.

> **WARNING**: A June 8 training run with `flow_count=300` was broken (flat
> negative rewards ~-82). Those weights were backed up to
> `host_data/results/june8_backup/` on the server and the May 29 weights were
> restored. Do NOT use june8_backup weights.

### Phase 2 — RUNNING ⏳
- Container `maddpg_paper1` started **2026-06-11 18:00:56 UTC** on the server.
- As of last check (~18:08 UTC), loading LC-Duelling-GNN (6th of 8 variants).
- Estimated completion: ~30–60 minutes from start (depends on load sweep).
- Uses corrected config: `flow_count=21, flow_duration_steps=12`.

### Phase 3 — NOT STARTED ⬜
- Will run FGSM adversarial attacks after Phase 2 completes and results are validated.

---

## 5. Critical Bugs Fixed (do not revert)

### Bug 1: `ospf_step` drop-mode asymmetry (FIXED)
**File**: `src/maddpg_clean/network_environment.py`, function `ospf_step()`.

**Problem**: The original `ospf_step` always *buffered* packets on link
congestion (packet-mode semantics), regardless of `traffic_mode`. The MADDPG
`step()` immediately *drops* packets in flow mode. This gave EVPN_SP an
unfair advantage: 0% packet loss at all loads while MADDPG dropped 28%.

**Fix applied** (lines ~1232–1250):
```python
if self.topology.avail_bw(host, nxt) < required_bw:
    self._episode_stats['capacity_block_events'] += 1
    if self.traffic_mode == 'flow':
        # Flow mode: immediate drop — same semantics as MADDPG step().
        dropped += 1
        self._episode_stats['drop_link_congestion'] += 1
        self._episode_stats['dropped_delay_samples'].append(...)
        self._episode_stats['dropped_hop_samples'].append(pkt_hops)
    else:
        # Packet mode: buffer at current node
        if len(next_queue[host]) < MAX_NODE_QUEUE_DEPTH:
            next_queue[host].append(pkt)
        else:
            dropped += 1
            self._episode_stats['drop_overflow'] += 1
            ...
    continue
```
This fix is applied locally and is live on the server via the `src/` volume mount.

### Bug 2: Phase 2 config mismatch (FIXED)
**Problem**: `experiment_config.json` had `flow_count=300, flow_duration_steps=30`
(from a June 7 commit) while the valid May 29 models were trained with
`flow_count=21, flow_duration_steps=12`. Phase 2 ran with the wrong config,
producing invalid results (64–94% packet loss for ALL variants including EVPN_SP).

**Fix applied**: Restored correct values in `experiment_config.json`:
```json
"flow_duration_steps": 12,
"flow_count": 21,
```
Current config is correct. **Do not change these values back to 300/30.**

---

## 6. Key Files

| File | Purpose |
|---|---|
| `experiment_config.json` | All hyperparameters, variants, traffic config, load sweep |
| `src/standalone_experiment_runner.py` | Main orchestrator for all 3 phases |
| `src/maddpg_clean/network_environment.py` | Core simulator: topology, routing, `step()`, `ospf_step()` |
| `src/maddpg_clean/maddpg_implementation.py` | MADDPG agent, actor/critic networks, GNN pre-processor |
| `src/attack_framework/improved_fgsm_attack.py` | FGSM attack implementation (Phase 3) |
| `run_phase.sh` | Launches Docker container for a given phase |
| `check_progress.sh` | `docker logs <container> 2>&1 \| tail -200` |
| `save_weights.sh` | rsync local models → remote server |
| `load_weights.sh` | rsync remote server models → local |
| `host_data/results/main_run/` | All output JSONs and saved model weights |
| `tools/plot_results.py` | Generate figures from result JSONs |

---

## 7. Phase 2 Result Structure

After Phase 2 completes, results are in:
- `host_data/results/main_run/phase2_maddpg_results.json` — default eval per variant
- `host_data/results/main_run/phase2_load_sweep_results.json` — metrics across 12 load points (0.4–3.0)
- `host_data/results/main_run/phase2_rankings.json` — variant rankings

### JSON structure

```
phase2_maddpg_results.json:
  {variant_name: {'normal': {metrics}, 'dual_link_failure': {metrics}}}

phase2_load_sweep_results.json:
  {'meta': {loads, eval_episodes, ...},
   'methods': {variant: {scenario: {load_key: {metrics}}}},
   'summary': {...}}
```

### Key metrics to look for
- `mean_true_pkt_loss` — unique dropped / unique injected (primary KPI, lower is better)
- `mean_end_to_end_pdr` — % packets delivered (higher is better)
- `mean_util_p95` — 95th-percentile link utilisation
- `mean_delay_p95` — 95th-percentile end-to-end delay (steps)
- `mean_reward` — average per-agent reward (positive = learning; EVPN_SP = 0.0 always)

### Expected result direction (with correct config)
With `flow_count=21` at moderate load, MADDPG variants trained with May 29 weights
should show meaningfully lower packet loss than EVPN_SP under high load, because the
trained policy routes flows across more diverse paths. At low load (0.4–0.8) all
methods should converge to low loss since the network is not congested.

---

## 8. What To Do Next

### Immediate: wait for Phase 2 to finish
```bash
ssh pedroamaral@10.26.110.14 "docker inspect maddpg_paper1 --format '{{.State.Status}} {{.State.ExitCode}} {{.State.FinishedAt}}'"
```
When `Status = exited` and `ExitCode = 0`, pull results (see commands in §3).

### Analyse Phase 2 results
Quick analysis script:
```python
import json

with open('/path/to/phase2_load_sweep_results.json') as f:
    sw = json.load(f)

methods = sw['methods']
loads = sw['meta']['loads']
variants_ordered = ['EVPN_SP', 'LC-Duelling-GNN', 'CC-Duelling-GNN', 'LC-Duelling',
                    'CC-Simple-GNN', 'LC-Simple', 'CC-Simple', 'CC-Duelling']

print(f"{'Variant':<22}" + ''.join(f'{l:>7.1f}' for l in loads))
for v in variants_ordered:
    row = f'{v:<22}'
    for l in loads:
        lk = f'load_{l:.2f}'
        pl = methods[v]['normal'][lk].get('mean_true_pkt_loss', 0)
        row += f'{pl:>7.1f}'
    print(row)
```

### After validating Phase 2 — run Phase 3
```bash
# On server
ssh pedroamaral@10.26.110.14 "cd maddpg-adversarial-experiments && bash run_phase.sh paper2"
# Container will be named: maddpg_paper2
```
Phase 3 applies FGSM attacks to the trained models and writes:
- `host_data/results/main_run/phase3_fgsm_results.json`
- `host_data/results/main_run/phase3_rankings.json`

---

## 9. Invariants — Do Not Change

1. `experiment_config.json` traffic: `flow_count=21, flow_duration_steps=12, mode=flow`
2. `ospf_step()` must use immediate-drop semantics in flow mode (current code is correct)
3. Models in `host_data/results/main_run/models/` are the May 29 valid checkpoints — do not overwrite
4. The june8_backup weights on the server (`host_data/results/june8_backup/`) are invalid (broken training run) — never restore them

---

## 10. Repo & Weights Backup Notes

- Local machine has all 934 `.pth` files in `host_data/results/main_run/models/`
- Server (`10.26.110.14`) has a copy of the same weights
- To sync local → server: `bash save_weights.sh pedroamaral@10.26.110.14`
- To pull server → local: `bash load_weights.sh pedroamaral@10.26.110.14`
- There is **no git remote** for the model weights (too large); only the code is in git
