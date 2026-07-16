"""
Standalone Experiment Runner
Three sequential phases:
  Phase 1  â€” training  (all 6 variants, hop-by-hop mode)
  Phase 2  â€” Paper-1 evaluation (architecture comparison + Shortest Path baseline)
  Phase 3  â€” Paper-2 evaluation (FGSM adversarial attack study)
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import gc
import multiprocessing as mp
import networkx as nx

try:
    from scipy.stats import kendalltau
except ImportError:
    kendalltau = None

sys.path.insert(0, 'src/maddpg_clean')
sys.path.insert(0, 'src/attack_framework')
sys.path.insert(0, 'tools')

from maddpg_implementation import MADDPG
from network_environment import NetworkEngine, NetworkEnv
from improved_fgsm_attack import FGSMAttackFramework, ThesisVisualizationSuite

try:
    from plot_results import plot_phase1_training, plot_phase2_evaluation, plot_phase3_fgsm
    PLOTTING_AVAILABLE = True
except ImportError:
    warnings.warn("Plotting utilities unavailable; plots will be skipped.")
    PLOTTING_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Per-worker-process cache: topology construction (path_cache + kpath_cache) is
# expensive (~2400 nx.shortest_simple_paths calls per NetworkEngine build).  Worker
# Pool processes are persistent across epochs, so we build the engine once per
# unique topology key and reuse it via reset() on subsequent episode calls.
_WORKER_ENV_CACHE: dict = {}


def _shared_flow_reward(info: Dict, reward_cfg: Dict) -> float:
    """Shared team reward used by every variant.

    The objective rewards delivered traffic and penalises congestion so the
    learned policy prefers load-balancing over always taking k=0 shortest-path
    routes.
    """
    delivery = float(info.get('delivery_rate', 0.0)) / 100.0
    loss = float(info.get('packet_loss_rate', 0.0)) / 100.0
    util = float(info.get('network_utilization', 0.0))
    util_var = float(info.get('util_variance', 0.0))
    backlog = float(info.get('backlog_packets', 0.0)) / 100.0

    delivery_weight = float(reward_cfg.get('delivery_weight', 1.0))
    drop_penalty = float(reward_cfg.get('drop_penalty', 1.0))
    mean_util_weight = float(reward_cfg.get('mean_util_weight', 0.5))
    var_util_penalty = float(reward_cfg.get('var_util_penalty', 0.3))
    backlog_penalty = float(reward_cfg.get('backlog_penalty', 0.0))

    return (
        delivery_weight * delivery
        - drop_penalty * loss
        - mean_util_weight * util
        - var_util_penalty * util_var
        - backlog_penalty * backlog
    )


def _episode_worker(args):
    """
    Collect one episode in a subprocess and return all transitions.

    Must be a module-level function for multiprocessing pickling.
    Workers run actors on CPU only; all gradient updates stay on the main process.

    When trainable_indices is not None the MADDPG agent count (n_agents) is a
    subset of the full topology (n_total_hosts).  The worker expands trainable
    actions to a full n_total_hosts-length list before calling env.step(), but
    only stores the trainable-agent slices in the returned transitions.
    """
    (actor_weights_cpu, all_actor_params, n_agents, deterministic_mask,
        topology_type, n_nodes, topo_seed, reward_cfg, topology_cfg,
     traffic_cfg, epsilon, decision_block_size, t_per_ep, worker_seed,
     trainable_indices, n_total_hosts, offered_load_factor,
     gnn_info_cpu, use_central_state) = args

    import os
    import sys
    import random
    import numpy as np
    import torch

    # Block GPU access in worker â€” all actor inference is CPU-only
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Ensure project modules are importable in forked/spawned workers
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maddpg_clean')
    _src_root = os.path.dirname(os.path.abspath(__file__))
    for _p in [_src, _src_root]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from maddpg_implementation import ActorNetwork, Agent, GNNProcessor
    from network_environment import NetworkEngine, NetworkEnv

    # Reconstruct GNN processor if the variant uses one
    gnn_proc = None
    if gnn_info_cpu is not None:
        obs_dim_gnn, n_agents_gnn, adjacency_gnn, n_relay_gnn, gnn_weights = gnn_info_cpu
        gnn_proc = GNNProcessor(
            obs_dim=obs_dim_gnn, hidden_dim=64,
            n_agents=n_agents_gnn, adjacency=adjacency_gnn,
            n_relay_nodes=n_relay_gnn,
        )
        sd_gnn = {k: torch.tensor(v) for k, v in gnn_weights.items()}
        gnn_proc.load_state_dict(sd_gnn, strict=False)
        gnn_proc.eval()

    # Reconstruct actor networks on CPU (explicit device='cpu' avoids CUDA in workers)
    actors: dict = {}
    for i, (actor_dim, fc1, fc2, n_actions) in enumerate(all_actor_params):
        if not deterministic_mask[i]:
            actor = ActorNetwork(
                input_dims=actor_dim, fc1_dims=fc1, fc2_dims=fc2,
                n_actions=n_actions, name=f'w_{i}', chkpt_dir='/tmp',
                device='cpu',
            )
            sd = {k: torch.tensor(v, dtype=torch.float) for k, v in actor_weights_cpu[i].items()}
            actor.load_state_dict(sd)
            actor.eval()
            actors[i] = actor

    # Build or reuse a cached NetworkEngine â€” topology is fully deterministic for
    # fixed (topology_type, n_nodes, topo_seed) so it never needs rebuilding.
    # NetworkTopology.__init__ runs ~2400 nx.shortest_simple_paths calls; caching
    # this once per worker process eliminates the reconstruction cost every epoch.
    _traffic_key = tuple(sorted((traffic_cfg or {}).items()))
    _topo_key = (topology_type, n_nodes, topo_seed, _traffic_key)
    if _topo_key not in _WORKER_ENV_CACHE:
        engine = NetworkEngine(
            topology_type=topology_type, n_nodes=n_nodes, seed=topo_seed,
            reward_config=reward_cfg,
            topology_config=topology_cfg,
            traffic_config=traffic_cfg,
        )
        _WORKER_ENV_CACHE[_topo_key] = engine
    else:
        engine = _WORKER_ENV_CACHE[_topo_key]
        # Refresh reward config in case it differs between variants
        engine.reward_cfg = {**engine.reward_cfg, **reward_cfg}
    env = NetworkEnv(engine)

    n_actions = all_actor_params[0][3]
    # If the trainable-host filter is active, env has more nodes than MADDPG agents.
    filtered = trainable_indices is not None and n_total_hosts != n_agents
    if filtered:
        trainable_set_idx = set(trainable_indices)
        # Map topology index â†’ MADDPG agent index for action lookup
        topo_to_maddpg = {topo_idx: ti for ti, topo_idx in enumerate(trainable_indices)}

    def _choose(agent_idx: int, obs) -> np.ndarray:
        if deterministic_mask[agent_idx]:
            policy = np.zeros(n_actions, dtype=np.float32)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                policy = actors[agent_idx](obs_t).squeeze(0).numpy()
        return Agent._build_executed_action(
            policy_action=policy,
            epsilon=epsilon,
            decision_block_size=decision_block_size,
            deterministic=deterministic_mask[agent_idx],
        )

    def _apply_gnn(all_obs: list) -> list:
        """Apply GNN to a list of per-agent observations. Returns enhanced list."""
        if gnn_proc is None:
            return all_obs
        return gnn_proc.process_observations([np.array(o) for o in all_obs])

    transitions = []
    env.engine.reset_with_load(offered_load_factor=offered_load_factor)
    states = [env.engine.get_state(h) for h in env.engine.get_all_hosts()]
    ep_r = 0.0
    ep_sent = 0
    ep_dropped = 0

    # Precompute trainable_hosts list for central state computation
    _trainable_hosts_local = (
        [engine.topology.hosts[i] for i in trainable_indices]
        if (use_central_state and trainable_indices is not None) else None
    )

    for t in range(t_per_ep):
        if filtered:
            # Build full n_total_hosts-length action array: trainable agents get
            # their chosen action; non-trainable nodes get a zero vector so the
            # environment falls back to deterministic shortest-path routing.
            full_executed = [np.zeros(n_actions, dtype=np.float32)
                             for _ in range(n_total_hosts)]
            maddpg_actions = []
            # Apply GNN to trainable agents' observations jointly
            trainable_obs = [states[i] for i in trainable_indices]
            enhanced_obs = _apply_gnn(trainable_obs)
            for ti, topo_idx in enumerate(trainable_indices):
                action = _choose(ti, enhanced_obs[ti])
                full_executed[topo_idx] = action
                maddpg_actions.append(action)
            # Capture central state before step() updates the environment
            cs = (engine.get_central_state(_trainable_hosts_local)
                  if use_central_state else None)
            next_states, rewards, info = env.step(full_executed,
                                                   obs_indices=trainable_indices)
            ncs = (engine.get_central_state(_trainable_hosts_local)
                   if use_central_state else None)
            done = [t == t_per_ep - 1] * n_agents
            # Store only the trainable-agent slices â€” MADDPG expects exactly
            # n_agents items in each list passed to store_transition.
            t_states = [np.array(states[i], dtype=np.float32) for i in trainable_indices]
            t_next   = [np.array(next_states[i], dtype=np.float32) for i in trainable_indices]
            # All variants optimise the same shared flow-level objective.
            # Only the critic/encoder architecture differs between LC and CC.
            r_shared = _shared_flow_reward(info, reward_cfg)
            t_rewards = [r_shared] * n_agents
            transitions.append((t_states, maddpg_actions, t_rewards, t_next, done, cs, ncs))
            ep_r += sum(t_rewards) / n_agents
        else:
            enhanced_states = _apply_gnn(states)
            executed_actions = [_choose(i, enhanced_states[i]) for i in range(n_agents)]
            next_states, rewards, info = env.step(executed_actions)
            done = [t == t_per_ep - 1] * n_agents
            r_shared = _shared_flow_reward(info, reward_cfg)
            transitions.append((
                [np.array(s, dtype=np.float32) for s in states],
                [np.array(a, dtype=np.float32) for a in executed_actions],
                [r_shared] * n_agents,
                [np.array(s, dtype=np.float32) for s in next_states],
                list(done),
                None, None,
            ))
            ep_r += r_shared
        states = next_states
        ep_sent += info.get('packets_sent', 0)
        ep_dropped += info.get('packets_dropped', 0)

    return transitions, ep_r, ep_sent, ep_dropped


class StandaloneExperimentRunner:

    def __init__(self, config_path: str, gpu_id: int = 0,
                 results_dir: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.gpu_id = gpu_id
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = results_dir or f"data/results/exp_{ts}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.attack_framework = FGSMAttackFramework()
        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Results â†’ {self.results_dir}")

    # â”€â”€ config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_config(self, path: str) -> Dict:
        with open(path) as f:
            cfg = json.load(f)
        cfg.setdefault('training', {'epochs': 200, 'episodes_per_epoch': 5,
                                    'timesteps_per_episode': 256})
        cfg.setdefault('fgsm_slo', {
            'max_pkt_loss_pct': 10.0,
            'min_delivery_rate_pct': 85.0,
            'max_reward_degradation_pct': 25.0,
            'max_delay_p95': 12.0,
        })
        cfg.setdefault('load_sweep', {
            'enabled': False,
            'offered_loads': [0.75, 1.00, 1.25, 1.50],
            'include_failures': False,
        })
        cfg.setdefault('traffic', {
            'mode': 'packet',
            'packet_size': 0.03,
            'flow_packet_size': 0.03,
            'flow_hold_steps': 12,
        })
        cfg.setdefault('clean_slo', {
            'max_pkt_loss_pct': 5.0,
            'max_delay_p95': 8.0,
            'min_end_to_end_pdr_pct': 85.0,
        })
        runtime_cfg = cfg.setdefault('runtime_control', {})
        runtime_cfg.setdefault('profile', 'standard')
        runtime_cfg.setdefault('max_attack_cases_per_variant', -1)
        runtime_cfg.setdefault('max_attack_variants', -1)
        runtime_cfg.setdefault('phase3_enable_slo_pruning', True)
        runtime_cfg.setdefault('phase3_consecutive_fail_limit', 2)
        runtime_cfg.setdefault('phase3_skip_after_critical_epsilon', True)
        seed_cfg = runtime_cfg.setdefault('seed_expansion', {})
        seed_cfg.setdefault('enable_adaptive', True)
        seed_cfg.setdefault('initial_seeds', 10)
        seed_cfg.setdefault('max_seeds_for_ranking', 50)
        seed_cfg.setdefault('rank_stability_check_interval', 5)
        seed_cfg.setdefault('top_k_for_stability', 5)
        seed_cfg.setdefault('stability_threshold', 0.95)
        runtime_cfg.setdefault('profiles', {
            'standard': {},
            'thorough': {},
            'quick': {
                'training': {'epochs': 1, 'episodes_per_epoch': 1, 'timesteps_per_episode': 10},
                'paper1_eval': {'evaluation_episodes': 2},
                'attack_eval_episodes': 2,
            },
        })
        self._apply_runtime_profile(cfg)
        return cfg

    @staticmethod
    def _apply_runtime_profile(cfg: Dict):
        runtime_cfg = cfg.get('runtime_control', {})
        profile_name = runtime_cfg.get('profile', 'standard')
        profiles = runtime_cfg.get('profiles', {})
        profile = profiles.get(profile_name, {})
        if not profile:
            return

        if 'training' in profile:
            cfg.setdefault('training', {}).update(profile['training'])
        if 'paper1_eval' in profile:
            cfg.setdefault('paper1_eval', {}).update(profile['paper1_eval'])

        attack_eval_eps = profile.get('attack_eval_episodes')
        if attack_eval_eps is not None:
            for attack_cfg in cfg.get('attack_configs', []):
                attack_cfg['evaluation_episodes'] = int(attack_eval_eps)

    # â”€â”€ factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _make_variant(self, vcfg: Dict) -> Tuple[MADDPG, NetworkEngine, NetworkEnv]:
        reward_cfg = self.config.get('reward', {})
        training_cfg = self.config.get('training', {})
        topology_cfg = self.config.get('topology', {})
        traffic_cfg = self.config.get('traffic', {})
        projection_cfg = training_cfg.get('learn_action_projection', {})
        engine = NetworkEngine(
            topology_type=topology_cfg.get('type', 'service_provider'),
            n_nodes=int(topology_cfg.get('nodes', vcfg.get('n_agents', 65))),
            reward_config=reward_cfg,
            topology_config=topology_cfg,
            traffic_config=traffic_cfg,
        )
        env = NetworkEnv(engine)

        # â”€â”€ Trainable-host filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Reduces the RL agent set from all topology nodes to a meaningful subset
        # (e.g. only switches with degree â‰¥ 3) to shrink critic dimensions and
        # speed up training without losing routing expressiveness.
        filter_mode = self.config.get('training', {}).get('trainable_host_filter', 'all')
        trainable_hosts = engine.get_trainable_hosts(filter_mode)
        all_hosts = engine.get_all_hosts()
        host_to_idx = {h: i for i, h in enumerate(all_hosts)}
        trainable_indices = [host_to_idx[h] for h in trainable_hosts]
        # Attach to engine so training/eval helpers can access without re-computing
        engine.trainable_host_indices = trainable_indices
        engine.n_total_hosts = len(all_hosts)

        critic_domain = vcfg['critic_domain']
        n_agents = len(trainable_hosts)
        actor_dims = engine.state_dims
        n_actions = engine.n_actions

        # Neighborhood critic: adjacency built within the trainable sub-graph
        if critic_domain == 'neighborhood_critic':
            trainable_set = set(trainable_hosts)
            trainable_idx_map = {h: i for i, h in enumerate(trainable_hosts)}
            adjacency = [
                [trainable_idx_map[nb]
                 for nb in engine.topology.get_neighbors(h)
                 if nb in trainable_set]
                for h in trainable_hosts
            ]
            critic_dims = [
                actor_dims * (1 + len(adjacency[i]))
                for i in range(n_agents)
            ]
            central_state_dims = None
        elif critic_domain == 'local_critic':
            adjacency = None
            critic_dims = actor_dims
            central_state_dims = None
        else:
            adjacency = None
            # CC critic uses compact <B, D> central state instead of all local obs.
            # central_state_dims = |E| + n_trainable_agents = 106 + 32 = 138.
            _n_edges = engine.topology.graph.number_of_edges()
            central_state_dims = _n_edges + n_agents
            critic_dims = central_state_dims

        # GNN adjacency: built over the full topology when trainable hosts are
        # a strict subset (e.g. PE-router / access-nodes model).  Switch nodes
        # become relay nodes that participate in message passing but produce no
        # policy output â€” two-hop communication between access nodes via shared
        # upstream switches emerges from the graph structure alone.
        if vcfg.get('use_gnn', False):
            relay_nodes = [h for h in all_hosts if h not in set(trainable_hosts)]
            gnn_n_relay = len(relay_nodes)
            if gnn_n_relay > 0:
                # Full-graph adjacency: agents indexed 0..n_agents-1,
                # relay (switch) nodes indexed n_agents..n_total-1
                full_node_order = trainable_hosts + relay_nodes
                full_node_idx = {h: i for i, h in enumerate(full_node_order)}
                gnn_adjacency = [
                    [full_node_idx[nb]
                     for nb in engine.topology.get_neighbors(h)
                     if nb in full_node_idx]
                    for h in full_node_order
                ]
            else:
                # Legacy: trainable-subgraph adjacency only
                trainable_set_gnn = set(trainable_hosts)
                trainable_idx_map_gnn = {h: i for i, h in enumerate(trainable_hosts)}
                gnn_adjacency = [
                    [trainable_idx_map_gnn[nb]
                     for nb in engine.topology.get_neighbors(h)
                     if nb in trainable_set_gnn]
                    for h in trainable_hosts
                ]
        else:
            gnn_adjacency = None
            gnn_n_relay = 0

        maddpg = MADDPG(
            actor_dims=actor_dims,
            critic_dims=critic_dims,
            n_agents=n_agents,
            n_actions=n_actions,
            chkpt_dir=f"{self.results_dir}/models/{vcfg['name']}",
            critic_type=critic_domain,
            network_type=vcfg['neural_network'],
            fc1=vcfg.get('fc1', 256),
            fc2=vcfg.get('fc2', 128),
            alpha=vcfg.get('alpha', 0.001),
            beta=vcfg.get('beta', 0.001),
            gamma=vcfg.get('gamma', training_cfg.get('gamma', 0.95)),
            tau=vcfg.get('tau', training_cfg.get('tau', 0.01)),
            use_gnn=vcfg.get('use_gnn', False),
            critic_target_mode=projection_cfg.get('critic_target_mode', 'block_argmax_onehot'),
            actor_mode=projection_cfg.get('actor_mode', 'soft'),
            adjacency=adjacency,
            gnn_adjacency=gnn_adjacency,
            gnn_n_relay=gnn_n_relay,
            central_state_dims=central_state_dims,
        )
        return maddpg, engine, env

    def _load_variant_checkpoint(self, maddpg: MADDPG, name: str):
        if maddpg.load_best_checkpoint():
            logger.info(f"[CKPT] {name} â€” loaded best checkpoint")
        else:
            maddpg.load_checkpoint()
            logger.info(f"[CKPT] {name} â€” loaded final checkpoint (best not found)")

    @staticmethod
    def _build_full_actions(
        maddpg_actions: List[np.ndarray],
        n_total_hosts: int,
        trainable_indices: List[int],
        n_actions: int,
    ) -> List[np.ndarray]:
        """Expand 32 trainable-agent actions to a full 86-length action list.

        Non-trainable nodes receive a zero vector (deterministic pass-through
        behaviour: their packets follow the shortest-path next-hop already
        baked into the environment's step logic via kpath_cache fallbacks).
        """
        full = [np.zeros(n_actions, dtype=np.float32) for _ in range(n_total_hosts)]
        for ti, topo_idx in enumerate(trainable_indices):
            full[topo_idx] = np.asarray(maddpg_actions[ti], dtype=np.float32)
        return full

    @staticmethod
    def _metric_value(metric: str, mean_reward: float, mean_pkt_loss: float) -> float:
        if metric == 'mean_reward':
            return mean_reward
        if metric == 'composite':
            return mean_reward - mean_pkt_loss
        return mean_pkt_loss

    @staticmethod
    def _is_improvement(metric: str, candidate: float, best: Optional[float],
                        min_improvement: float) -> bool:
        if best is None:
            return True
        if metric in ('mean_reward',):
            return (candidate - best) > min_improvement
        return (best - candidate) > min_improvement

    def _validate_variant_inline(self, maddpg: MADDPG, env: NetworkEnv,
                                 n_eps: int, t_per_ep: int,
                                 engine=None) -> Tuple[float, float]:
        val_rewards, val_losses = [], []
        trainable_indices = getattr(engine, 'trainable_host_indices', None)
        n_total_hosts = getattr(engine, 'n_total_hosts', maddpg.n_agents)
        n_actions = maddpg.n_actions
        for _ in range(max(1, n_eps)):
            states = env.reset()
            ep_r = ep_sent = ep_dropped = 0
            for _ in range(t_per_ep):
                if trainable_indices is not None:
                    t_states = [states[i] for i in trainable_indices]
                    t_actions = maddpg.choose_action(t_states)
                    actions = self._build_full_actions(t_actions, n_total_hosts,
                                                      trainable_indices, n_actions)
                else:
                    actions = maddpg.choose_action(states)
                next_states, rewards, info = env.step(actions)
                states = next_states
                ep_r += sum(rewards)
                ep_sent += info.get('packets_sent', 0)
                ep_dropped += info.get('packets_dropped', 0)
            val_rewards.append(ep_r / len(maddpg.agents))
            val_losses.append(ep_dropped / max(1, ep_sent) * 100)

        return float(np.mean(val_rewards)), float(np.mean(val_losses))

    def _collect_episodes_parallel(
        self, pool, maddpg: MADDPG, engine,
        n_episodes: int, t_per_ep: int, epsilon: float,
        deterministic_mask: List[bool], decision_block_size: int,
        epoch: int, base_seed: int = 42,
    ) -> Tuple[List[float], List[float]]:
        """Collect n_episodes in parallel workers; push transitions to replay buffer."""
        actor_weights_cpu = maddpg.get_actor_weights_cpu()
        gnn_info_cpu = maddpg.get_gnn_info_cpu() if maddpg.gnn_processor is not None else None
        all_actor_params = [
            (a.actor.input_dims, a.actor.fc1_dims, a.actor.fc2_dims, a.actor.n_actions)
            for a in maddpg.agents
        ]
        trainable_indices = getattr(engine, 'trainable_host_indices', None)
        n_total_hosts = getattr(engine, 'n_total_hosts', maddpg.n_agents)
        train_load = float(self.config.get('training', {}).get('offered_load_factor', 1.0))
        traffic_cfg = self.config.get('traffic', {})
        worker_args = [
            (
                actor_weights_cpu, all_actor_params, maddpg.n_agents, deterministic_mask,
                engine.topology_type, engine.n_nodes, engine.topo_seed, engine.reward_cfg,
            self.config.get('topology', {}), traffic_cfg,
                epsilon, decision_block_size, t_per_ep,
                base_seed + epoch * 10_000 + ep_idx,
                trainable_indices, n_total_hosts, train_load,
                gnn_info_cpu,
                maddpg.memory.central_state_memory is not None,  # use_central_state
            )
            for ep_idx in range(n_episodes)
        ]
        results = pool.map(_episode_worker, worker_args)
        ep_rewards, ep_losses = [], []
        for transitions, ep_r, ep_sent, ep_dropped in results:
            for step in transitions:
                maddpg.store_transition(*step)
            ep_rewards.append(ep_r)
            ep_losses.append(ep_dropped / max(1, ep_sent) * 100)
        return ep_rewards, ep_losses

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 1 â€” Training
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def train_variant(self, vcfg: Dict) -> Dict:
        name = vcfg['name']
        logger.info(f"[TRAIN] {name} â€” start")

        # Always use 'spawn' for the worker pool.  'fork' after CUDA has been
        # initialised in the parent (which happens after the first variant
        # completes) corrupts the CUDA context in every child and causes
        # pool.map() to deadlock permanently.  'spawn' starts fresh Python
        # interpreters that have no CUDA state; workers also explicitly set
        # CUDA_VISIBLE_DEVICES='' as an extra guard.
        cfg_t      = self.config['training']
        eps_per_ep = cfg_t['episodes_per_epoch']
        n_workers_cfg = cfg_t.get('parallel_workers', 0)
        n_workers = n_workers_cfg if n_workers_cfg > 0 else min(eps_per_ep, max(1, mp.cpu_count() - 1))
        _mp_ctx = mp.get_context('spawn')
        worker_pool = _mp_ctx.Pool(processes=n_workers)
        logger.info("[TRAIN] %s â€” using %d parallel episode workers (start=spawn)",
                    name, n_workers)

        maddpg, engine, env = self._make_variant(vcfg)

        epochs     = cfg_t['epochs']
        t_per_ep   = cfg_t['timesteps_per_episode']
        batch_size = cfg_t.get('batch_size', 256)
        exploration_cfg = cfg_t.get('exploration', {})
        best_ckpt_cfg = cfg_t.get('best_checkpoint', {})
        early_stop_cfg = cfg_t.get('early_stopping', {})
        exploration_enabled = exploration_cfg.get('enabled', False)
        epsilon_start = float(exploration_cfg.get('epsilon_start', 0.0))
        epsilon_end = float(exploration_cfg.get('epsilon_end', epsilon_start))
        epsilon_decay_epochs = max(1, int(exploration_cfg.get('epsilon_decay_epochs', 1)))
        decision_block_size = int(exploration_cfg.get('decision_block_size', 1))
        freeze_single_link_nodes = bool(cfg_t.get('freeze_single_link_nodes', False))

        best_ckpt_enabled = bool(best_ckpt_cfg.get('enabled', False))
        best_metric_name = str(best_ckpt_cfg.get('metric', 'mean_pkt_loss'))
        validation_interval_epochs = max(1, int(best_ckpt_cfg.get('validation_interval_epochs', 20)))
        validation_episodes = max(1, int(best_ckpt_cfg.get('validation_episodes', 2)))
        min_improvement = float(best_ckpt_cfg.get('min_improvement', 0.0))
        warmup_epochs = max(0, int(best_ckpt_cfg.get('warmup_epochs', 0)))

        early_stop_enabled = bool(early_stop_cfg.get('enabled', False))
        early_stop_metric = str(early_stop_cfg.get('metric', best_metric_name))
        patience_checks = max(1, int(early_stop_cfg.get('patience_checks', 8)))
        smooth_window_checks = max(1, int(early_stop_cfg.get('smooth_window_checks', 3)))
        min_epochs = max(1, int(early_stop_cfg.get('min_epochs', validation_interval_epochs)))

        hosts = engine.get_all_hosts()
        trainable_indices = getattr(engine, 'trainable_host_indices', None)
        if trainable_indices is not None:
            # deterministic_mask is scoped to the n_trainable MADDPG agents only
            trainable_hosts_list = [hosts[i] for i in trainable_indices]
            deterministic_mask = (
                [engine.get_number_neighbors(h) <= 1 for h in trainable_hosts_list]
                if freeze_single_link_nodes
                else [False] * len(trainable_indices)
            )
        else:
            deterministic_mask = (
                [engine.get_number_neighbors(host) <= 1 for host in hosts]
                if freeze_single_link_nodes
                else [False] * len(hosts)
            )

        all_rewards, all_losses = [], []
        validation_trace = []
        best_metric = None
        best_epoch = -1
        checks_without_improvement = 0
        metric_window = deque(maxlen=smooth_window_checks)
        stop_metric_best = None
        stop_metric_window = deque(maxlen=smooth_window_checks)
        early_stopped = False
        stop_epoch = epochs - 1

        if freeze_single_link_nodes:
            logger.info(
                "[TRAIN] %s â€” freezing %d/%d deterministic nodes",
                name,
                sum(deterministic_mask),
                len(deterministic_mask),
            )

        for epoch in range(epochs):
            if exploration_enabled:
                decay_progress = min(epoch, epsilon_decay_epochs) / epsilon_decay_epochs
                epoch_epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay_progress
            else:
                epoch_epsilon = 0.0

            # Collect all episodes in parallel, then do equivalent gradient updates
            ep_rewards, ep_losses = self._collect_episodes_parallel(
                pool=worker_pool, maddpg=maddpg, engine=engine,
                n_episodes=eps_per_ep, t_per_ep=t_per_ep,
                epsilon=epoch_epsilon, deterministic_mask=deterministic_mask,
                decision_block_size=decision_block_size,
                epoch=epoch, base_seed=42,
            )
            freq = int(cfg_t.get('learn_freq', 25))
            n_learns = (t_per_ep // freq) * eps_per_ep
            for _ in range(n_learns):
                maddpg.learn(
                    batch_size=batch_size,
                    deterministic_mask=deterministic_mask,
                    decision_block_size=decision_block_size,
                )

            all_rewards.extend(ep_rewards)
            all_losses.extend(ep_losses)

            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(
                    f"[TRAIN] {name}  epoch {epoch:3d}  "
                    f"reward={np.mean(ep_rewards):7.2f}  "
                    f"pkt_loss={np.mean(ep_losses):5.2f}%"
                )

            should_validate = (
                (best_ckpt_enabled or early_stop_enabled)
                and epoch >= warmup_epochs
                and (epoch % validation_interval_epochs == 0 or epoch == epochs - 1)
            )

            if should_validate:
                val_reward, val_pkt_loss = self._validate_variant_inline(
                    maddpg=maddpg,
                    env=env,
                    n_eps=validation_episodes,
                    t_per_ep=t_per_ep,
                    engine=engine,
                )
                raw_metric = self._metric_value(best_metric_name, val_reward, val_pkt_loss)
                metric_window.append(raw_metric)
                smoothed_metric = float(np.mean(metric_window))
                validation_trace.append({
                    'epoch': int(epoch),
                    'mean_reward': val_reward,
                    'mean_pkt_loss': val_pkt_loss,
                    'metric_raw': raw_metric,
                    'metric_smoothed': smoothed_metric,
                })

                improved = best_ckpt_enabled and self._is_improvement(
                    metric=best_metric_name,
                    candidate=smoothed_metric,
                    best=best_metric,
                    min_improvement=min_improvement,
                )

                if improved and best_ckpt_enabled:
                    best_metric = smoothed_metric
                    best_epoch = epoch
                    maddpg.save_best_checkpoint()
                    logger.info(
                        "[TRAIN] %s â€” new best checkpoint at epoch %d (%s=%.4f)",
                        name,
                        epoch,
                        best_metric_name,
                        smoothed_metric,
                    )

                stop_raw_metric = self._metric_value(early_stop_metric, val_reward, val_pkt_loss)
                stop_metric_window.append(stop_raw_metric)
                stop_smoothed_metric = float(np.mean(stop_metric_window))
                if self._is_improvement(
                    metric=early_stop_metric,
                    candidate=stop_smoothed_metric,
                    best=stop_metric_best,
                    min_improvement=min_improvement,
                ):
                    stop_metric_best = stop_smoothed_metric
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1

                if early_stop_enabled and epoch >= min_epochs and checks_without_improvement >= patience_checks:
                    early_stopped = True
                    stop_epoch = epoch
                    logger.info(
                        "[TRAIN] %s â€” early stopping at epoch %d after %d checks without improvement",
                        name,
                        epoch,
                        checks_without_improvement,
                    )
                    break

        worker_pool.close()
        worker_pool.join()

        maddpg.save_checkpoint()
        logger.info(f"[TRAIN] {name} â€” done")
        return {
            'name':           name,
            'model_dir':      f"{self.results_dir}/models/{name}",
            'rewards':        all_rewards,
            'pkt_losses':     all_losses,
            'final_reward':   float(np.mean(all_rewards[-50:])),
            'final_pkt_loss': float(np.mean(all_losses[-50:])),
            'best_checkpoint_enabled': best_ckpt_enabled,
            'best_checkpoint_metric_name': best_metric_name,
            'best_checkpoint_metric': best_metric,
            'best_checkpoint_epoch': int(best_epoch),
            'validation_trace': validation_trace,
            'stopped_early': early_stopped,
            'stop_epoch': int(stop_epoch),
        }

    def run_training(self) -> Dict:
        logger.info("â•â•â•â•â•â• PHASE 1 â€” TRAINING â•â•â•â•â•â•")
        # Resume: load existing results so already-trained variants are skipped
        try:
            results = self._load('phase1_training_results.json')
            logger.info("[TRAIN] Resuming â€” found existing results for: %s", list(results.keys()))
        except Exception:
            results = {}
        for vcfg in self.config['variants']:
            if vcfg['name'] in results:
                logger.info("[TRAIN] %s â€” already trained, skipping", vcfg['name'])
                continue
            try:
                results[vcfg['name']] = self.train_variant(vcfg)
            except Exception as exc:
                logger.error(f"[TRAIN] {vcfg['name']} FAILED: {exc}")
            # Save incrementally so a crash doesn't lose completed variants
            self._save(results, 'phase1_training_results.json')
        self._save(results, 'phase1_training_results.json')
        self._generate_plots_for_phase(1)
        return results

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 2 â€” MADDPG evaluation
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def evaluate_maddpg(self, training_results: Optional[Dict] = None) -> Dict:
        logger.info("â•â•â•â•â•â• PHASE 2 â€” MADDPG EVALUATION â•â•â•â•â•â•")
        if training_results is None:
            training_results = self._load('phase1_training_results.json')

        seed_cfg = self.config.get('runtime_control', {}).get('seed_expansion', {})
        adaptive_enabled = seed_cfg.get('enable_adaptive', True)
        initial_seeds = seed_cfg.get('initial_seeds', 10)
        max_seeds = seed_cfg.get('max_seeds_for_ranking', 50)
        check_interval = seed_cfg.get('rank_stability_check_interval', 5)
        top_k = seed_cfg.get('top_k_for_stability', 5)
        stability_thresh = seed_cfg.get('stability_threshold', 0.95)

        base_eval_eps = self.config.get('paper1_eval', {}).get('evaluation_episodes', 30)
        t_per_ep = self.config['training']['timesteps_per_episode']

        convergence_history = {}
        curr_seeds = initial_seeds if adaptive_enabled else base_eval_eps
        prev_ranking = None
        stability_achieved = False
        convergence_seed_n = -1

        if adaptive_enabled:
            logger.info(f"[P2] Adaptive seed expansion: init={initial_seeds}, max={max_seeds}, "
                       f"interval={check_interval}, top_k={top_k}, thresh={stability_thresh:.2f}")

        # Iteratively expand seeds until stable or hit max
        while curr_seeds <= max_seeds:
            logger.info(f"[P2] Running evaluation with {curr_seeds} seeds per scenario")
            
            results = {}
            
            # MADDPG variants
            for vcfg in self.config['variants']:
                name = vcfg['name']
                if name not in training_results:
                    if name not in convergence_history:
                        logger.warning(f"[P2] {name} â€” no trained model, skipping")
                    continue
                
                logger.info(f"[P2]   {name}")
                maddpg, engine, env = self._make_variant(vcfg)
                self._load_variant_checkpoint(maddpg, name)
                
                # Run evaluation episodes
                normal   = self._run_eval_episodes(maddpg, env, curr_seeds, t_per_ep,
                                                   n_link_failures=0)
                failures = self._run_eval_episodes(maddpg, env, curr_seeds, t_per_ep,
                                                   n_link_failures=2)
                
                results[name] = {'normal': normal, 'dual_link_failure': failures}

            # EVPN shortest-path baseline (same tunnel model as MADDPG, k=0 always)
            if len(results) > 0:  # Only run if at least one variant ran
                logger.info(f"[P2]   EVPN-SP baseline")
                _evpn_engine = NetworkEngine(
                    topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
                    n_nodes=65,
                    traffic_config=self.config.get('traffic', {}),
                )
                results['EVPN_SP'] = {
                    'normal': self._run_evpn_sp_episodes(_evpn_engine, curr_seeds, t_per_ep),
                    'dual_link_failure': self._run_evpn_sp_episodes(_evpn_engine, curr_seeds, t_per_ep,
                                                                     n_link_failures=2),
                }

            # Build rankings
            ranking = self._build_phase2_rankings(results)
            
            # Check ranking stability if adaptive enabled
            if adaptive_enabled and prev_ranking is not None:
                # Compare top-K rankings across scenarios
                tau_scores = []
                for scenario in ['normal', 'dual_link_failure']:
                    for profile_name in ['default']:  # Can be extended to multiple profiles
                        if scenario in ranking.get('scenarios', {}) \
                           and profile_name in ranking['scenarios'][scenario]:
                            curr_ranked = ranking['scenarios'][scenario][profile_name]
                            prev_ranked = prev_ranking.get('scenarios', {}).get(scenario, {}).get(profile_name, [])
                            if prev_ranked and curr_ranked:
                                tau = self._compute_rank_stability(prev_ranked, curr_ranked, top_k)
                                tau_scores.append(tau)
                
                if tau_scores:
                    mean_tau = float(np.mean(tau_scores))
                    convergence_history[curr_seeds] = mean_tau
                    logger.info(f"[P2]   Rank stability (top-{top_k}): tau={mean_tau:.3f}")
                    
                    if mean_tau >= stability_thresh:
                        logger.info(f"[P2] âœ“ Rank stability achieved at {curr_seeds} seeds "
                                   f"(tau={mean_tau:.3f} >= {stability_thresh:.2f})")
                        stability_achieved = True
                        convergence_seed_n = curr_seeds
                        break
            
            prev_ranking = ranking
            
            # Expand seeds for next iteration
            if adaptive_enabled and curr_seeds < max_seeds:
                next_seeds = min(curr_seeds + check_interval, max_seeds)
                if next_seeds > curr_seeds:
                    curr_seeds = next_seeds
                else:
                    break
            else:
                break

        # Final results with ONE seed count (use largest evaluated)
        logger.info(f"[P2] Finalizing results with {curr_seeds} seeds")
        results = {}
        
        for vcfg in self.config['variants']:
            name = vcfg['name']
            if name not in training_results:
                logger.warning(f"[P2] {name} â€” no trained model, skipping")
                continue
            logger.info(f"[P2]   {name}")
            maddpg, engine, env = self._make_variant(vcfg)
            self._load_variant_checkpoint(maddpg, name)
            
            normal   = self._run_eval_episodes(maddpg, env, curr_seeds, t_per_ep)
            failures = self._run_eval_episodes(maddpg, env, curr_seeds, t_per_ep, n_link_failures=2)
            
            results[name] = {'normal': normal, 'dual_link_failure': failures}

        logger.info("[P2] EVPN-SP baseline")
        _evpn_engine = NetworkEngine(
            topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
            n_nodes=65,
            traffic_config=self.config.get('traffic', {}),
        )
        results['EVPN_SP'] = {
            'normal': self._run_evpn_sp_episodes(_evpn_engine, curr_seeds, t_per_ep),
            'dual_link_failure': self._run_evpn_sp_episodes(_evpn_engine, curr_seeds, t_per_ep,
                                                             n_link_failures=2),
        }

        self._save(results, 'phase2_maddpg_results.json')
        ranking = self._build_phase2_rankings(results)
        
        # Add convergence telemetry to ranking output
        ranking['seed_convergence'] = {
            'total_seeds_used': curr_seeds,
            'stability_achieved_at_seed_n': convergence_seed_n if stability_achieved else -1,
            'stability_threshold': stability_thresh,
            'stability_history': convergence_history,
            'top_k_for_stability': top_k,
        }

        load_cfg = self.config.get('load_sweep', {})
        if bool(load_cfg.get('enabled', False)):
            logger.info("[P2] Running offered-load sweep for fairness analysis")
            load_sweep_results = self._run_phase2_load_sweep(training_results, curr_seeds, t_per_ep)
            self._save(load_sweep_results, 'phase2_load_sweep_results.json')
            ranking['load_sweep_summary'] = load_sweep_results.get('summary', {})
        
        self._save(ranking, 'phase2_rankings.json')
        self._generate_plots_for_phase(2)
        logger.info("[P2] evaluation complete")
        return results

    def _run_eval_episodes(self, maddpg: MADDPG, env: NetworkEnv,
                           n_eps: int, t_per_ep: int,
                           n_link_failures: int = 0,
                           offered_load_factor: float = 1.0) -> Dict:
        ep_rewards, ep_utils = [], []
        ep_pdr, ep_resolved_pdr, ep_true_loss, ep_hop_frac, ep_goodput = [], [], [], [], []
        ep_delay_p95, ep_backlog, ep_util_p95 = [], [], []
        ep_hops_mean, ep_overload_frac = [], []
        ep_drop_ttl, ep_drop_overflow, ep_drop_no_path = [], [], []
        ep_cap_blocks_per_step, ep_cap_blocks_per_injected = [], []
        ep_max_node_queue_peak, ep_max_node_queue_avg, ep_active_queues_avg = [], [], []
        trainable_indices = getattr(env.engine, 'trainable_host_indices', None)
        n_total_hosts = getattr(env.engine, 'n_total_hosts', maddpg.n_agents)
        n_actions = maddpg.n_actions
        # Snapshot topology before the loop so each episode gets a fresh graph
        _topo_snapshot = (
            [(u, v, dict(d)) for u, v, d in env.engine.topology.graph.edges(data=True)]
            if n_link_failures else None
        )
        for _ in range(n_eps):
            if _topo_snapshot is not None:
                G = env.engine.topology.graph
                G.remove_edges_from(list(G.edges()))
                G.add_edges_from(_topo_snapshot)
                env.engine.topology.refresh_path_cache()
            env.engine.reset_with_load(offered_load_factor=offered_load_factor)
            if n_link_failures:
                self._inject_failures(env.engine, n_link_failures)
                # Rebuild path caches on the post-failure topology so agents
                # select from K paths that are actually reachable.  This lets
                # different variants show differentiated routing behaviour;
                # without this refresh all K paths in kpath_cache still route
                # through removed links and _next_hop_for_transit falls back
                # identically for every variant.
                env.engine.topology.refresh_path_cache()
            states = [env.engine.get_state(h) for h in env.engine.get_all_hosts()]
            ep_r = 0
            for t in range(t_per_ep):
                if trainable_indices is not None:
                    t_states = [states[i] for i in trainable_indices]
                    t_actions = maddpg.choose_action(t_states)
                    actions = self._build_full_actions(t_actions, n_total_hosts,
                                                      trainable_indices, n_actions)
                else:
                    actions = maddpg.choose_action(states)
                next_states, rewards, info = env.step(actions)
                states = next_states
                ep_r += sum(rewards)
            ep_rewards.append(ep_r / len(maddpg.agents))
            ep_utils.append(float(np.nanmean(env.engine.get_link_utilization_distribution())))
            ep_stats = env.get_stats()
            ep_pdr.append(float(ep_stats.get('end_to_end_pdr', 0.0)))
            ep_resolved_pdr.append(float(ep_stats.get('resolved_pdr', 0.0)))
            ep_true_loss.append(float(ep_stats.get('true_loss_rate', 0.0)))
            ep_hop_frac.append(float(ep_stats.get('hop_delivery_frac', 0.0)))
            ep_goodput.append(float(ep_stats.get('goodput_per_step', 0.0)))
            ep_delay_p95.append(float(ep_stats.get('delay_p95', 0.0)))
            ep_backlog.append(float(ep_stats.get('backlog_end', 0.0)))
            ep_util_p95.append(float(ep_stats.get('util_p95', 0.0)))
            ep_hops_mean.append(float(ep_stats.get('hops_mean', 0.0)))
            ep_overload_frac.append(float(ep_stats.get('overload_step_fraction', 0.0)))
            ep_drop_ttl.append(float(ep_stats.get('drop_ttl_rate', 0.0)))
            ep_drop_overflow.append(float(ep_stats.get('drop_overflow_rate', 0.0)))
            ep_drop_no_path.append(float(ep_stats.get('drop_no_path_rate', 0.0)))
            ep_cap_blocks_per_step.append(float(ep_stats.get('capacity_block_per_step', 0.0)))
            ep_cap_blocks_per_injected.append(float(ep_stats.get('capacity_block_per_injected', 0.0)))
            ep_max_node_queue_peak.append(float(ep_stats.get('max_node_queue_peak', 0.0)))
            ep_max_node_queue_avg.append(float(ep_stats.get('max_node_queue_avg', 0.0)))
            ep_active_queues_avg.append(float(ep_stats.get('active_queues_avg', 0.0)))
        return {
            'mean_reward':      float(np.mean(ep_rewards)),
            'std_reward':       float(np.std(ep_rewards)),
            # true_pkt_loss: unique dropped / unique injected (per-packet, correct denominator)
            'mean_true_pkt_loss':    float(np.mean(ep_true_loss)),
            'std_true_pkt_loss':     float(np.std(ep_true_loss)),
            # resolved_pdr: delivered / (delivered + dropped) â€” excludes in-transit backlog
            'mean_resolved_pdr':     float(np.mean(ep_resolved_pdr)),
            'std_resolved_pdr':      float(np.std(ep_resolved_pdr)),
            # legacy metric: delivered / injected (penalised by episode-end backlog)
            'mean_end_to_end_pdr':   float(np.mean(ep_pdr)),
            'mean_util':        float(np.mean(ep_utils)),
            'mean_hop_delivery_frac': float(np.mean(ep_hop_frac)),
            'mean_goodput_per_step': float(np.mean(ep_goodput)),
            'mean_delay_p95':   float(np.mean(ep_delay_p95)),
            'mean_backlog_end': float(np.mean(ep_backlog)),
            'mean_util_p95':    float(np.mean(ep_util_p95)),
            'mean_hops_mean':   float(np.mean(ep_hops_mean)),
            'mean_overload_fraction': float(np.mean(ep_overload_frac)),
            'mean_drop_ttl_rate': float(np.mean(ep_drop_ttl)),
            'mean_drop_overflow_rate': float(np.mean(ep_drop_overflow)),
            'mean_drop_no_path_rate': float(np.mean(ep_drop_no_path)),
            'mean_capacity_block_per_step': float(np.mean(ep_cap_blocks_per_step)),
            'mean_capacity_block_per_injected': float(np.mean(ep_cap_blocks_per_injected)),
            'mean_max_node_queue_peak': float(np.mean(ep_max_node_queue_peak)),
            'mean_max_node_queue_avg': float(np.mean(ep_max_node_queue_avg)),
            'mean_active_queues_avg': float(np.mean(ep_active_queues_avg)),
            'offered_load_factor': float(offered_load_factor),
        }

    def _run_evpn_sp_episodes(self, engine: NetworkEngine,
                              n_eps: int, t_per_ep: int,
                              n_link_failures: int = 0,
                              offered_load_factor: float = 1.0) -> Dict:
        """EVPN with shortest-path tunnels: same forwarding model as MADDPG but always
        selects k=0 (shortest path) per destination.  Primary baseline â€” same buffering,
        same queue dynamics, no learning."""
        ep_utils = []
        ep_pdr, ep_resolved_pdr, ep_true_loss, ep_hop_frac, ep_goodput = [], [], [], [], []
        ep_delay_p95, ep_backlog, ep_util_p95 = [], [], []
        ep_hops_mean, ep_overload_frac = [], []
        ep_drop_ttl, ep_drop_overflow, ep_drop_no_path = [], [], []
        ep_cap_blocks_per_step, ep_cap_blocks_per_injected = [], []
        ep_max_node_queue_peak, ep_max_node_queue_avg, ep_active_queues_avg = [], [], []

        n_hosts = len(engine.topology.hosts)
        n_dest = engine.n_destinations
        n_actions = engine.n_actions
        k_per_dest = n_actions // max(1, n_dest)
        # k=0 action for every destination: action matrix row = [1, 0, 0, ...]
        k0_action = np.zeros(n_actions, dtype=np.float32)
        k0_action[0::k_per_dest] = 1.0
        all_actions = [k0_action for _ in range(n_hosts)]
        # Snapshot topology before the loop so each episode gets a fresh graph
        _topo_snapshot = (
            [(u, v, dict(d)) for u, v, d in engine.topology.graph.edges(data=True)]
            if n_link_failures else None
        )

        for _ in range(n_eps):
            if _topo_snapshot is not None:
                G = engine.topology.graph
                G.remove_edges_from(list(G.edges()))
                G.add_edges_from(_topo_snapshot)
                engine.topology.refresh_path_cache()
            engine.reset_with_load(offered_load_factor=offered_load_factor)
            if n_link_failures:
                self._inject_failures(engine, n_link_failures)
                engine.topology.refresh_path_cache()
            for _ in range(t_per_ep):
                engine.step(all_actions)
            ep_utils.append(float(np.nanmean(engine.get_link_utilization_distribution())))
            ep_stats = engine.get_episode_stats()
            ep_pdr.append(float(ep_stats.get('end_to_end_pdr', 0.0)))
            ep_resolved_pdr.append(float(ep_stats.get('resolved_pdr', 0.0)))
            ep_true_loss.append(float(ep_stats.get('true_loss_rate', 0.0)))
            ep_hop_frac.append(float(ep_stats.get('hop_delivery_frac', 0.0)))
            ep_goodput.append(float(ep_stats.get('goodput_per_step', 0.0)))
            ep_delay_p95.append(float(ep_stats.get('delay_p95', 0.0)))
            ep_backlog.append(float(ep_stats.get('backlog_end', 0.0)))
            ep_util_p95.append(float(ep_stats.get('util_p95', 0.0)))
            ep_hops_mean.append(float(ep_stats.get('hops_mean', 0.0)))
            ep_overload_frac.append(float(ep_stats.get('overload_step_fraction', 0.0)))
            ep_drop_ttl.append(float(ep_stats.get('drop_ttl_rate', 0.0)))
            ep_drop_overflow.append(float(ep_stats.get('drop_overflow_rate', 0.0)))
            ep_drop_no_path.append(float(ep_stats.get('drop_no_path_rate', 0.0)))
            ep_cap_blocks_per_step.append(float(ep_stats.get('capacity_block_per_step', 0.0)))
            ep_cap_blocks_per_injected.append(float(ep_stats.get('capacity_block_per_injected', 0.0)))
            ep_max_node_queue_peak.append(float(ep_stats.get('max_node_queue_peak', 0.0)))
            ep_max_node_queue_avg.append(float(ep_stats.get('max_node_queue_avg', 0.0)))
            ep_active_queues_avg.append(float(ep_stats.get('active_queues_avg', 0.0)))
        return {
            'mean_true_pkt_loss': float(np.mean(ep_true_loss)),
            'std_true_pkt_loss':  float(np.std(ep_true_loss)),
            'mean_resolved_pdr':  float(np.mean(ep_resolved_pdr)),
            'std_resolved_pdr':   float(np.std(ep_resolved_pdr)),
            'mean_end_to_end_pdr':   float(np.mean(ep_pdr)),
            'mean_util':     float(np.mean(ep_utils)),
            'mean_hop_delivery_frac': float(np.mean(ep_hop_frac)),
            'mean_goodput_per_step': float(np.mean(ep_goodput)),
            'mean_delay_p95':   float(np.mean(ep_delay_p95)),
            'mean_backlog_end': float(np.mean(ep_backlog)),
            'mean_util_p95':    float(np.mean(ep_util_p95)),
            'mean_hops_mean':   float(np.mean(ep_hops_mean)),
            'mean_overload_fraction': float(np.mean(ep_overload_frac)),
            'mean_drop_ttl_rate': float(np.mean(ep_drop_ttl)),
            'mean_drop_overflow_rate': float(np.mean(ep_drop_overflow)),
            'mean_drop_no_path_rate': float(np.mean(ep_drop_no_path)),
            'mean_capacity_block_per_step': float(np.mean(ep_cap_blocks_per_step)),
            'mean_capacity_block_per_injected': float(np.mean(ep_cap_blocks_per_injected)),
            'mean_max_node_queue_peak': float(np.mean(ep_max_node_queue_peak)),
            'mean_max_node_queue_avg': float(np.mean(ep_max_node_queue_avg)),
            'mean_active_queues_avg': float(np.mean(ep_active_queues_avg)),
            'offered_load_factor': float(offered_load_factor),
            'routing_mode': 'evpn_sp',
        }

    def _run_ospf_episodes(self, engine: NetworkEngine,
                           n_eps: int, t_per_ep: int,
                           n_link_failures: int = 0,
                           offered_load_factor: float = 1.0,
                           mode: str = 'full') -> Dict:
        ep_utils = []
        ep_pdr, ep_resolved_pdr, ep_true_loss, ep_hop_frac, ep_goodput = [], [], [], [], []
        ep_delay_p95, ep_backlog, ep_util_p95 = [], [], []
        ep_hops_mean, ep_overload_frac = [], []
        ep_drop_ttl, ep_drop_overflow, ep_drop_no_path = [], [], []
        ep_cap_blocks_per_step, ep_cap_blocks_per_injected = [], []
        ep_max_node_queue_peak, ep_max_node_queue_avg, ep_active_queues_avg = [], [], []
        # Snapshot topology before the loop so each episode gets a fresh graph
        _topo_snapshot = (
            [(u, v, dict(d)) for u, v, d in engine.topology.graph.edges(data=True)]
            if n_link_failures else None
        )
        for _ in range(n_eps):
            if _topo_snapshot is not None:
                G = engine.topology.graph
                G.remove_edges_from(list(G.edges()))
                G.add_edges_from(_topo_snapshot)
                engine.topology.refresh_path_cache()
            engine.reset_with_load(offered_load_factor=offered_load_factor)
            if n_link_failures:
                self._inject_failures(engine, n_link_failures)
                engine.topology.refresh_path_cache()
            for _ in range(t_per_ep):
                if mode == 'queue_level':
                    engine.ospf_queue_level_step()
                else:
                    engine.ospf_step()
            ep_utils.append(float(np.nanmean(engine.get_link_utilization_distribution())))
            ep_stats = engine.get_episode_stats()
            ep_pdr.append(float(ep_stats.get('end_to_end_pdr', 0.0)))
            ep_resolved_pdr.append(float(ep_stats.get('resolved_pdr', 0.0)))
            ep_true_loss.append(float(ep_stats.get('true_loss_rate', 0.0)))
            ep_hop_frac.append(float(ep_stats.get('hop_delivery_frac', 0.0)))
            ep_goodput.append(float(ep_stats.get('goodput_per_step', 0.0)))
            ep_delay_p95.append(float(ep_stats.get('delay_p95', 0.0)))
            ep_backlog.append(float(ep_stats.get('backlog_end', 0.0)))
            ep_util_p95.append(float(ep_stats.get('util_p95', 0.0)))
            ep_hops_mean.append(float(ep_stats.get('hops_mean', 0.0)))
            ep_overload_frac.append(float(ep_stats.get('overload_step_fraction', 0.0)))
            ep_drop_ttl.append(float(ep_stats.get('drop_ttl_rate', 0.0)))
            ep_drop_overflow.append(float(ep_stats.get('drop_overflow_rate', 0.0)))
            ep_drop_no_path.append(float(ep_stats.get('drop_no_path_rate', 0.0)))
            ep_cap_blocks_per_step.append(float(ep_stats.get('capacity_block_per_step', 0.0)))
            ep_cap_blocks_per_injected.append(float(ep_stats.get('capacity_block_per_injected', 0.0)))
            ep_max_node_queue_peak.append(float(ep_stats.get('max_node_queue_peak', 0.0)))
            ep_max_node_queue_avg.append(float(ep_stats.get('max_node_queue_avg', 0.0)))
            ep_active_queues_avg.append(float(ep_stats.get('active_queues_avg', 0.0)))
        return {
            # true_pkt_loss: unique dropped / unique injected (per-packet, correct denominator)
            'mean_true_pkt_loss': float(np.mean(ep_true_loss)),
            'std_true_pkt_loss':  float(np.std(ep_true_loss)),
            # resolved_pdr: delivered / (delivered + dropped) â€” excludes in-transit backlog
            'mean_resolved_pdr':  float(np.mean(ep_resolved_pdr)),
            'std_resolved_pdr':   float(np.std(ep_resolved_pdr)),
            # legacy metric: delivered / injected (penalised by episode-end backlog)
            'mean_end_to_end_pdr':   float(np.mean(ep_pdr)),
            'mean_util':     float(np.mean(ep_utils)),
            'mean_hop_delivery_frac': float(np.mean(ep_hop_frac)),
            'mean_goodput_per_step': float(np.mean(ep_goodput)),
            'mean_delay_p95':   float(np.mean(ep_delay_p95)),
            'mean_backlog_end': float(np.mean(ep_backlog)),
            'mean_util_p95':    float(np.mean(ep_util_p95)),
            'mean_hops_mean':   float(np.mean(ep_hops_mean)),
            'mean_overload_fraction': float(np.mean(ep_overload_frac)),
            'mean_drop_ttl_rate': float(np.mean(ep_drop_ttl)),
            'mean_drop_overflow_rate': float(np.mean(ep_drop_overflow)),
            'mean_drop_no_path_rate': float(np.mean(ep_drop_no_path)),
            'mean_capacity_block_per_step': float(np.mean(ep_cap_blocks_per_step)),
            'mean_capacity_block_per_injected': float(np.mean(ep_cap_blocks_per_injected)),
            'mean_max_node_queue_peak': float(np.mean(ep_max_node_queue_peak)),
            'mean_max_node_queue_avg': float(np.mean(ep_max_node_queue_avg)),
            'mean_active_queues_avg': float(np.mean(ep_active_queues_avg)),
            'offered_load_factor': float(offered_load_factor),
            'routing_mode': mode,
        }

    @staticmethod
    def _inject_failures(engine: NetworkEngine, n: int):
        G = engine.topology.graph
        # Only remove non-bridge edges: bridges create hard path cuts where no
        # routing policy can recover, making all variants degrade identically.
        bridge_set: set = set()
        for u, v in nx.bridges(G):
            bridge_set.add((u, v))
            bridge_set.add((v, u))
        candidates = [(u, v) for u, v in G.edges() if (u, v) not in bridge_set]
        if not candidates:
            # Fallback: no non-bridge edges available (rare), use all edges
            candidates = list(G.edges())
        chosen_indices = np.random.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        for idx in chosen_indices:
            u, v = candidates[int(idx)]
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        # Do NOT refresh_path_cache here: callers rebuild the path caches
        # immediately after this call so that all K paths in kpath_cache are
        # reachable on the post-failure topology.  Refreshing inside this
        # function would prevent the caller from first restoring the snapshot
        # (needed to give each episode a fresh starting topology).

    def _run_phase2_load_sweep(self, training_results: Dict, n_eps: int, t_per_ep: int) -> Dict:
        load_cfg = self.config.get('load_sweep', {})
        loads = [float(x) for x in load_cfg.get('offered_loads', [1.0])]
        include_failures = bool(load_cfg.get('include_failures', False))

        out = {
            'meta': {
                'loads': loads,
                'evaluation_episodes': int(n_eps),
                'timesteps_per_episode': int(t_per_ep),
                'include_failures': include_failures,
            },
            'methods': {},
            'summary': {},
        }

        for vcfg in self.config['variants']:
            name = vcfg['name']
            if name not in training_results:
                continue
            logger.info(f"[P2][SWEEP] {name}")
            maddpg, _, env = self._make_variant(vcfg)
            self._load_variant_checkpoint(maddpg, name)
            method_payload = {'normal': {}, 'dual_link_failure': {}}
            for load in loads:
                key = f"load_{load:.2f}"
                method_payload['normal'][key] = self._run_eval_episodes(
                    maddpg, env, n_eps, t_per_ep,
                    n_link_failures=0,
                    offered_load_factor=load,
                )
                if include_failures:
                    method_payload['dual_link_failure'][key] = self._run_eval_episodes(
                        maddpg, env, n_eps, t_per_ep,
                        n_link_failures=2,
                        offered_load_factor=load,
                    )
            out['methods'][name] = method_payload

        logger.info("[P2][SWEEP] EVPN_SP")
        _evpn_sweep_engine = NetworkEngine(
            topology_type=self.config.get('topology', {}).get('type', 'service_provider'),
            n_nodes=65,
            traffic_config=self.config.get('traffic', {}),
        )
        evpn_payload = {'normal': {}, 'dual_link_failure': {}}
        for load in loads:
            key = f"load_{load:.2f}"
            evpn_payload['normal'][key] = self._run_evpn_sp_episodes(
                _evpn_sweep_engine, n_eps, t_per_ep,
                n_link_failures=0,
                offered_load_factor=load,
            )
            if include_failures:
                evpn_payload['dual_link_failure'][key] = self._run_evpn_sp_episodes(
                    _evpn_sweep_engine, n_eps, t_per_ep,
                    n_link_failures=2,
                    offered_load_factor=load,
                )
        out['methods']['EVPN_SP'] = evpn_payload

        for name, payload in out['methods'].items():
            out['summary'][name] = {
                'critical_load_normal': self._compute_critical_load(payload.get('normal', {})),
                'critical_load_failure': self._compute_critical_load(payload.get('dual_link_failure', {}))
                    if include_failures else None,
            }

        return out

    def _compute_critical_load(self, scenario_payload: Dict) -> Optional[float]:
        if not scenario_payload:
            return None

        clean_slo = self.config.get('clean_slo', {})
        max_loss = float(clean_slo.get('max_pkt_loss_pct', 5.0))
        max_delay = float(clean_slo.get('max_delay_p95', 8.0))
        min_pdr = float(clean_slo.get('min_end_to_end_pdr_pct', 85.0))

        ordered = sorted(
            scenario_payload.items(),
            key=lambda kv: float(kv[0].split('_')[-1])
        )
        last_ok = None
        for key, metrics in ordered:
            load = float(key.split('_')[-1])
            loss_ok = float(metrics.get('mean_true_pkt_loss', 1000.0)) <= max_loss
            delay_ok = float(metrics.get('mean_delay_p95', 1e9)) <= max_delay
            pdr_ok = float(metrics.get('mean_resolved_pdr', 0.0)) >= min_pdr
            if loss_ok and delay_ok and pdr_ok:
                last_ok = load
            else:
                return last_ok
        return last_ok

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 3 â€” FGSM atttack evaluation
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def evaluate_fgsm(self, training_results: Optional[Dict] = None) -> Dict:
        logger.info("â•â•â•â•â•â• PHASE 3 â€” FGSM ATTACK EVALUATION â•â•â•â•â•â•")
        if training_results is None:
            training_results = self._load('phase1_training_results.json')

        phase_t0 = time.time()
        runtime_cfg = self.config.get('runtime_control', {})
        seed_cfg = runtime_cfg.get('seed_expansion', {})
        max_attack_variants = int(runtime_cfg.get('max_attack_variants', -1))
        max_attack_cases = int(runtime_cfg.get('max_attack_cases_per_variant', -1))
        phase3_enable_slo_pruning = bool(runtime_cfg.get('phase3_enable_slo_pruning', True))
        phase3_consecutive_fail_limit = int(runtime_cfg.get('phase3_consecutive_fail_limit', 2))
        phase3_skip_after_critical = bool(
            runtime_cfg.get('phase3_skip_after_critical_epsilon', True)
        )
        attack_configs = self.config.get('attack_configs', [])
        if max_attack_cases > 0:
            attack_configs = attack_configs[:max_attack_cases]

        all_results = {}
        all_results['_run_config'] = {
            'runtime_profile': runtime_cfg.get('profile', 'standard'),
            'max_attack_cases_per_variant': max_attack_cases,
            'max_attack_variants': max_attack_variants,
            'phase3_enable_slo_pruning': phase3_enable_slo_pruning,
            'phase3_consecutive_fail_limit': phase3_consecutive_fail_limit,
            'phase3_skip_after_critical_epsilon': phase3_skip_after_critical,
            'attack_cases_used': len(attack_configs),
        }

        # Resume: load any partial results saved by a previous run
        _partial_path = os.path.join(self.results_dir, 'phase3_fgsm_results.json')
        if os.path.exists(_partial_path):
            try:
                with open(_partial_path) as _pf:
                    _partial = json.load(_pf)
                for _k, _v in _partial.items():
                    if _k != '_run_config':
                        all_results[_k] = _v
                logger.info(f"[P3] Resumed partial results â€” {len(all_results) - 1} variant(s) already done")
            except Exception as _e:
                logger.warning(f"[P3] Could not load partial results: {_e}")

        variants = self.config['variants']
        if max_attack_variants > 0:
            variants = variants[:max_attack_variants]

        for vcfg in variants:
            variant_t0 = time.time()
            name = vcfg['name']
            if name not in training_results:
                logger.warning(f"[P2] {name} â€” no trained model, skipping")
                continue
            if name in all_results:
                logger.info(f"[P3] {name} â€” already complete (resuming), skipping")
                continue
            logger.info(f"[P2] Attacking {name}")
            maddpg, engine, env = self._make_variant(vcfg)
            self._load_variant_checkpoint(maddpg, name)
            variant_results = {}
            skipped_cases = []
            critical_by_type = {}
            consecutive_failures = 0
            variant_pruned = False

            # Standard FGSM sweep
            for acfg in attack_configs:
                atype    = acfg['attack_type']
                eps      = acfg['epsilon']
                n_eps    = acfg['evaluation_episodes']
                t_per_ep = self.config['training']['timesteps_per_episode']
                key      = f"{atype}_eps{eps}"

                if phase3_skip_after_critical and atype in critical_by_type and eps > critical_by_type[atype]:
                    skipped_cases.append({
                        'attack_type': atype,
                        'epsilon': eps,
                        'reason': 'skipped_after_critical_epsilon',
                    })
                    continue

                logger.info(f"[P2]   {name}  {key}")
                case_t0 = time.time()
                try:
                    clean    = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=False)
                    attacked = self._attack_episodes(maddpg, env, n_eps, t_per_ep,
                                                     attack=True, attack_type=atype,
                                                     epsilon=eps)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"[P3]   {name}  {key} â€” CUDA OOM, recording as failed")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    variant_results[key] = {'error': 'CUDA_OOM', 'run_config': {'attack_type': atype, 'epsilon': eps}}
                    skipped_cases.append({'attack_type': atype, 'epsilon': eps, 'reason': 'CUDA_OOM'})
                    consecutive_failures += 1
                    if atype not in critical_by_type:
                        critical_by_type[atype] = eps
                    continue
                slo_eval = self._evaluate_attack_slo(clean, attacked)
                variant_results[key] = {
                    'clean': clean, 'attacked': attacked,
                    'metrics': self._compare(clean, attacked),
                    'slo': slo_eval,
                    'run_config': {
                        'attack_type': atype,
                        'epsilon': eps,
                        'evaluation_episodes': n_eps,
                        'timesteps_per_episode': t_per_ep,
                        'runtime_sec': float(time.time() - case_t0),
                    },
                }

                if not slo_eval['success']:
                    consecutive_failures += 1
                    if atype not in critical_by_type:
                        critical_by_type[atype] = eps
                else:
                    consecutive_failures = 0

                if phase3_enable_slo_pruning and phase3_consecutive_fail_limit > 0 \
                   and consecutive_failures >= phase3_consecutive_fail_limit:
                    variant_pruned = True
                    variant_results['_pruned'] = {
                        'reason': 'consecutive_slo_failures',
                        'limit': phase3_consecutive_fail_limit,
                        'at_case': key,
                    }
                    break

            if not variant_pruned:
                # Attack-surface analysis (core vs dist vs access)
                logger.info(f"[P2]   {name}  attack surface analysis")
                n_eps    = self.config['attack_configs'][-1]['evaluation_episodes']
                t_per_ep = self.config['training']['timesteps_per_episode']
                try:
                    variant_results['surface'] = self._attack_surface_analysis(
                        maddpg, env, n_eps=n_eps, t_per_ep=t_per_ep, epsilon=0.10)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"[P3]   {name}  attack surface analysis â€” CUDA OOM, skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    variant_results['surface'] = {'error': 'CUDA_OOM'}

                # GNN embedding attack (GNN variants only)
                if vcfg.get('use_gnn', False):
                    logger.info(f"[P2]   {name}  GNN embedding attack")
                    try:
                        variant_results['gnn_embedding_attack'] = self._attack_episodes(
                            maddpg, env, n_eps=n_eps, t_per_ep=t_per_ep,
                            attack=True, attack_type='packet_loss', epsilon=0.10)
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"[P3]   {name}  GNN embedding attack â€” CUDA OOM, skipping")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        variant_results['gnn_embedding_attack'] = {'error': 'CUDA_OOM'}
            else:
                skipped_cases.append({
                    'attack_type': 'surface_and_gnn',
                    'epsilon': None,
                    'reason': 'variant_pruned',
                })

            variant_results['attack_summary'] = self._summarize_variant_attacks(variant_results)
            variant_results['_runtime'] = {
                'variant_runtime_sec': float(time.time() - variant_t0),
                'skipped_cases': skipped_cases,
                'critical_epsilon_by_type': critical_by_type,
            }
            all_results[name] = variant_results

            # Incremental save â€” preserve progress even if a later variant crashes
            self._save(all_results, 'phase3_fgsm_results.json')

            # Release GPU memory before processing the next variant to prevent OOM
            del maddpg, engine, env
            gc.collect()
            gc.collect()  # two passes to break cycles
            if torch.cuda.is_available():
                torch.cuda.synchronize()   # wait for any async kernels to finish
                torch.cuda.empty_cache()   # release allocator cache back to OS
                logger.info(f"[P3] GPU after {name}: "
                            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated, "
                            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB reserved")

        all_results['_run_config']['phase3_runtime_sec'] = float(time.time() - phase_t0)
        self._save(all_results, 'phase3_fgsm_results.json')
        ranking = self._build_phase3_rankings(all_results)
        
        # Add seed convergence telemetry for Phase 3
        # Track the attack evaluation episodes used
        attack_episodes_used = set()
        for acfg in attack_configs:
            attack_episodes_used.add(acfg.get('evaluation_episodes', 30))
        
        ranking['seed_convergence'] = {
            'attack_evaluation_episodes': list(attack_episodes_used),
            'adaptive_expansion_enabled': seed_cfg.get('enable_adaptive', False),
            'total_attack_cases': len(attack_configs),
        }
        
        self._save(ranking, 'phase3_rankings.json')
        self._generate_plots_for_phase(3)
        logger.info("[P3] evaluation complete")
        return all_results

    def _attack_episodes(self, maddpg: MADDPG, env: NetworkEnv,
                         n_eps: int, t_per_ep: int,
                         attack: bool = False,
                         attack_type: str = 'packet_loss',
                         epsilon: float = 0.05,
                         targeted_tiers: Optional[List[str]] = None) -> Dict:
        ep_rewards, ep_losses, ep_delivery = [], [], []
        ep_delay_p95, ep_backlog, ep_goodput = [], [], []
        hosts = env.engine.get_all_hosts()
        trainable_indices = getattr(env.engine, 'trainable_host_indices', None)
        n_total_hosts = getattr(env.engine, 'n_total_hosts', maddpg.n_agents)
        n_actions = maddpg.n_actions
        if trainable_indices is not None:
            trainable_set_idx = set(trainable_indices)
            topo_to_maddpg = {topo_idx: ti for ti, topo_idx in enumerate(trainable_indices)}
        else:
            trainable_set_idx = None
            topo_to_maddpg = None

        if attack:
            self.attack_framework.epsilon = float(epsilon)
            self.attack_framework.attack_type = attack_type

        for _ in range(n_eps):
            states = env.reset()
            ep_r = ep_sent = ep_dropped = ep_delivered = 0
            for _ in range(t_per_ep):
                if attack:
                    adv = []
                    for topo_idx, (host, s) in enumerate(zip(hosts, states)):
                        # Only attack trainable agents; non-trainable nodes have no actor
                        if trainable_set_idx is not None and topo_idx not in trainable_set_idx:
                            adv.append(s)
                        elif targeted_tiers and env.engine.get_tier(host) not in targeted_tiers:
                            adv.append(s)
                        else:
                            agent_idx = topo_to_maddpg[topo_idx] if topo_to_maddpg else topo_idx
                            adv.append(self.attack_framework.generate_adversarial_state(
                                state=s,
                                agent_network=maddpg.agents[agent_idx],
                                network_engine=env.engine,
                                agent_index=agent_idx,
                            ))
                    states = adv
                with torch.no_grad():
                    if trainable_indices is not None:
                        t_states = [states[i] for i in trainable_indices]
                        t_actions = maddpg.choose_action(t_states)
                        actions = self._build_full_actions(t_actions, n_total_hosts,
                                                          trainable_indices, n_actions)
                    else:
                        actions = maddpg.choose_action(states)
                next_states, rewards, info = env.step(actions)
                states = next_states
                ep_r      += sum(rewards)
                ep_sent   += info['packets_sent']
                ep_delivered += info.get('packets_delivered', 0)
                ep_dropped += info['packets_dropped']
            ep_rewards.append(ep_r / len(maddpg.agents))
            ep_losses.append(ep_dropped / max(1, ep_sent) * 100)
            ep_delivery.append(ep_delivered / max(1, ep_sent) * 100)
            ep_stats = env.get_stats()
            ep_delay_p95.append(float(ep_stats.get('delay_p95', 0.0)))
            ep_backlog.append(float(ep_stats.get('backlog_end', 0.0)))
            ep_goodput.append(float(ep_stats.get('goodput_per_step', 0.0)))
            ep_delivery[-1] = float(ep_stats.get('end_to_end_pdr', ep_delivery[-1]))

        return {
            'mean_reward':   float(np.mean(ep_rewards)),
            'std_reward':    float(np.std(ep_rewards)),
            'mean_pkt_loss': float(np.mean(ep_losses)),
            'std_pkt_loss':  float(np.std(ep_losses)),
            'mean_end_to_end_pdr': float(np.mean(ep_delivery)),
            'std_end_to_end_pdr':  float(np.std(ep_delivery)),
            'mean_delay_p95': float(np.mean(ep_delay_p95)),
            'mean_backlog_end': float(np.mean(ep_backlog)),
            'mean_goodput_per_step': float(np.mean(ep_goodput)),
            'run_config': {
                'attack': attack,
                'attack_type': attack_type if attack else 'clean',
                'epsilon': float(epsilon) if attack else 0.0,
                'targeted_tiers': targeted_tiers or [],
                'evaluation_episodes': int(n_eps),
                'timesteps_per_episode': int(t_per_ep),
            },
        }

    def _attack_surface_analysis(self, maddpg: MADDPG, env: NetworkEnv,
                                  n_eps: int, t_per_ep: int,
                                  epsilon: float = 0.10) -> Dict:
        clean  = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=False)
        core_r = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['core'])
        dist_r = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['dist'])
        acc_r  = self._attack_episodes(maddpg, env, n_eps, t_per_ep, attack=True,
                                       attack_type='packet_loss', epsilon=epsilon,
                                       targeted_tiers=['access'])
        return {
            'clean':           clean,
            'core_attacked': {
                'results': core_r,
                'metrics': self._compare(clean, core_r),
                'slo': self._evaluate_attack_slo(clean, core_r),
            },
            'dist_attacked': {
                'results': dist_r,
                'metrics': self._compare(clean, dist_r),
                'slo': self._evaluate_attack_slo(clean, dist_r),
            },
            'access_attacked': {
                'results': acc_r,
                'metrics': self._compare(clean, acc_r),
                'slo': self._evaluate_attack_slo(clean, acc_r),
            },
            'n_core_agents':   len(env.engine.topology.core_nodes),
            'n_dist_agents':   len(env.engine.topology.dist_nodes),
            'n_access_agents': len(env.engine.topology.access_nodes),
        }

    def _compare(self, clean: Dict, attacked: Dict) -> Dict:
        cr, ar = clean['mean_reward'], attacked['mean_reward']
        clean_delivery = clean.get('mean_end_to_end_pdr', 0.0)
        attacked_delivery = attacked.get('mean_end_to_end_pdr', 0.0)
        return {
            'reward_degradation_pct':  float((cr - ar) / abs(cr) * 100 if cr != 0 else 0),
            'pkt_loss_increase_pct':   float(attacked['mean_pkt_loss'] - clean['mean_pkt_loss']),
            'delivery_rate_drop_pct':  float(clean_delivery - attacked_delivery),
        }

    def _evaluate_attack_slo(self, clean: Dict, attacked: Dict) -> Dict:
        slo_cfg = self.config.get('fgsm_slo', {})
        max_pkt_loss_pct = float(slo_cfg.get('max_pkt_loss_pct', 100.0))
        min_delivery_rate_pct = float(slo_cfg.get('min_delivery_rate_pct', 0.0))
        max_reward_degradation_pct = float(slo_cfg.get('max_reward_degradation_pct', 100.0))
        max_delay_p95 = float(slo_cfg.get('max_delay_p95', 1e9))

        reward_deg = self._compare(clean, attacked)['reward_degradation_pct']
        pkt_loss = float(attacked.get('mean_pkt_loss', 0.0))
        delivery = float(attacked.get('mean_end_to_end_pdr', 0.0))
        delay_p95 = float(attacked.get('mean_delay_p95', 0.0))

        checks = {
            'pkt_loss_within_slo': pkt_loss <= max_pkt_loss_pct,
            'delivery_within_slo': delivery >= min_delivery_rate_pct,
            'reward_degradation_within_slo': reward_deg <= max_reward_degradation_pct,
            'delay_p95_within_slo': delay_p95 <= max_delay_p95,
        }
        violated = [name for name, passed in checks.items() if not passed]

        return {
            'success': len(violated) == 0,
            'violated_checks': violated,
            'thresholds': {
                'max_pkt_loss_pct': max_pkt_loss_pct,
                'min_delivery_rate_pct': min_delivery_rate_pct,
                'max_reward_degradation_pct': max_reward_degradation_pct,
                'max_delay_p95': max_delay_p95,
            },
            'observed': {
                'mean_pkt_loss': pkt_loss,
                'mean_end_to_end_pdr': delivery,
                'reward_degradation_pct': reward_deg,
                'mean_delay_p95': delay_p95,
            },
        }

    def _summarize_variant_attacks(self, variant_results: Dict) -> Dict:
        by_type = defaultdict(list)
        total_cases = 0
        success_cases = 0

        for key, payload in variant_results.items():
            if not isinstance(payload, dict):
                continue
            if 'run_config' not in payload or 'slo' not in payload:
                continue
            run_cfg = payload['run_config']
            attack_type = run_cfg.get('attack_type', 'unknown')
            epsilon = float(run_cfg.get('epsilon', 0.0))
            success = bool(payload['slo'].get('success', False))
            by_type[attack_type].append((epsilon, success))
            total_cases += 1
            if success:
                success_cases += 1

        attack_type_summary = {}
        for attack_type, cases in by_type.items():
            ordered = sorted(cases, key=lambda x: x[0])
            critical = None
            for eps, success in ordered:
                if not success:
                    critical = eps
                    break
            attack_type_summary[attack_type] = {
                'tested_cases': len(ordered),
                'slo_success_rate_pct': float(
                    100.0 * sum(1 for _, success in ordered if success) / max(1, len(ordered))
                ),
                'critical_epsilon': critical,
            }

        return {
            'tested_cases': total_cases,
            'slo_success_rate_pct': float(100.0 * success_cases / max(1, total_cases)),
            'attack_types': attack_type_summary,
        }

    # â”€â”€ ranking helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _score_profiles(self) -> Dict:
        cfg = self.config.get('scoring', {})
        profiles = cfg.get('profiles')
        if profiles:
            return profiles
        return {
            'default': {
                'higher_is_better': {
                    # resolved_pdr = delivered/(delivered+dropped): excludes in-transit backlog
                    'mean_resolved_pdr':     0.35,
                    'mean_goodput_per_step': 0.20,
                },
                'lower_is_better': {
                    # true_pkt_loss = dropped/injected: per unique packet, correct denominator
                    'mean_true_pkt_loss':     0.20,
                    'mean_delay_p95':         0.10,
                    'mean_backlog_end':       0.05,
                    'mean_hops_mean':         0.05,
                    'mean_overload_fraction': 0.05,
                },
            },
            'robustness': {
                'higher_is_better': {
                    'mean_resolved_pdr':     0.35,
                    'mean_goodput_per_step': 0.20,
                },
                'lower_is_better': {
                    'mean_true_pkt_loss':     0.20,
                    'mean_delay_p95':         0.10,
                    'mean_backlog_end':       0.05,
                    'mean_hops_mean':         0.05,
                    'mean_overload_fraction': 0.05,
                },
            },
        }

    @staticmethod
    def _normalized_scores(items: List[Dict], metric: str, higher_is_better: bool) -> Dict[str, float]:
        vals = [float(it.get(metric, 0.0)) for it in items]
        if not vals:
            return {}
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-12:
            return {it['name']: 1.0 for it in items}

        out = {}
        for it in items:
            v = float(it.get(metric, 0.0))
            base = (v - vmin) / (vmax - vmin)
            out[it['name']] = base if higher_is_better else (1.0 - base)
        return out

    def _compute_composite_scores(self, items: List[Dict], profile: Dict) -> List[Dict]:
        higher = profile.get('higher_is_better', {})
        lower = profile.get('lower_is_better', {})
        metric_scores = {}

        for metric in higher:
            metric_scores[metric] = self._normalized_scores(items, metric, higher_is_better=True)
        for metric in lower:
            metric_scores[metric] = self._normalized_scores(items, metric, higher_is_better=False)

        total_weight = float(sum(higher.values()) + sum(lower.values())) or 1.0
        ranked = []
        for it in items:
            name = it['name']
            weighted = 0.0
            for metric, w in higher.items():
                weighted += float(w) * metric_scores.get(metric, {}).get(name, 0.0)
            for metric, w in lower.items():
                weighted += float(w) * metric_scores.get(metric, {}).get(name, 0.0)
            ranked.append({
                'name': name,
                'composite_score': float(100.0 * weighted / total_weight),
                'metrics': it,
            })

        ranked.sort(key=lambda x: x['composite_score'], reverse=True)
        for i, row in enumerate(ranked, start=1):
            row['rank'] = i
        return ranked

    def _build_phase2_rankings(self, phase2_results: Dict) -> Dict:
        profiles = self._score_profiles()
        out = {'profiles': {}, 'scenarios': {}}

        for scenario in ['normal', 'dual_link_failure']:
            items = []
            for name, payload in phase2_results.items():
                if str(name).startswith('SP') or name == 'EVPN_SP':
                    continue
                if scenario not in payload:
                    continue
                metrics = payload[scenario]
                items.append({'name': name, **metrics})

            out['scenarios'][scenario] = {}
            for profile_name, profile in profiles.items():
                out['scenarios'][scenario][profile_name] = self._compute_composite_scores(items, profile)

        out['profiles'] = profiles
        return out

    def _build_phase3_rankings(self, phase3_results: Dict) -> Dict:
        profiles = self._score_profiles()
        out = {'profiles': profiles, 'overall': {}, 'by_attack_type': {}}

        # Aggregate per variant across all standard attack cases
        per_variant = []
        by_attack_type = defaultdict(list)
        for name, payload in phase3_results.items():
            if name.startswith('_') or not isinstance(payload, dict):
                continue

            summary = payload.get('attack_summary', {})
            metrics_acc = {
                'mean_pkt_loss': [],
                'mean_delivery_rate': [],
                'mean_delay_p95': [],
                'mean_backlog_end': [],
                'mean_goodput_per_step': [],
            }

            for key, case in payload.items():
                if not isinstance(case, dict):
                    continue
                if 'attacked' not in case or 'run_config' not in case:
                    continue
                attacked = case['attacked']
                run_cfg = case['run_config']
                atype = run_cfg.get('attack_type', 'unknown')
                row = {
                    'name': name,
                    'attack_type': atype,
                    'mean_pkt_loss': float(attacked.get('mean_pkt_loss', 0.0)),
                    'mean_delivery_rate': float(attacked.get('mean_delivery_rate', 0.0)),
                    'mean_delay_p95': float(attacked.get('mean_delay_p95', 0.0)),
                    'mean_backlog_end': float(attacked.get('mean_backlog_end', 0.0)),
                    'mean_goodput_per_step': float(attacked.get('mean_goodput_per_step', 0.0)),
                }
                by_attack_type[atype].append(row)
                for metric in metrics_acc:
                    metrics_acc[metric].append(row[metric])

            per_variant.append({
                'name': name,
                'mean_pkt_loss': float(np.mean(metrics_acc['mean_pkt_loss'])) if metrics_acc['mean_pkt_loss'] else 0.0,
                'mean_delivery_rate': float(np.mean(metrics_acc['mean_delivery_rate'])) if metrics_acc['mean_delivery_rate'] else 0.0,
                'mean_delay_p95': float(np.mean(metrics_acc['mean_delay_p95'])) if metrics_acc['mean_delay_p95'] else 0.0,
                'mean_backlog_end': float(np.mean(metrics_acc['mean_backlog_end'])) if metrics_acc['mean_backlog_end'] else 0.0,
                'mean_goodput_per_step': float(np.mean(metrics_acc['mean_goodput_per_step'])) if metrics_acc['mean_goodput_per_step'] else 0.0,
                'slo_success_rate_pct': float(summary.get('slo_success_rate_pct', 0.0)),
            })

        for profile_name, profile in profiles.items():
            out['overall'][profile_name] = self._compute_composite_scores(per_variant, profile)

        # Per attack type ranking
        for atype, rows in by_attack_type.items():
            out['by_attack_type'][atype] = {}
            # collapse by variant within attack type
            per_variant_rows = defaultdict(lambda: {
                'mean_pkt_loss': [],
                'mean_delivery_rate': [],
                'mean_delay_p95': [],
                'mean_backlog_end': [],
                'mean_goodput_per_step': [],
            })
            for row in rows:
                p = per_variant_rows[row['name']]
                for k in p:
                    p[k].append(row[k])

            collapsed = []
            for name, data in per_variant_rows.items():
                collapsed.append({
                    'name': name,
                    'mean_pkt_loss': float(np.mean(data['mean_pkt_loss'])) if data['mean_pkt_loss'] else 0.0,
                    'mean_delivery_rate': float(np.mean(data['mean_delivery_rate'])) if data['mean_delivery_rate'] else 0.0,
                    'mean_delay_p95': float(np.mean(data['mean_delay_p95'])) if data['mean_delay_p95'] else 0.0,
                    'mean_backlog_end': float(np.mean(data['mean_backlog_end'])) if data['mean_backlog_end'] else 0.0,
                    'mean_goodput_per_step': float(np.mean(data['mean_goodput_per_step'])) if data['mean_goodput_per_step'] else 0.0,
                })

            for profile_name, profile in profiles.items():
                out['by_attack_type'][atype][profile_name] = self._compute_composite_scores(
                    collapsed, profile
                )

        return out

    # â”€â”€ seed expansion & rank stability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _kendall_tau_simple(rank1: List[str], rank2: List[str]) -> float:
        """Compute Kendall tau correlation between two rank orderings.
        
        Simple implementation: count concordant vs discordant pairs.
        Returns tau in [-1, 1] where 1 = perfect agreement, 0 = independence.
        """
        if not rank1 or not rank2:
            return 1.0
        
        # Map names to indices in rank2
        rank2_idx = {name: i for i, name in enumerate(rank2)}
        
        # Count concordant and discordant pairs in rank1 that map to rank2
        concordant = discordant = 0
        for i in range(len(rank1)):
            for j in range(i + 1, len(rank1)):
                name_i, name_j = rank1[i], rank1[j]
                if name_i not in rank2_idx or name_j not in rank2_idx:
                    continue
                idx_i, idx_j = rank2_idx[name_i], rank2_idx[name_j]
                if idx_i < idx_j:  # Same order in rank2
                    concordant += 1
                else:
                    discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return 1.0
        
        tau = (concordant - discordant) / total
        return float(tau)

    def _compute_rank_stability(self, prev_ranked: List[Dict], curr_ranked: List[Dict],
                                top_k: int = 5) -> float:
        """Compute stability between previous and current rankings using top-K order.
        
        Uses Kendall tau correlation on top-K variants.
        Returns tau in [0, 1] (clamped to non-negative for agreement metric).
        """
        prev_names = [r['name'] for r in prev_ranked[:top_k]]
        curr_names = [r['name'] for r in curr_ranked[:top_k]]
        
        if not prev_names or not curr_names:
            return 1.0
        
        if kendalltau is not None:
            try:
                # Use scipy if available
                tau_val, _ = kendalltau(
                    [next((i for i, n in enumerate(prev_names) if n == name), len(prev_names)) 
                     for name in curr_names],
                    list(range(len(curr_names)))
                )
                return float(max(0.0, tau_val))  # Ensure non-negative
            except Exception:
                pass  # Fall back to simple implementation
        
        # Simple Kendall tau implementation
        tau = self._kendall_tau_simple(prev_names, curr_names)
        return float(max(0.0, tau))

    # â”€â”€ plotting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _generate_plots_for_phase(self, phase: int):
        if not PLOTTING_AVAILABLE:
            return

        logger.info(f"[Plot] Generating plots for Phase {phase}")
        try:
            output_dir = Path(self.results_dir)
            if phase == 1:
                plot_phase1_training(
                    self._load('phase1_training_results.json'),
                    output_dir,
                )
            elif phase == 2:
                plot_phase2_evaluation(
                    self._load('phase2_maddpg_results.json'),
                    output_dir,
                    self._load('phase2_rankings.json'),
                )
            elif phase == 3:
                plot_phase3_fgsm(
                    self._load('phase3_fgsm_results.json'),
                    output_dir,
                    self._load('phase3_rankings.json'),
                )
        except Exception as exc:
            logger.error(f"[Plot] Phase {phase} plot generation failed: {exc}")

    # â”€â”€ full pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_all(self):
        t0 = time.time()
        tr = self.run_training()
        p2 = self.evaluate_maddpg(tr)
        p3 = self.evaluate_fgsm(tr)
        summary = {
            'phase2': self._build_phase2_rankings(p2),
            'phase3': self._build_phase3_rankings(p3),
        }
        self._save(summary, 'experiment_summary_rankings.json')
        logger.info(f"All phases done in {(time.time()-t0)/3600:.2f} h  â†’  {self.results_dir}")

    # â”€â”€ persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _save(self, obj, filename: str):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
        logger.info(f"Saved {path}")

    def _load(self, filename: str) -> Dict:
        path = os.path.join(self.results_dir, filename)
        with open(path) as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',      default='experiment_config.json')
    ap.add_argument('--gpu',         type=int, default=0)
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--phase', choices=['all', 'train', 'paper1', 'paper2'], default='all')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='Smoke-test mode: 80 epochs / 3 eps / 64 steps â€” enough to '
                         'verify reward signal and full pipeline without a full run')
    ap.add_argument('--variants', default=None,
                    help='Comma-separated list of variant names to train; '
                         'all others are skipped (e.g. CC-Simple,LC-Duelling-GNN)')
    args = ap.parse_args()

    runner = StandaloneExperimentRunner(args.config, args.gpu, args.results_dir)

    if args.quick:
        runner.config['training'].update(
            epochs=1, episodes_per_epoch=1, timesteps_per_episode=10)
        for a in runner.config['attack_configs']:
            a['evaluation_episodes'] = 2
        runner.config.setdefault('paper1_eval', {})['evaluation_episodes'] = 2
        logger.info("Quick mode active")

    if args.smoke:
        runner.config['training'].update(
            epochs=80, episodes_per_epoch=3, timesteps_per_episode=64,
            batch_size=256)
        runner.config['training']['best_checkpoint'].update(
            warmup_epochs=25, validation_interval_epochs=5, validation_episodes=3)
        runner.config['training']['early_stopping'].update(
            min_epochs=50, patience_checks=8, smooth_window_checks=2)
        for a in runner.config['attack_configs']:
            a['evaluation_episodes'] = 10
        runner.config.setdefault('paper1_eval', {})['evaluation_episodes'] = 10
        runner.config.setdefault('load_sweep', {})['offered_loads'] = [0.5, 0.75, 1.0, 1.3]
        logger.info("Smoke mode active â€” 80 epochs / 3 eps / 64 steps")

    if args.variants:
        allowed = {v.strip() for v in args.variants.split(',')}
        runner.config['variants'] = [
            v for v in runner.config['variants'] if v['name'] in allowed
        ]
        logger.info("Variant filter active â€” training: %s", [v['name'] for v in runner.config['variants']])

    if   args.phase == 'all':    runner.run_all()
    elif args.phase == 'train':  runner.run_training()
    elif args.phase == 'paper1': runner.evaluate_maddpg()
    elif args.phase == 'paper2': runner.evaluate_fgsm()


if __name__ == '__main__':
    main()