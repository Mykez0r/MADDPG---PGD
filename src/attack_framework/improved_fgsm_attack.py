"""
Improved FGSM Attack Framework for MADDPG Routing.
Fixes attack objective and implements proper comparative metrics for thesis analysis.
"""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'thesis_graphs')

class FGSMAttackFramework:
    """
    Enhanced FGSM attack framework for adversarial analysis of MADDPG routing variants.
    """
    def __init__(self, epsilon: float = 0.05, attack_type: str = 'packet_loss'):
        """
        Args:
            epsilon: Perturbation magnitude (L-inf ball radius).
            attack_type: One of 'packet_loss', 'reward_minimize', 'confusion'.
        """
        self.epsilon = epsilon
        self.attack_type = attack_type
        # PGD controls: n_steps=1 → single-step FGSM (default); n_steps>1 runs
        # projected gradient descent with per-step size step_alpha (0 → epsilon).
        self.n_steps = 1
        self.step_alpha = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attack_stats: Dict = {
            'clean_rewards': [],
            'attacked_rewards': [],
            'clean_packet_loss': [],
            'attacked_packet_loss': [],
            'attack_success_count': 0,
            'total_attacks': 0,
        }

    def generate_adversarial_state(
        self,
        state: np.ndarray,
        agent_network,
        network_engine,
        agent_index: int,
        bandwidth_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Generate adversarial state using FGSM with proper attack objective.
        Args:
            state: Original 1-D state vector.
            agent_network: MADDPG Agent or MADDPG orchestrator.
            network_engine: Network environment engine.
            agent_index: Index of the target agent.
            bandwidth_indices: State indices that represent bandwidth (clamped to [0,1]).
        Returns:
            Perturbed state as a 1-D numpy array.
        """
        # Random-noise control: budget-matched L-inf perturbation with a RANDOM
        # sign per feature and NO gradient information. Same epsilon and same
        # domain clamping as the gradient attacks, so any effect isolates "does
        # perturbation direction matter?" from "does jostling an off-optimal
        # policy help?". Uses numpy RNG only (traffic is driven by Python's
        # random module), so clean/attacked runs stay paired on the same traffic.
        if self.attack_type == 'random':
            adv = np.asarray(state, dtype=np.float32).copy()
            signs = np.where(np.random.random(adv.shape) < 0.5, -1.0, 1.0).astype(np.float32)
            adv = adv + self.epsilon * signs
            if bandwidth_indices is not None:
                adv[bandwidth_indices] = np.clip(adv[bandwidth_indices], 0.0, 1.0)
            else:
                k = min(4, adv.shape[0])
                adv[:k] = np.clip(adv[:k], 0.0, 1.0)
            return adv

        # Resolve specific agent if the MADDPG orchestrator was passed
        maddpg_ref = agent_network if hasattr(agent_network, 'agents') else None
        if hasattr(agent_network, 'agents'):
            agent = agent_network.agents[agent_index]
        else:
            agent = agent_network

        # Create fresh state tensor for gradient computation
        state_np = np.asarray(state, dtype=np.float32)[np.newaxis, :]  # shape: (1, 26)
        state_tensor = torch.from_numpy(state_np.copy()).to(self.device).requires_grad_(True)

        # Save original training state
        was_training = agent.actor.training
        
        # PGD generalises FGSM. n_steps=1 (default) reproduces single-step FGSM
        # exactly; n_steps>1 runs projected gradient descent inside the L-inf
        # epsilon-ball around the original observation, taking per-step size
        # step_alpha (defaults to epsilon, which is correct for the single step).
        n_steps = max(1, int(getattr(self, 'n_steps', 1)))
        step_alpha = float(getattr(self, 'step_alpha', 0.0)) or self.epsilon

        def _actor_probs(cur):
            """Forward the (possibly GNN-encoded) working state through the actor."""
            current_state = cur
            gnn_proc = getattr(maddpg_ref, 'gnn_processor', None)
            if gnn_proc is not None and getattr(gnn_proc, 'available', False):
                # Apply the GNN to a batch where every other agent slot holds a
                # zero observation and only the target slot carries grad back to
                # the working state; padded to the full node count if needed.
                n_agents_gnn = gnn_proc.n_agents
                obs_dim_gnn = gnn_proc.obs_dim
                dummy = torch.zeros(n_agents_gnn, obs_dim_gnn, device=self.device)
                adv_obs = cur.squeeze(0)[:obs_dim_gnn]
                if adv_obs.shape[0] < obs_dim_gnn:
                    adv_obs = torch.cat(
                        [adv_obs,
                         torch.zeros(obs_dim_gnn - adv_obs.shape[0], device=self.device)]
                    )
                batch = dummy.clone()
                batch[agent_index % n_agents_gnn] = adv_obs
                n_relay = getattr(gnn_proc, 'n_relay_nodes', 0)
                if n_relay > 0:
                    relay_pad = torch.zeros(n_relay, obs_dim_gnn, device=self.device)
                    batch_full = torch.cat([batch, relay_pad], dim=0)
                else:
                    batch_full = batch
                out = gnn_proc(batch_full, gnn_proc.edge_index)
                current_state = out[agent_index % n_agents_gnn].unsqueeze(0)
            return agent.actor(current_state)

        def _objective(cur, probs):
            if self.attack_type == 'packet_loss':
                return self._packet_loss_objective(cur, probs, network_engine, agent_index)
            elif self.attack_type == 'reward_minimize':
                return self._reward_minimize_objective(cur, probs, network_engine, agent_index)
            elif self.attack_type == 'confusion':
                return self._confusion_objective(cur, probs, network_engine, agent_index)
            raise ValueError(f'Unknown attack type: {self.attack_type}')

        try:
            # Force gradient computation (overrides any torch.no_grad context)
            with torch.enable_grad():
                agent.actor.eval()  # eval mode → no BN/Dropout noise during the attack
                orig = state_tensor.detach().clone()
                adv = state_tensor  # requires_grad already set
                for _step in range(n_steps):
                    if adv.grad is not None:
                        adv.grad.zero_()
                    action_probs = _actor_probs(adv)
                    loss = _objective(adv, action_probs)
                    loss.backward()
                    if adv.grad is None:
                        raise RuntimeError("Gradients not computed. Gradient flow may be broken.")
                    with torch.no_grad():
                        adv = adv + step_alpha * torch.sign(adv.grad.data)
                        # project the accumulated perturbation back into the L-inf ball
                        adv = orig + torch.clamp(adv - orig, -self.epsilon, self.epsilon)
                        adv = self._apply_domain_constraints(adv, bandwidth_indices)
                    adv = adv.detach().requires_grad_(True)
                return adv.detach().cpu().numpy()[0]

        except Exception as e:
            logger.error(f'FGSM/PGD generation failed for agent {agent_index}: {str(e)}')
            return state
        finally:
            # Restore original training state
            if was_training:
                agent.actor.train()

    # ------------------------------------------------------------------
    # Critic-grounded attack
    # ------------------------------------------------------------------
    @staticmethod
    def _straight_through_block_onehot(a_soft: torch.Tensor, block_size: int) -> torch.Tensor:
        """Per-block argmax one-hot with a straight-through gradient.

        Forward pass is the hard one-hot (one path selected per K_PATHS block, as
        the environment decodes routing); backward pass passes the gradient of the
        soft actor output. Matches the block-projected actions the central critic
        was trained on.
        """
        n = a_soft.shape[-1]
        hard = torch.zeros_like(a_soft)
        for bs in range(0, n, block_size):
            be = min(bs + block_size, n)
            idx = a_soft[:, bs:be].argmax(dim=1)
            hard[torch.arange(a_soft.shape[0]), bs + idx] = 1.0
        return (hard - a_soft).detach() + a_soft

    def generate_adversarial_state_critic(
        self,
        state: np.ndarray,
        maddpg,
        agent_index: int,
        central_state: np.ndarray,
        clean_joint_onehot: np.ndarray,
        block_size: int,
        bandwidth_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Critic-grounded FGSM: perturb agent i's observation to minimise the
        agent's own central critic value Q(s, a).

        The true central state ``s`` and the other agents' (block-onehot) actions
        are held fixed; only agent i's action a_i = π_i(o_i+δ) — projected with a
        straight-through block one-hot — varies through the perturbation. The
        gradient direction that most *decreases* Q is, by construction, the
        direction the agent's own value model rates as most harmful, so (unlike the
        congestion proxy) it cannot accidentally improve routing.
        """
        agent = maddpg.agents[agent_index]
        device = agent.actor.device
        was_training_a = agent.actor.training
        was_training_c = agent.critic.training
        try:
            with torch.enable_grad():
                agent.actor.eval()
                agent.critic.eval()
                s = torch.tensor(np.asarray(state, dtype=np.float32)[None, :],
                                 device=device).requires_grad_(True)
                cs = torch.tensor(np.asarray(central_state, dtype=np.float32)[None, :],
                                  device=device)                         # fixed true state
                joint = torch.tensor(np.asarray(clean_joint_onehot, dtype=np.float32),
                                     device=device)                      # [n_agents, n_actions]

                a_soft = agent.actor(s)                                   # [1, n_actions]
                a_st = self._straight_through_block_onehot(a_soft, block_size)
                joint_b = joint.clone()
                joint_b[agent_index] = a_st.squeeze(0)
                joint_flat = joint_b.view(1, -1)                         # [1, n_agents*n_actions]

                q = agent.critic(cs, joint_flat)                        # [1, 1]
                # Ascend (-Q) ⇒ descend Q: steer toward the lowest-value action.
                loss = -q.sum()

                if s.grad is not None:
                    s.grad.zero_()
                loss.backward()
                if s.grad is None:
                    raise RuntimeError("Critic-grounded attack produced no gradient.")

                perturbation = self.epsilon * torch.sign(s.grad.data)
                adv = self._apply_domain_constraints(s + perturbation, bandwidth_indices)
                return adv.detach().cpu().numpy()[0]
        except Exception as e:
            logger.error(f'Critic-grounded FGSM failed for agent {agent_index}: {str(e)}')
            return state
        finally:
            if was_training_a:
                agent.actor.train()
            if was_training_c:
                agent.critic.train()

    # ------------------------------------------------------------------
    # Action-aligned congestion signal
    # ------------------------------------------------------------------
    def _per_action_util(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
    ) -> Optional[torch.Tensor]:
        """Extract the per-action k-path bottleneck utilisation from the observation.

        The observation contains an n_dest × K_PATHS block of path bottleneck
        utilisations laid out in the *same order as the action matrix* (for each
        destination d, K_PATHS consecutive slots). Action i therefore selects the
        path whose congestion is slot i of this block — a 1:1, environment-grounded
        congestion signal. (The previous implementation cycled action index over
        neighbour bandwidth via ``i % num_neighbors``, which mapped congestion to
        the wrong actions and produced near-random gradients.)

        Returns a detached (1, L) tensor of utilisations aligned to the first L
        actions, or None if the layout cannot be resolved.
        """
        n_dest = getattr(network_engine, 'n_destinations', None)
        max_neighbors = getattr(network_engine, 'max_neighbors', None)
        if n_dest is None or max_neighbors is None or n_dest == 0:
            return None
        num_actions = action_probs.shape[-1]
        k_paths = max(1, num_actions // n_dest)
        block_w = n_dest * k_paths
        # Layout: [mn bw | 3 queue | n_dest dest-indicator | 3 link-stats | k-path utils | mean-hops]
        util_start = max_neighbors + n_dest + 6
        if util_start >= state.shape[-1]:
            return None
        # The observation is truncated to state_dims, which drops the final K_PATHS
        # constructed slots (mean-hops + the last destination's last two path utils),
        # so the retained k-path block is K*(n_dest-1)+1 slots — action-aligned for
        # the first L actions. Clamp to whatever is present rather than bail.
        util_end = min(util_start + block_w, state.shape[-1])
        # Detach: the congestion target is fixed by the *true* current state; the
        # attack should only differentiate through the policy (action_probs), not
        # inflate the utilisation features themselves.
        util = state[:, util_start:util_end].detach()
        L = min(util.shape[-1], num_actions)
        return util[:, :L]

    # ------------------------------------------------------------------
    # Attack objectives
    # ------------------------------------------------------------------
    def _packet_loss_objective(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
        agent_index: int,
    ) -> torch.Tensor:
        """Encourage congested-path selection to maximise packet loss.

        Weights each action by a sharpened (sigmoid) function of its true k-path
        bottleneck utilisation, then rewards placing probability mass on the most
        congested paths. Gradient ascent perturbs the observation so the policy
        prefers congested paths.
        """
        util = self._per_action_util(state, action_probs, network_engine)
        if util is None:
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)
        L = util.shape[-1]
        probs = action_probs[:, :L]
        # High utilisation → weight ≈ 1 (congested); low utilisation → weight ≈ 0.
        congestion_weights = torch.sigmoid((util - 0.5) * 10.0)
        congestion_loss = torch.sum(probs * congestion_weights)
        return congestion_loss  # gradient ascent maximises congestion-path selection

    def _reward_minimize_objective(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
        agent_index: int,
    ) -> torch.Tensor:
        """Minimise expected routing quality (a smooth, linear reward proxy).

        Routing quality is high when chosen paths have low utilisation, so the
        expected utilisation Σ π(a)·util(a) is a differentiable proxy for *negative*
        reward. Gradient ascent maximises expected utilisation, steering the policy
        toward worse (more saturated) paths.
        """
        util = self._per_action_util(state, action_probs, network_engine)
        if util is None:
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)
        L = util.shape[-1]
        probs = action_probs[:, :L]
        expected_util = torch.sum(probs * util)
        return expected_util  # gradient ascent maximises expected path utilisation

    def _confusion_objective(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
        agent_index: int,
    ) -> torch.Tensor:
        """Targeted misrouting toward the single most-congested action.

        Identifies the worst action (highest k-path bottleneck utilisation) from the
        true current state and maximises the policy's probability of selecting it.
        Unlike entropy maximisation — which can inadvertently load-balance and improve
        performance — this concentrates traffic on the single most saturated path.
        """
        util = self._per_action_util(state, action_probs, network_engine)
        if util is None:
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)
        L = util.shape[-1]
        probs = action_probs[:, :L]
        # Worst action = highest path utilisation (most congested).
        worst_action = util.argmax(dim=-1).detach()  # shape: (1,)
        worst_prob = probs.gather(1, worst_action.unsqueeze(1))  # shape: (1, 1)
        # Gradient ascent on log(P(worst)) maximises the probability of the worst action.
        return torch.log(worst_prob + 1e-8).mean()

    # ------------------------------------------------------------------
    # Domain constraints
    # ------------------------------------------------------------------
    def _apply_domain_constraints(
        self,
        adversarial_state: torch.Tensor,
        bandwidth_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Clamp bandwidth features to valid [0, 1] range."""
        constrained = adversarial_state.clone()
        if bandwidth_indices is not None:
            constrained[:, bandwidth_indices] = torch.clamp(
                constrained[:, bandwidth_indices], 0.0, 1.0
            )
        else:
            bw_size = min(4, adversarial_state.shape[1])
            constrained[:, :bw_size] = torch.clamp(
                constrained[:, :bw_size], 0.0, 1.0
            )
        return constrained

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def update_statistics(
        self,
        clean_reward: float,
        attacked_reward: float,
        clean_packet_loss: float,
        attacked_packet_loss: float,
    ):
        """Track per-step attack effectiveness statistics."""
        self.attack_stats['clean_rewards'].append(clean_reward)
        self.attack_stats['attacked_rewards'].append(attacked_reward)
        self.attack_stats['clean_packet_loss'].append(clean_packet_loss)
        self.attack_stats['attacked_packet_loss'].append(attacked_packet_loss)
        self.attack_stats['total_attacks'] += 1
        
        if (
            attacked_packet_loss > clean_packet_loss * 1.1 or
            attacked_reward < clean_reward * 0.9
        ):
            self.attack_stats['attack_success_count'] += 1

class MADDPGRobustnessEvaluator:
    """Comprehensive evaluation framework for MADDPG variant robustness."""
    def __init__(self, maddpg_variants: Dict, network_engine):
        """
        Args:
            maddpg_variants: {name: maddpg_instance} mapping.
            network_engine: Network simulation engine.
        """
        self.maddpg_variants = maddpg_variants
        self.network_engine = network_engine
        self.results: Dict = defaultdict(lambda: defaultdict(list))

    def evaluate_attack_effectiveness(
        self,
        attack_framework: FGSMAttackFramework,
        num_episodes: int = 100,
        epsilon_values: List[float] = None,
        attack_types: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate attack effectiveness across all variants and epsilon values."""
        if epsilon_values is None:
            epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
        if attack_types is None:
            attack_types = ['packet_loss']

        evaluation_results = {}
        for variant_name, maddpg_agent in self.maddpg_variants.items():
            logger.info('Evaluating %s ...', variant_name)
            variant_results = {}
            for attack_type in attack_types:
                for epsilon in epsilon_values:
                    logger.info('  type = %s, epsilon = %.3f', attack_type, epsilon)
                    attack_framework.attack_type = attack_type
                    attack_framework.epsilon = epsilon

                    clean_metrics = self._run_episodes(maddpg_agent, num_episodes, attack=False)
                    attacked_metrics = self._run_episodes(
                        maddpg_agent, num_episodes, attack=True, attack_framework=attack_framework,
                    )

                    variant_results[f'{attack_type}_epsilon_{epsilon}'] = {
                        'clean': clean_metrics,
                        'attacked': attacked_metrics,
                        'comparison': self._compute_comparison_metrics(
                            clean_metrics, attacked_metrics
                        ),
                        'run_config': {
                            'attack_type': attack_type,
                            'epsilon': float(epsilon),
                            'evaluation_episodes': int(num_episodes),
                        },
                    }
            evaluation_results[variant_name] = variant_results
        return evaluation_results

    def _run_episodes(
        self,
        maddpg_agent,
        num_episodes: int,
        attack: bool = False,
        attack_framework: Optional[FGSMAttackFramework] = None,
    ) -> Dict:
        episode_rewards, episode_packet_losses, episode_util_dists = [], [], []
        
        for _ in range(num_episodes):
            self.network_engine.reset()
            total_reward = total_packet_loss = total_packets_sent = 0
            
            # Initialise states once at episode start
            all_hosts = self.network_engine.get_all_hosts()
            states = [self.network_engine.get_state(host, 1) for host in all_hosts]

            for _ in range(256):
                # Apply adversarial perturbation if attacking
                if attack and attack_framework is not None:
                    states = [
                        attack_framework.generate_adversarial_state(
                            state, maddpg_agent, self.network_engine, agent_idx
                        )
                        for agent_idx, state in enumerate(states)
                    ]

                actions = maddpg_agent.choose_action(states)
                next_states, rewards, packet_loss_info = self._execute_actions(actions)

                total_reward += sum(rewards)
                total_packet_loss += packet_loss_info['packets_lost']
                total_packets_sent += packet_loss_info['packets_sent']

                states = next_states  # ← advance state for next timestep
            
            episode_rewards.append(total_reward)
            episode_packet_losses.append(
                total_packet_loss / max(total_packets_sent, 1) * 100
            )
            episode_util_dists.append(
                self.network_engine.get_link_utilization_distribution()
            )
            
        return {
            'rewards': episode_rewards,
            'packet_losses': episode_packet_losses,
            'utilization_distributions': episode_util_dists,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_packet_loss': float(np.mean(episode_packet_losses)),
            'std_packet_loss': float(np.std(episode_packet_losses)),
        }

def _execute_actions(
    self, actions: List[int]
) -> Tuple[List, List[float], Dict]:
    """Execute joint actions in the real NetworkEngine."""
    all_hosts = self.network_engine.get_all_hosts()
    n_actions = self.network_engine.n_actions  # adapts to per-destination action space

    # Convert actions → arrays for NetworkEngine.step().
    # choose_action() already returns np.ndarray probability vectors; integer
    # indices (legacy callers) are converted to one-hot.
    action_arrays = []
    for action_idx in actions:
        if isinstance(action_idx, np.ndarray):
            action_arrays.append(action_idx.astype(np.float32))
        else:
            one_hot = np.zeros(n_actions, dtype=np.float32)
            one_hot[int(action_idx) % n_actions] = 1.0
            action_arrays.append(one_hot)

    next_states, rewards, info = self.network_engine.step(action_arrays)

    packet_loss_info = {
        'packets_lost': info.get('packets_dropped', 0),
        'packets_sent': max(info.get('packets_sent', 1), 1),
    }
    return next_states, rewards, packet_loss_info

    def _compute_comparison_metrics(
        self, clean_metrics: Dict, attacked_metrics: Dict
    ) -> Dict:
        clean_mean = clean_metrics['mean_reward']
        reward_degradation = (
            (clean_mean - attacked_metrics['mean_reward']) / clean_mean * 100
            if clean_mean != 0 else 0.0
        )
        
        packet_loss_increase = (
            attacked_metrics['mean_packet_loss'] - clean_metrics['mean_packet_loss']
        )
        
        clean_rewards = np.array(clean_metrics['rewards'])
        attacked_rewards = np.array(attacked_metrics['rewards'])
        successful_attacks = int(np.sum(attacked_rewards < clean_rewards * 0.9))
        attack_success_rate = successful_attacks / max(len(clean_rewards), 1) * 100
        
        clean_std = clean_metrics['std_reward']
        variance_change = (
            (attacked_metrics['std_reward'] - clean_std) / clean_std * 100
            if clean_std != 0 else 0.0
        )
        
        return {
            'reward_degradation_percent': reward_degradation,
            'packet_loss_increase_percent': packet_loss_increase,
            'attack_success_rate_percent': attack_success_rate,
            'variance_change_percent': variance_change,
            'robustness_score': max(0.0, 100 - reward_degradation - packet_loss_increase),
        }

class ThesisVisualizationSuite:
    """Generate publication-quality graphs for thesis inclusion."""
    def __init__(
        self, results_data: Dict, save_path: str = _DEFAULT_SAVE_PATH,
    ):
        self.results_data = results_data
        self.save_path = os.path.abspath(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette('husl')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })

    def generate_all_thesis_plots(self):
        self.plot_architecture_robustness()
        self.plot_attack_intensity_analysis()
        self.plot_performance_degradation_matrix()
        self.plot_reward_packet_loss_tradeoffs()
        self.plot_gnn_robustness_impact()
        self.plot_attack_success_rates()
        logger.info('All thesis plots saved to %s', self.save_path)

    def _save(self, filename: str):
        """Save current figure and close it."""
        path = os.path.join(self.save_path, filename)
        plt.savefig(path)
        plt.close()
        logger.info('Saved %s', path)

    def _epsilon_values(self) -> List[float]:
        if not self.results_data: return []
        first_variant = next(iter(self.results_data.values()))
        return [
            float(k.replace('epsilon_', '')) 
            for k in first_variant.keys() if k.startswith('epsilon_')
        ]

    def plot_architecture_robustness(self):
        epsilon_values = self._epsilon_values()
        variants = list(self.results_data.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for variant in variants:
            rd = [
                self.results_data[variant][f'epsilon_{e}']['comparison']['reward_degradation_percent']
                for e in epsilon_values
            ]
            pl = [
                self.results_data[variant][f'epsilon_{e}']['comparison']['packet_loss_increase_percent']
                for e in epsilon_values
            ]
            ax1.plot(epsilon_values, rd, 'o-', label=variant, linewidth=2, markersize=6)
            ax2.plot(epsilon_values, pl, 's-', label=variant, linewidth=2, markersize=6)
            
        for ax, ylabel, title in [
            (ax1, 'Reward Degradation (%)', 'Reward Degradation vs Attack Intensity'),
            (ax2, 'Packet Loss Increase (%)', 'Packet Loss Increase vs Attack Intensity'),
        ]:
            ax.set_xlabel('Attack Intensity (ε)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        self._save('architecture_robustness_comparison.png')

    def plot_attack_intensity_analysis(self):
        epsilon_values = self._epsilon_values()
        variants = list(self.results_data.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        heatmap_data = [
            [
                self.results_data[v][f'epsilon_{e}']['comparison']['robustness_score']
                for e in epsilon_values
            ] for v in variants
        ]
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(range(len(epsilon_values)))
        ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values])
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Robustness Score', rotation=270, labelpad=20)
        
        for i in range(len(variants)):
            for j in range(len(epsilon_values)):
                ax.text(j, i, f'{heatmap_data[i][j]:.1f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_xlabel('Attack Intensity (ε)')
        ax.set_ylabel('MADDPG Variant')
        ax.set_title('Robustness Score Heatmap')
        plt.tight_layout()
        self._save('attack_intensity_heatmap.png')

    def plot_performance_degradation_matrix(self):
        epsilon_values = self._epsilon_values()
        variants = list(self.results_data.keys())
        metrics = [
            ('reward_degradation_percent', 'Reward Degradation (%)'),
            ('packet_loss_increase_percent', 'Packet Loss Increase (%)'),
            ('attack_success_rate_percent', 'Attack Success Rate (%)'),
            ('variance_change_percent', 'Performance Variance Change (%)'),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        x = np.arange(len(epsilon_values))
        width = 0.8 / len(variants)
        
        for ax, (metric, metric_name) in zip(axes.flat, metrics):
            for i, variant in enumerate(variants):
                values = [
                    self.results_data[variant][f'epsilon_{e}']['comparison'][metric]
                    for e in epsilon_values
                ]
                ax.bar(x + i * width, values, width, label=variant, alpha=0.8)
                
            ax.set_xlabel('Attack Intensity (ε)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by Variant')
            ax.set_xticks(x + width * (len(variants) - 1) / 2)
            ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        self._save('performance_degradation_matrix.png')

    def plot_reward_packet_loss_tradeoffs(self):
        colors = sns.color_palette('husl', len(self.results_data))
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for idx, (variant, variant_data) in enumerate(self.results_data.items()):
            clean_r, clean_pl, att_r, att_pl = [], [], [], []
            for eps_key, eps_data in variant_data.items():
                if not eps_key.startswith('epsilon_'): continue
                clean_r.append(eps_data['clean']['mean_reward'])
                clean_pl.append(eps_data['clean']['mean_packet_loss'])
                att_r.append(eps_data['attacked']['mean_reward'])
                att_pl.append(eps_data['attacked']['mean_packet_loss'])
                
            ax.scatter(clean_pl, clean_r, c=[colors[idx]], s=100, marker='o', 
                       label=f'{variant} (Clean)', alpha=0.8)
            ax.scatter(att_pl, att_r, c=[colors[idx]], s=100, marker='x', 
                       label=f'{variant} (Attacked)', alpha=0.8)
            
            for cr, cpl, ar, apl in zip(clean_r, clean_pl, att_r, att_pl):
                ax.annotate('', xy=(apl, ar), xytext=(cpl, cr),
                            arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.5))
                            
        ax.set_xlabel('Packet Loss (%)')
        ax.set_ylabel('Average Reward')
        ax.set_title('Reward vs Packet Loss Trade-offs Under FGSM Attack')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save('reward_packet_loss_tradeoffs.png')

    def plot_gnn_robustness_impact(self):
        epsilon_values = self._epsilon_values()
        gnn_variants = [k for k in self.results_data if 'GNN' in k]
        non_gnn_variants = [k for k in self.results_data if 'GNN' not in k]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for variant in gnn_variants:
            scores = [self.results_data[variant][f'epsilon_{e}']['comparison']['robustness_score'] for e in epsilon_values]
            ax1.plot(epsilon_values, scores, 'o-', label=variant, linewidth=2)
        for variant in non_gnn_variants:
            scores = [self.results_data[variant][f'epsilon_{e}']['comparison']['robustness_score'] for e in epsilon_values]
            ax1.plot(epsilon_values, scores, 's--', label=variant, linewidth=2)
            
        if gnn_variants and non_gnn_variants:
            gnn_sr = [np.mean([self.results_data[v][f'epsilon_{e}']['comparison']['attack_success_rate_percent'] for v in gnn_variants]) for e in epsilon_values]
            non_gnn_sr = [np.mean([self.results_data[v][f'epsilon_{e}']['comparison']['attack_success_rate_percent'] for v in non_gnn_variants]) for e in epsilon_values]
            ax2.plot(epsilon_values, gnn_sr, 'o-', label='With GNN', linewidth=2, markersize=8)
            ax2.plot(epsilon_values, non_gnn_sr, 's-', label='Without GNN', linewidth=2, markersize=8)
            
        for ax, ylabel, title in [
            (ax1, 'Robustness Score', 'GNN vs Non-GNN Robustness'),
            (ax2, 'Attack Success Rate (%)', 'Attack Success Rate: GNN vs Non-GNN'),
        ]:
            ax.set_xlabel('Attack Intensity (ε)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        self._save('gnn_robustness_impact.png')

    def plot_attack_success_rates(self):
        epsilon_values = self._epsilon_values()
        variants = list(self.results_data.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(epsilon_values))
        width = 0.8 / len(variants)
        
        for i, variant in enumerate(variants):
            success_rates = [
                self.results_data[variant][f'epsilon_{e}']['comparison']['attack_success_rate_percent']
                for e in epsilon_values
            ]
            bars = ax.bar(x + i * width, success_rates, width, label=variant, alpha=0.8)
            for bar, rate in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
                
        ax.set_xlabel('Attack Intensity (ε)')
        ax.set_ylabel('Attack Success Rate (%)')
        ax.set_title('Attack Success Rate Across MADDPG Variants')
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        self._save('attack_success_rates.png')

def generate_mock_results() -> Dict:
    """Generate mock results for demonstration purposes."""
    variants = [
        'CC-Simple', 'CC-Duelling', 'LC-Duelling',
        'CC-Simple-GNN', 'CC-Duelling-GNN', 'LC-Duelling-GNN',
        'LC-Simple',
    ]
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    rng = np.random.default_rng(42)
    
    results: Dict = {}
    for variant in variants:
        base_robustness = 85 if 'GNN' in variant else 80
        if 'LC' in variant: base_robustness += 5
        if 'Duelling' in variant: base_robustness += 3
        
        variant_results = {}
        for eps in epsilon_values:
            robustness_score = max(20.0, base_robustness - eps * 100)
            variant_results[f'epsilon_{eps}'] = {
                'clean': {
                    'mean_reward': 1400 + float(rng.normal(0, 20)),
                    'mean_packet_loss': 0.5 + float(rng.normal(0, 0.1)),
                },
                'attacked': {
                    'mean_reward': 1400 - eps * 200 + float(rng.normal(0, 30)),
                    'mean_packet_loss': 0.5 + eps * 10 + float(rng.normal(0, 0.15)),
                },
                'comparison': {
                    'reward_degradation_percent': eps * 15 + float(rng.normal(0, 2)),
                    'packet_loss_increase_percent': eps * 20 + float(rng.normal(0, 3)),
                    'attack_success_rate_percent': min(95.0, eps * 300 + float(rng.normal(0, 5))),
                    'variance_change_percent': eps * 25 + float(rng.normal(0, 4)),
                    'robustness_score': robustness_score,
                },
            }
        results[variant] = variant_results
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('FGSM Attack Framework for MADDPG Routing Analysis')
    print('=' * 60)
    mock_results = generate_mock_results()
    viz_suite = ThesisVisualizationSuite(mock_results)
    viz_suite.generate_all_thesis_plots()
    print('\nThesis-quality plots generated successfully!')
    print(f'Plots saved to: {viz_suite.save_path}')
