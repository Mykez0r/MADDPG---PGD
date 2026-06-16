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
        
        try:
            # Force gradient computation (overrides any torch.no_grad context)
            with torch.enable_grad():
                agent.actor.eval()  # Use eval mode for consistent attack (prevents BN/Dropout noise)
                
                # Handle GNN processing differentiably if the variant uses a GNN.
                # GNN now lives on the MADDPG orchestrator and needs all agents'
                # observations, but for adversarial input computation we only
                # have access to the single target agent's state tensor.
                # Strategy: apply GNN to a batch where all other agents receive
                # their current (unperturbed) observations; only the target slot
                # retains the grad_fn back through state_tensor.
                current_state = state_tensor
                gnn_proc = getattr(maddpg_ref, 'gnn_processor', None)
                if gnn_proc is not None and getattr(gnn_proc, 'available', False):
                    # Build a [n_agents, obs_dim] batch; target agent keeps grad
                    n_agents_gnn = gnn_proc.n_agents
                    obs_dim_gnn = gnn_proc.obs_dim
                    dummy = torch.zeros(n_agents_gnn, obs_dim_gnn,
                                        device=self.device)
                    # Overwrite the target agent's slot with the attacked state
                    # (trimmed/padded to obs_dim to handle shape mismatches)
                    adv_obs = state_tensor.squeeze(0)[:obs_dim_gnn]
                    if adv_obs.shape[0] < obs_dim_gnn:
                        adv_obs = torch.cat(
                            [adv_obs,
                             torch.zeros(obs_dim_gnn - adv_obs.shape[0], device=self.device)]
                        )
                    batch = dummy.clone()
                    batch[agent_index % n_agents_gnn] = adv_obs
                    # Full-graph mode: pad relay (switch) nodes with zeros so the
                    # edge_index referencing all n_total nodes doesn't crash.
                    n_relay = getattr(gnn_proc, 'n_relay_nodes', 0)
                    if n_relay > 0:
                        relay_pad = torch.zeros(n_relay, obs_dim_gnn, device=self.device)
                        batch_full = torch.cat([batch, relay_pad], dim=0)
                    else:
                        batch_full = batch
                    out = gnn_proc(batch_full, gnn_proc.edge_index)  # [n_total, obs_dim]
                    current_state = out[agent_index % n_agents_gnn].unsqueeze(0)
                action_probs = agent.actor(current_state)

                if self.attack_type == 'packet_loss':
                    loss = self._packet_loss_objective(
                        state_tensor, action_probs, network_engine, agent_index
                    )
                elif self.attack_type == 'reward_minimize':
                    loss = self._reward_minimize_objective(
                        state_tensor, action_probs, network_engine, agent_index
                    )
                elif self.attack_type == 'confusion':
                    loss = self._confusion_objective(
                        state_tensor, action_probs, network_engine, agent_index
                    )
                else:
                    raise ValueError(f'Unknown attack type: {self.attack_type}')

                # Zero any existing gradients before backward
                if state_tensor.grad is not None:
                    state_tensor.grad.zero_()
                
                loss.backward()

                # Check if gradients were computed
                if state_tensor.grad is None:
                    raise RuntimeError("Gradients not computed. Gradient flow may be broken or model parameters are non-differentiable.")

                perturbation = self.epsilon * torch.sign(state_tensor.grad.data)
                adversarial_state = state_tensor + perturbation
                adversarial_state = self._apply_domain_constraints(
                    adversarial_state, bandwidth_indices
                )
                
                return adversarial_state.detach().cpu().numpy()[0]

        except Exception as e:
            logger.error(f'FGSM generation failed for agent {agent_index}: {str(e)}')
            return state
        finally:
            # Restore original training state
            if was_training:
                agent.actor.train()

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
        """Encourage congested-path selection to maximise packet loss."""
        num_neighbors = network_engine.get_number_neighbors(
            network_engine.get_all_hosts()[agent_index]
        )
        if num_neighbors == 0:
            # Return zero loss that is still connected to the graph to avoid grad errors
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)

        num_actions = action_probs.shape[-1]  # n_dest × K_PATHS (e.g. 63 = 21 × 3)
        bandwidth_states = state[:, :num_neighbors]  # shape: (1, num_neighbors)

    # Build per-action congestion weight by cycling over neighbor bandwidths.
    # Action i is associated with the link to neighbor (i % num_neighbors).
    # This maps each pre-computed path to its first-hop link congestion.
        neighbor_indices = torch.arange(num_actions, device=self.device) % num_neighbors
        per_action_bw = bandwidth_states[:, neighbor_indices]  # shape: (1, num_actions)
    
    # High congestion (low bandwidth) → high weight → push agent toward congested paths
        congestion_weights = torch.sigmoid((1.0 - per_action_bw) * 10.0)
        congestion_loss = torch.sum(action_probs * congestion_weights)
        return -congestion_loss  # maximise congestion selection

    def _reward_minimize_objective(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
        agent_index: int,
    ) -> torch.Tensor:
        """Minimise expected routing quality by pushing toward low-bandwidth action choices.

        Uses the same bandwidth-aware setup as _packet_loss_objective but minimises the
        expected bandwidth utility directly (a differentiable reward proxy), rather than
        using uniform weights whose gradient is identically zero.
        """
        num_neighbors = network_engine.get_number_neighbors(
            network_engine.get_all_hosts()[agent_index]
        )
        if num_neighbors == 0:
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)

        num_actions = action_probs.shape[-1]
        bandwidth_states = state[:, :num_neighbors]  # shape: (1, num_neighbors)
        neighbor_indices = torch.arange(num_actions, device=self.device) % num_neighbors
        per_action_bw = bandwidth_states[:, neighbor_indices]  # shape: (1, num_actions)

        # Negate expected bandwidth so FGSM (gradient ascent) minimises it —
        # this deceives the agent into believing low-BW (bad) paths are attractive.
        expected_bw = torch.sum(action_probs * per_action_bw)
        return -expected_bw  # gradient pushes agent toward low-bandwidth (bad) paths

    def _confusion_objective(
        self,
        state: torch.Tensor,
        action_probs: torch.Tensor,
        network_engine,
        agent_index: int,
    ) -> torch.Tensor:
        """Targeted misrouting: cross-entropy toward the most-congested (worst) action.

        Unlike entropy maximisation — which can inadvertently produce load-balancing and
        improve performance — this forces the agent to assign maximum probability to the
        single action that routes traffic through the most congested link.
        """
        num_neighbors = network_engine.get_number_neighbors(
            network_engine.get_all_hosts()[agent_index]
        )
        if num_neighbors == 0:
            return (state.sum() * 0.0) + (action_probs.sum() * 0.0)

        num_actions = action_probs.shape[-1]
        bandwidth_states = state[:, :num_neighbors]  # shape: (1, num_neighbors)
        neighbor_indices = torch.arange(num_actions, device=self.device) % num_neighbors
        per_action_bw = bandwidth_states[:, neighbor_indices]  # shape: (1, num_actions)

        # Most-congested action = lowest bandwidth among first-hop links.
        worst_action = per_action_bw.argmin(dim=-1).detach()  # shape: (1,)
        worst_prob = action_probs.gather(1, worst_action.unsqueeze(1))  # shape: (1, 1)

        # Maximise probability of worst action (minimise negative log-likelihood).
        return -torch.log(worst_prob + 1e-8).mean()

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
