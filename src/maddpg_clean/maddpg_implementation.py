"""
Clean MADDPG Implementation - Standalone version
Based on working components from student's code but reimplemented cleanly
"""

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional
import random
from collections import deque

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = 42):
    """Centralised seed setup for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seeds set to %d", seed)


class ActorNetwork(nn.Module):
    """Clean Actor Network implementation."""

    def __init__(self, input_dims: int, fc1_dims: int, fc2_dims: int,
                 n_actions: int, name: str, chkpt_dir: str,
                 device: Optional[str] = None):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_actor.pth')

        # Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_out = nn.Linear(fc2_dims, n_actions)

        self.init_weights()

        self.device = (
            torch.device(device) if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.to(self.device)

    def init_weights(self):
        """Initialise network weights with Xavier uniform."""
        for layer in [self.fc1, self.fc2, self.action_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.action_out(x))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            # weights_only=True avoids arbitrary code execution on load
            self.load_state_dict(
                torch.load(self.checkpoint_file, weights_only=True)
            )


class CriticNetwork(nn.Module):
    """Critic Network with support for Simple Q and Duelling Q architectures."""

    def __init__(self, input_dims: int, fc1_dims: int, fc2_dims: int,
                 n_agents: int, n_actions: int, name: str, chkpt_dir: str,
                 action_input_dims: Optional[int] = None,
                 network_type: str = 'simple_q_network'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_input_dims = action_input_dims or (n_agents * n_actions)
        self.network_type = network_type
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_critic.pth')

        # Shared trunk: concatenate state + action
        self.fc1 = nn.Linear(input_dims + self.action_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        if network_type == 'duelling_q_network':
            # Duelling streams: value V(s) and advantage A(s, a)
            self.value_stream = nn.Linear(fc2_dims, 1)
            self.advantage_stream = nn.Linear(fc2_dims, 1)
        else:  # simple_q_network
            self.q_head = nn.Linear(fc2_dims, 1)

        self.init_weights()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))

        if self.network_type == 'duelling_q_network':
            # With the action already concatenated to the input, both duelling
            # streams should remain scalar for the specific (s, a) pair.
            value = self.value_stream(x)                          # [B, 1]
            advantage = self.advantage_stream(x)                  # [B, 1]
            return value + advantage                              # [B, 1]
        else:
            return self.q_head(x)                                  # [B, 1]

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(
                torch.load(self.checkpoint_file, weights_only=True)
            )


class GNNProcessor(nn.Module):
    """Multi-agent GNN: encodes [n_agents, obs_dim] jointly on the agent graph.

    Each trainable agent is a graph node whose feature vector is its full
    obs_dim-dimensional observation.  GraphConv aggregates neighbour information
    and a residual skip-connection preserves the original state so the network
    starts as a near-identity transform and degrades gracefully when
    torch_geometric is unavailable.

    The GNN lives in MADDPG (not in Agent) because it requires all agents'
    observations simultaneously.  Workers receive serialised weights and the
    stored edge_index so they can apply the same transform during rollout,
    keeping the training and execution distributions aligned.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, n_agents: int,
                 adjacency: Optional[List[List[int]]] = None,
                 n_relay_nodes: int = 0):
        """Initialise GNN encoder.

        When ``n_relay_nodes > 0`` (Option-B / full-graph mode), the GNN
        processes ``n_agents + n_relay_nodes`` nodes.  Relay nodes (transit
        switches) are injected with zero feature vectors so they act purely
        as message-passing intermediaries — their edges propagate agent
        observations two hops through the physical topology, letting each
        access-node implicitly observe congestion at peer nodes that share
        the same upstream switch.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_relay_nodes = n_relay_nodes
        self._n_total = n_agents + n_relay_nodes

        try:
            from torch_geometric.nn import GraphConv
            self.conv1 = GraphConv(obs_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, obs_dim)
            self.available = True
        except ImportError:
            logger.warning("torch_geometric not available – GNN disabled.")
            self.available = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build edge_index from the adjacency list.
        # When n_relay_nodes > 0, adjacency covers all _n_total nodes;
        # relay node indices start at n_agents.
        if adjacency is not None:
            src, dst = [], []
            for i, neighbours in enumerate(adjacency):
                for j in neighbours:
                    src.append(i)
                    dst.append(j)
            if src:
                ei = torch.tensor([src, dst], dtype=torch.long)
            else:
                # No edges — use self-loops over all nodes so GraphConv doesn't crash
                ei = torch.stack([torch.arange(self._n_total), torch.arange(self._n_total)])
        else:
            # Fully-connected fallback among agents only (no relay nodes)
            pairs = [(a, b) for a in range(n_agents) for b in range(n_agents) if a != b]
            src, dst = zip(*pairs) if pairs else ([], [])
            ei = torch.tensor([list(src), list(dst)], dtype=torch.long)

        self.register_buffer('edge_index', ei)

        if self.available:
            self.to(self.device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [N, obs_dim].  Returns [N, obs_dim] with residual skip."""
        if not self.available:
            return x
        h = F.relu(self.conv1(x, edge_index))
        return F.relu(self.conv2(h, edge_index) + x)

    # ── Inference (no-grad, numpy in/out) ────────────────────────────────────

    def process_observations(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """Encode a list of per-agent obs arrays.  Used during rollout.

        Relay nodes receive zero feature vectors and participate only as
        message-passing intermediaries.  Only the first ``n_agents`` rows
        of the GNN output are returned.
        """
        if not self.available:
            return observations
        x_agents = np.stack(observations)  # [n_agents, obs_dim]
        if self.n_relay_nodes > 0:
            relay_zeros = np.zeros((self.n_relay_nodes, self.obs_dim), dtype=np.float32)
            x_np = np.concatenate([x_agents, relay_zeros], axis=0)  # [n_total, obs_dim]
        else:
            x_np = x_agents
        x = torch.tensor(x_np, dtype=torch.float).to(self.device)
        with torch.no_grad():
            out = self(x, self.edge_index)  # [n_total, obs_dim]
        return [out[i].cpu().numpy() for i in range(self.n_agents)]

    # ── Training (differentiable) ─────────────────────────────────────────────

    def process_batch(self, states_t: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode a list of per-agent [B, obs_dim] tensors.

        Returns a list of enhanced [B, obs_dim] tensors that carry a grad_fn
        back through the GNN parameters.  Uses the standard PyG batch trick:
        the single-graph edge_index is tiled B times with per-sample node
        offsets so message-passing respects mini-batch boundaries.

        Relay nodes receive zero features per sample and act only as
        message-passing intermediaries; only agent-node outputs are returned.
        """
        if not self.available:
            return states_t
        B = states_t[0].shape[0]
        n = len(states_t)          # == n_agents
        n_total = self._n_total    # n_agents + n_relay_nodes
        # Stack agent states: [B, n_agents, obs_dim]
        x_agents = torch.stack(states_t, dim=1)
        if self.n_relay_nodes > 0:
            relay_zeros = torch.zeros(
                B, self.n_relay_nodes, self.obs_dim,
                dtype=x_agents.dtype, device=x_agents.device,
            )
            x_full = torch.cat([x_agents, relay_zeros], dim=1)  # [B, n_total, obs_dim]
        else:
            x_full = x_agents
        # [B * n_total, obs_dim]
        x = x_full.reshape(B * n_total, self.obs_dim)
        # Tile edge_index B times, offsetting node ids by n_total per sample
        E = self.edge_index.shape[1]
        offsets = torch.arange(B, device=self.device).repeat_interleave(E) * n_total
        ei_src = self.edge_index[0].repeat(B) + offsets
        ei_dst = self.edge_index[1].repeat(B) + offsets
        batched_ei = torch.stack([ei_src, ei_dst])
        out = self(x, batched_ei)                    # [B * n_total, obs_dim]
        out = out.reshape(B, n_total, self.obs_dim)  # [B, n_total, obs_dim]
        return [out[:, i, :] for i in range(n)]      # only agent slices

    # ── Serialisation (for worker-process IPC) ───────────────────────────────

    def get_weights_cpu(self) -> Dict[str, np.ndarray]:
        """State dict as numpy arrays, excluding non-parameter buffers."""
        return {
            k: v.detach().cpu().numpy()
            for k, v in self.state_dict().items()
            if 'edge_index' not in k
        }


class Agent:
    """Single MADDPG agent with actor/critic networks and target counterparts."""

    def __init__(self, actor_dims: int, critic_dims: int, n_actions: int,
                 n_agents: int, agent_idx: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01,
                 fc1: int = 64, fc2: int = 64, gamma: float = 0.95,
                 tau: float = 0.01, critic_type: str = 'central_critic',
                 network_type: str = 'simple_q_network', use_gnn: bool = False,
                 neighborhood_action_dims: Optional[int] = None):

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.agent_name = f'agent_{agent_idx}'

        os.makedirs(chkpt_dir, exist_ok=True)
        self.best_checkpoint_files = {
            'actor': os.path.join(chkpt_dir, f'{self.agent_name}_actor_best.pth'),
            'critic': os.path.join(chkpt_dir, f'{self.agent_name}_critic_best.pth'),
            'target_actor': os.path.join(chkpt_dir, f'{self.agent_name}_target_actor_best.pth'),
            'target_critic': os.path.join(chkpt_dir, f'{self.agent_name}_target_critic_best.pth'),
        }

        if critic_type == 'central_critic':
            critic_action_dims = n_agents * n_actions
        elif critic_type == 'neighborhood_critic':
            critic_action_dims = (
                neighborhood_action_dims if neighborhood_action_dims is not None else n_actions
            )
        else:
            critic_action_dims = n_actions

        # Online networks
        self.actor = ActorNetwork(
            input_dims=actor_dims, fc1_dims=fc1, fc2_dims=fc2,
            n_actions=n_actions, name=f'{self.agent_name}_actor', chkpt_dir=chkpt_dir
        )
        self.critic = CriticNetwork(
            input_dims=critic_dims, fc1_dims=fc1, fc2_dims=fc2,
            n_agents=n_agents, n_actions=n_actions,
            name=f'{self.agent_name}_critic', chkpt_dir=chkpt_dir,
            action_input_dims=critic_action_dims,
            network_type=network_type
        )

        # Target networks (no gradients needed)
        self.target_actor = ActorNetwork(
            input_dims=actor_dims, fc1_dims=fc1, fc2_dims=fc2,
            n_actions=n_actions, name=f'{self.agent_name}_target_actor', chkpt_dir=chkpt_dir
        )
        self.target_critic = CriticNetwork(
            input_dims=critic_dims, fc1_dims=fc1, fc2_dims=fc2,
            n_agents=n_agents, n_actions=n_actions,
            name=f'{self.agent_name}_target_critic', chkpt_dir=chkpt_dir,
            action_input_dims=critic_action_dims,
            network_type=network_type
        )

        # Optimisers live in Agent, not in the network, for clean separation
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)

        # Hard-copy online weights into targets at init
        self.update_network_parameters(tau=1.0)

        # Note: GNN pre-processing is handled at the MADDPG level, not here.

    @staticmethod
    def _build_executed_action(policy_action: np.ndarray, epsilon: float,
                               decision_block_size: int,
                               deterministic: bool = False) -> np.ndarray:
        block_size = max(1, int(decision_block_size))
        executed_action = np.zeros_like(policy_action, dtype=np.float32)

        for block_start in range(0, len(policy_action), block_size):
            block_end = min(block_start + block_size, len(policy_action))
            block = policy_action[block_start:block_end]

            if deterministic:
                block_choice = 0
            elif np.random.random() < epsilon:
                block_choice = np.random.randint(block_end - block_start)
            else:
                block_choice = int(np.argmax(block))

            executed_action[block_start + block_choice] = 1.0

        return executed_action

    @staticmethod
    def _build_fixed_action_tensor(batch_size: int, n_actions: int,
                                   decision_block_size: int,
                                   device: torch.device) -> torch.Tensor:
        block_size = max(1, int(decision_block_size))
        fixed_actions = torch.zeros((batch_size, n_actions), device=device)
        for block_start in range(0, n_actions, block_size):
            fixed_actions[:, block_start] = 1.0
        return fixed_actions

    def choose_action(self, observation,
                      training: bool = False, epsilon: float = 0.0,
                      decision_block_size: int = 1,
                      deterministic: bool = False):
        """Return policy action, and optionally the executed one-hot action."""
        if isinstance(observation, list):
            observation = np.array(observation)

        state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.actor.device)
        policy_action = self.actor.forward(state).detach().cpu().numpy()[0]

        if not training:
            return policy_action

        executed_action = self._build_executed_action(
            policy_action=policy_action,
            epsilon=epsilon,
            decision_block_size=decision_block_size,
            deterministic=deterministic,
        )
        return policy_action, executed_action

    @torch.no_grad()
    def update_network_parameters(self, tau: Optional[float] = None):
        """Polyak-average online weights into target networks."""
        if tau is None:
            tau = self.tau

        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.lerp_(param.data, tau)

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.lerp_(param.data, tau)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def save_best_models(self):
        """Persist best-so-far snapshot separate from final checkpoint."""
        torch.save(self.actor.state_dict(), self.best_checkpoint_files['actor'])
        torch.save(self.critic.state_dict(), self.best_checkpoint_files['critic'])
        torch.save(self.target_actor.state_dict(), self.best_checkpoint_files['target_actor'])
        torch.save(self.target_critic.state_dict(), self.best_checkpoint_files['target_critic'])

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def load_best_models(self) -> bool:
        """Load best snapshot when available for all model files."""
        if not all(os.path.exists(path) for path in self.best_checkpoint_files.values()):
            return False

        self.actor.load_state_dict(
            torch.load(self.best_checkpoint_files['actor'], weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(self.best_checkpoint_files['critic'], weights_only=True)
        )
        self.target_actor.load_state_dict(
            torch.load(self.best_checkpoint_files['target_actor'], weights_only=True)
        )
        self.target_critic.load_state_dict(
            torch.load(self.best_checkpoint_files['target_critic'], weights_only=True)
        )
        return True


class ReplayBuffer:
    """Centralised experience replay buffer supporting heterogeneous obs dims."""

    def __init__(self, max_size: int, actor_dims: List[int],
                 n_actions: int, n_agents: int,
                 central_state_dims: Optional[int] = None):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents

        # Per-agent state arrays to handle heterogeneous observation spaces
        self.state_memory = [
            np.zeros((max_size, actor_dims[i])) for i in range(n_agents)
        ]
        self.new_state_memory = [
            np.zeros((max_size, actor_dims[i])) for i in range(n_agents)
        ]
        self.action_memory = np.zeros((max_size, n_agents, n_actions))
        self.reward_memory = np.zeros((max_size, n_agents))
        self.terminal_memory = np.zeros((max_size, n_agents), dtype=bool)

        # Optional central state for CC critics (<B, D> vector)
        if central_state_dims is not None:
            self.central_state_memory = np.zeros(
                (max_size, central_state_dims), dtype=np.float32
            )
            self.new_central_state_memory = np.zeros(
                (max_size, central_state_dims), dtype=np.float32
            )
        else:
            self.central_state_memory = None
            self.new_central_state_memory = None

    def store_transition(self, obs, action, reward, obs_, done,
                         central_state=None, new_central_state=None):
        idx = self.mem_cntr % self.mem_size
        for i in range(self.n_agents):
            self.state_memory[i][idx] = obs[i]
            self.new_state_memory[i][idx] = obs_[i]
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        if self.central_state_memory is not None and central_state is not None:
            self.central_state_memory[idx] = central_state
            self.new_central_state_memory[idx] = new_central_state
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = [self.state_memory[i][batch] for i in range(self.n_agents)]
        states_ = [self.new_state_memory[i][batch] for i in range(self.n_agents)]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        central_states = (
            self.central_state_memory[batch]
            if self.central_state_memory is not None else None
        )
        new_central_states = (
            self.new_central_state_memory[batch]
            if self.new_central_state_memory is not None else None
        )
        return states, actions, rewards, states_, terminal, central_states, new_central_states

    def ready(self, batch_size: int) -> bool:
        return self.mem_cntr >= batch_size


class MADDPG:
    """Multi-Agent DDPG orchestrator."""

    def __init__(self, actor_dims: List[int], critic_dims: List[int],
                 n_agents: int, n_actions: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01,
                 fc1: int = 64, fc2: int = 64, gamma: float = 0.95,
                 tau: float = 0.01, critic_type: str = 'central_critic',
                 network_type: str = 'simple_q_network', use_gnn: bool = False,
                 critic_target_mode: str = 'block_argmax_onehot',
                 actor_mode: str = 'soft',
                 adjacency: Optional[List[List[int]]] = None,
                 gnn_adjacency: Optional[List[List[int]]] = None,
                 gnn_n_relay: int = 0,
                 central_state_dims: Optional[int] = None):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.critic_type = critic_type
        self.critic_target_mode = critic_target_mode
        self.actor_mode = actor_mode
        self.adjacency = adjacency
        self.chkpt_dir = chkpt_dir

        actor_dims_list = (
            actor_dims if isinstance(actor_dims, list) else [actor_dims] * n_agents
        )
        critic_dims_list = (
            critic_dims if isinstance(critic_dims, list) else [critic_dims] * n_agents
        )

        self.agents = [
            Agent(
                actor_dims=actor_dims_list[i],
                critic_dims=critic_dims_list[i],
                n_actions=n_actions,
                n_agents=n_agents,
                agent_idx=i,
                chkpt_dir=os.path.join(chkpt_dir, f'agent_{i}'),
                alpha=alpha,
                beta=beta,
                fc1=fc1,
                fc2=fc2,
                gamma=gamma,
                tau=tau,
                critic_type=critic_type,
                network_type=network_type,
                use_gnn=False,  # GNN is handled at MADDPG level
                neighborhood_action_dims=(
                    (1 + len(adjacency[i])) * n_actions
                    if critic_type == 'neighborhood_critic' and adjacency is not None
                    else None
                ),
            )
            for i in range(n_agents)
        ]

        # Shared GNN encoder — operates on all agents' observations jointly
        self.gnn_adjacency = gnn_adjacency
        self.gnn_processor: Optional[GNNProcessor] = None
        self.gnn_optimizer = None
        if use_gnn:
            obs_dim = actor_dims_list[0]  # all agents share the same obs_dim
            self.gnn_processor = GNNProcessor(
                obs_dim=obs_dim, hidden_dim=64,
                n_agents=n_agents, adjacency=gnn_adjacency,
                n_relay_nodes=gnn_n_relay,
            )
            if self.gnn_processor.available:
                self.gnn_optimizer = optim.Adam(
                    self.gnn_processor.parameters(), lr=alpha,
                )

        self.memory = ReplayBuffer(
            max_size=50_000,
            actor_dims=actor_dims_list,
            n_actions=n_actions,
            n_agents=n_agents,
            central_state_dims=central_state_dims,
        )
        # Mixed-precision scaler (no-op on CPU, ~1.5-2x speedup on GPU)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

    @staticmethod
    def _block_argmax_onehot(actions: torch.Tensor, decision_block_size: int) -> torch.Tensor:
        """Project [B, A] soft actions to one-hot per decision block."""
        block_size = max(1, int(decision_block_size))
        onehot = torch.zeros_like(actions)
        total_actions = actions.shape[1]

        for block_start in range(0, total_actions, block_size):
            block_end = min(block_start + block_size, total_actions)
            block = actions[:, block_start:block_end]
            block_indices = torch.argmax(block, dim=1)
            row_indices = torch.arange(actions.shape[0], device=actions.device)
            onehot[row_indices, block_start + block_indices] = 1.0

        return onehot

    def _project_actions(self, actions: torch.Tensor, decision_block_size: int,
                         mode: str, straight_through: bool = False) -> torch.Tensor:
        if mode != 'block_argmax_onehot':
            return actions

        onehot = self._block_argmax_onehot(actions, decision_block_size)
        if straight_through:
            return onehot + (actions - actions.detach())
        return onehot

    def choose_action(self, observations, topology_info=None,
                      training: bool = False, epsilon: float = 0.0,
                      deterministic_mask: Optional[List[bool]] = None,
                      decision_block_size: int = 1):
        if deterministic_mask is None:
            deterministic_mask = [False] * self.n_agents

        # Apply shared GNN encoder to all agents' observations jointly
        if self.gnn_processor is not None:
            observations = self.gnn_processor.process_observations(
                [np.array(o) for o in observations]
            )

        if not training:
            return [
                agent.choose_action(
                    observations[i] if isinstance(observations[0], (list, np.ndarray)) else observations,
                    topology_info,
                )
                for i, agent in enumerate(self.agents)
            ]

        policy_actions, executed_actions = [], []
        for i, agent in enumerate(self.agents):
            policy_action, executed_action = agent.choose_action(
                observations[i] if isinstance(observations[0], (list, np.ndarray)) else observations,
                topology_info,
                training=True,
                epsilon=epsilon,
                decision_block_size=decision_block_size,
                deterministic=deterministic_mask[i],
            )
            policy_actions.append(policy_action)
            executed_actions.append(executed_action)

        return policy_actions, executed_actions

    def learn(self, batch_size: int = 256,
              deterministic_mask: Optional[List[bool]] = None,
              decision_block_size: int = 1):
        """Sample a mini-batch and update all agents."""
        if not self.memory.ready(batch_size):
            return

        if deterministic_mask is None:
            deterministic_mask = [False] * self.n_agents

        states_list, actions, rewards, states_next_list, dones, \
            central_states_np, new_central_states_np = \
            self.memory.sample_buffer(batch_size)

        device = self.agents[0].actor.device
        logger.debug("MADDPG.learn – device: %s, batch: %d", device, batch_size)

        # Convert to tensors
        # states_list / states_next_list: list of [B, obs_dim_i] per agent
        states_t = [torch.tensor(s, dtype=torch.float).to(device) for s in states_list]
        states_next_t = [torch.tensor(s, dtype=torch.float).to(device) for s in states_next_list]
        actions_t = torch.tensor(actions, dtype=torch.float).to(device)   # [B, n_agents, n_actions]
        rewards_t = torch.tensor(rewards, dtype=torch.float).to(device)   # [B, n_agents]
        dones_t = torch.tensor(dones).to(device)                           # [B, n_agents]

        # ---- GNN encoding ----
        # states_t_gnn carries grad_fn back through the GNN for actor updates.
        # states_next_t_gnn is computed under no_grad (target network logic).
        # Critic always receives .detach() tensors to prevent circular dependency.
        if self.gnn_processor is not None:
            if self.gnn_optimizer is not None:
                self.gnn_optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                states_t_gnn = self.gnn_processor.process_batch(states_t)
            with torch.no_grad():
                with autocast(enabled=torch.cuda.is_available()):
                    states_next_t_gnn = self.gnn_processor.process_batch(states_next_t)
        else:
            states_t_gnn = states_t
            states_next_t_gnn = states_next_t

        if self.critic_type == 'central_critic':
            if central_states_np is not None:
                # Use compact <B, D> central state (106+32=138 dims) instead of
                # concatenating all local obs (32×36=1152 dims).
                all_states = torch.tensor(
                    central_states_np, dtype=torch.float
                ).to(device)                                     # [B, central_state_dims]
                all_states_next = torch.tensor(
                    new_central_states_np, dtype=torch.float
                ).to(device)
            else:
                # Fallback: concatenate local states (used if central state not stored)
                all_states = torch.cat(
                    [s.detach() for s in states_t_gnn], dim=1
                )
                all_states_next = torch.cat(states_next_t_gnn, dim=1)
            all_actions_flat = actions_t.view(batch_size, -1)   # [B, n_agents*n_actions]

        for i, agent in enumerate(self.agents):
            if deterministic_mask[i]:
                continue

            agent_reward = rewards_t[:, i]         # [B]
            agent_done = dones_t[:, i]             # [B]

            with autocast(enabled=torch.cuda.is_available()):
                # ---- Critic update ----
                if self.critic_type == 'central_critic':
                    with torch.no_grad():
                        next_action_tensors = []
                        for j, other_agent in enumerate(self.agents):
                            if deterministic_mask[j]:
                                next_action_tensors.append(
                                    Agent._build_fixed_action_tensor(
                                        batch_size=batch_size,
                                        n_actions=self.n_actions,
                                        decision_block_size=decision_block_size,
                                        device=device,
                                    )
                                )
                            else:
                                next_action_tensors.append(
                                    self._project_actions(
                                        other_agent.target_actor(states_next_t_gnn[j]),
                                        decision_block_size=decision_block_size,
                                        mode=self.critic_target_mode,
                                        straight_through=False,
                                    )
                                )
                        next_actions = torch.cat(
                            next_action_tensors,
                            dim=1,
                        )  # [B, n_agents*n_actions]
                        target_q = agent.target_critic(all_states_next, next_actions)
                        target_q[agent_done] = 0.0
                        target_q = agent_reward.unsqueeze(1) + agent.gamma * target_q
    
                    current_q = agent.critic(all_states, all_actions_flat)
                elif self.critic_type == 'neighborhood_critic':
                    nbr_idxs = self.adjacency[i] if self.adjacency else []
                    nc_states = torch.cat(
                        [states_t_gnn[i].detach()] + [states_t_gnn[j].detach() for j in nbr_idxs], dim=1
                    )
                    nc_states_next = torch.cat(
                        [states_next_t_gnn[i]] + [states_next_t_gnn[j] for j in nbr_idxs], dim=1
                    )
                    nc_actions = torch.cat(
                        [actions_t[:, i, :]] + [actions_t[:, j, :] for j in nbr_idxs], dim=1
                    )
                    with torch.no_grad():
                        next_self_a = self._project_actions(
                            agent.target_actor(states_next_t_gnn[i]),
                            decision_block_size=decision_block_size,
                            mode=self.critic_target_mode,
                            straight_through=False,
                        )
                        next_nbr_actions = [next_self_a]
                        for j in nbr_idxs:
                            if deterministic_mask[j]:
                                next_nbr_actions.append(
                                    Agent._build_fixed_action_tensor(
                                        batch_size=batch_size,
                                        n_actions=self.n_actions,
                                        decision_block_size=decision_block_size,
                                        device=device,
                                    )
                                )
                            else:
                                next_nbr_actions.append(
                                    self._project_actions(
                                        self.agents[j].target_actor(states_next_t_gnn[j]),
                                        decision_block_size=decision_block_size,
                                        mode=self.critic_target_mode,
                                        straight_through=False,
                                    )
                                )
                        next_nc_actions = torch.cat(next_nbr_actions, dim=1)
                        target_q = agent.target_critic(nc_states_next, next_nc_actions)
                        target_q[agent_done] = 0.0
                        target_q = agent_reward.unsqueeze(1) + agent.gamma * target_q
                    current_q = agent.critic(nc_states, nc_actions)
                else:  # local critic
                    agent_states = states_t_gnn[i].detach()
                    agent_states_next = states_next_t_gnn[i]
                    agent_actions = actions_t[:, i, :]

                    with torch.no_grad():
                        next_a = self._project_actions(
                            agent.target_actor(agent_states_next),
                            decision_block_size=decision_block_size,
                            mode=self.critic_target_mode,
                            straight_through=False,
                        )
                        target_q = agent.target_critic(agent_states_next, next_a)
                        target_q[agent_done] = 0.0
                        target_q = agent_reward.unsqueeze(1) + agent.gamma * target_q

                    current_q = agent.critic(agent_states, agent_actions)

                critic_loss = F.mse_loss(current_q, target_q)
            agent.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(agent.critic_optimizer)   
            logger.debug("Agent %d critic loss: %.6f", i, critic_loss.item())

            with autocast(enabled=torch.cuda.is_available()):
                # ---- Actor update ----
                predicted_actions_soft = agent.actor(states_t_gnn[i])
                if self.actor_mode == 'st_onehot':
                    predicted_actions = self._project_actions(
                        predicted_actions_soft,
                        decision_block_size=decision_block_size,
                        mode='block_argmax_onehot',
                        straight_through=True,
                    )
                else:
                    predicted_actions = predicted_actions_soft
    
                if self.critic_type == 'central_critic':
                    actions_for_actor = actions_t.clone()
                    actions_for_actor[:, i, :] = predicted_actions
                    actor_loss = -agent.critic(
                        all_states, actions_for_actor.view(batch_size, -1)
                    ).mean()
                elif self.critic_type == 'neighborhood_critic':
                    nbr_idxs = self.adjacency[i] if self.adjacency else []
                    nc_act_states = torch.cat(
                        [states_t_gnn[i].detach()] + [states_t_gnn[j].detach() for j in nbr_idxs], dim=1
                    )
                    nc_act_actions = torch.cat(
                        [predicted_actions] + [actions_t[:, j, :] for j in nbr_idxs], dim=1
                    )
                    actor_loss = -agent.critic(nc_act_states, nc_act_actions).mean()
                else:
                    actor_loss = -agent.critic(states_t_gnn[i].detach(), predicted_actions).mean()
    
            agent.actor_optimizer.zero_grad()
            # retain_graph keeps GNN intermediates alive so all 32 agents can
            # accumulate gradients through the single shared GNN forward pass.
            _retain = (self.gnn_processor is not None) and (i < len(self.agents) - 1)
            self.scaler.scale(actor_loss).backward(retain_graph=_retain)
            self.scaler.step(agent.actor_optimizer)
            logger.debug("Agent %d actor loss:  %.6f", i, actor_loss.item())

            agent.update_network_parameters()

        # Step GNN optimizer after all agents (grads accumulated across all actors)
        if self.gnn_processor is not None and self.gnn_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.gnn_processor.parameters(), 1.0)
            self.scaler.step(self.gnn_optimizer)

        # GradScaler must be updated once per full optimiser cycle, not once per
        # agent — calling it 65x per learn() causes premature scale reduction.
        self.scaler.update()

    def store_transition(self, obs, action, reward, obs_, done,
                         central_state=None, new_central_state=None):
        self.memory.store_transition(obs, action, reward, obs_, done,
                                     central_state, new_central_state)

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()
        if self.gnn_processor is not None:
            os.makedirs(self.chkpt_dir, exist_ok=True)
            torch.save(
                self.gnn_processor.state_dict(),
                os.path.join(self.chkpt_dir, 'gnn_processor.pth'),
            )

    def save_best_checkpoint(self):
        for agent in self.agents:
            agent.save_best_models()
        if self.gnn_processor is not None:
            os.makedirs(self.chkpt_dir, exist_ok=True)
            torch.save(
                self.gnn_processor.state_dict(),
                os.path.join(self.chkpt_dir, 'gnn_processor_best.pth'),
            )

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()
        if self.gnn_processor is not None:
            path = os.path.join(self.chkpt_dir, 'gnn_processor.pth')
            if os.path.exists(path):
                self.gnn_processor.load_state_dict(
                    torch.load(path, weights_only=True)
                )

    def load_best_checkpoint(self) -> bool:
        loaded_all = True
        for agent in self.agents:
            loaded_all = agent.load_best_models() and loaded_all
        if self.gnn_processor is not None:
            path = os.path.join(self.chkpt_dir, 'gnn_processor_best.pth')
            if os.path.exists(path):
                self.gnn_processor.load_state_dict(
                    torch.load(path, weights_only=True)
                )
            else:
                loaded_all = False
        return loaded_all

    def get_actor_weights_cpu(self) -> List[Dict[str, 'np.ndarray']]:
        """Serialise actor state dicts to CPU numpy arrays for worker-process IPC."""
        return [
            {k: v.detach().cpu().numpy() for k, v in agent.actor.state_dict().items()}
            for agent in self.agents
        ]

    def get_gnn_info_cpu(self):
        """Serialise GNN state for worker-process IPC.

        Returns a tuple ``(obs_dim, n_agents, adjacency, weights)`` or ``None``
        when there is no GNN or the GNN is unavailable (torch_geometric missing).
        The adjacency list-of-lists is picklable without any tensor overhead.
        """
        if self.gnn_processor is None or not self.gnn_processor.available:
            return None
        return (
            self.gnn_processor.obs_dim,
            self.gnn_processor.n_agents,
            self.gnn_adjacency,
            self.gnn_processor.n_relay_nodes,
            self.gnn_processor.get_weights_cpu(),
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('Testing Clean MADDPG Implementation')
    print('=' * 40)

    set_global_seeds(42)

    actor_dims = [26] * 5
    n_agents = 5
    n_actions = 3

    configs = [
        {
            'critic_type': 'central_critic',
            'critic_dims': [130] * 5,
            'network_type': 'simple_q_network',
            'use_gnn': False,
        },
        {
            'critic_type': 'central_critic',
            'critic_dims': [130] * 5,
            'network_type': 'duelling_q_network',
            'use_gnn': False,
        },
        {
            'critic_type': 'local_critic',
            'critic_dims': [26] * 5,
            'network_type': 'duelling_q_network',
            'use_gnn': False,
        },
        {
            'critic_type': 'central_critic',
            'critic_dims': [130] * 5,
            'network_type': 'simple_q_network',
            'use_gnn': True,
        },
    ]

    for i, config in enumerate(configs):
        print(f'\nConfiguration {i + 1}: {config}')
        maddpg = MADDPG(
            actor_dims=actor_dims, critic_dims=config['critic_dims'],
            n_agents=n_agents, n_actions=n_actions,
            chkpt_dir=f'test_models/config_{i}', **config
        )
        observations = [np.random.random(26) for _ in range(n_agents)]
        actions = maddpg.choose_action(observations)
        print(f'  Actions shape: {np.array(actions).shape}')
        print(f'  Action sample: {actions[0][:3]}')

    print('\nAll tests passed!')
