"""
Network Environment — Hop-by-Hop MADDPG Routing
All 65 nodes are agents. Each step every node forwards one packet
from its local queue to a chosen neighbour (action = argmax over
neighbour softmax). The path_cache is used only by the OSPF
baseline evaluator; it plays no role during RL training.
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import random
import itertools
from pathlib import Path
import re

# ── Reward hyper-parameters ──────────────────────────────────────────────────
ALPHA = 1.0      # delivery weight
BETA  = 2.0      # congestion penalty (only fires above CONGESTION_THRESHOLD)
GAMMA = 0.4      # utilisation variance penalty (load-balance incentive)
PATH_ALPHA = 0.7 # weight of path-bottleneck penalty vs local bw reward
CONGESTION_THRESHOLD = 0.7  # utilisation below this is not penalised

# ── Packet parameters ─────────────────────────────────────────────────────────
PACKET_SIZE    = 0.03   # normalised bandwidth cost per packet per hop
TTL_INIT       = 15     # max hops before drop (diameter ≈ 6, so 2× headroom)
UTIL_DECAY     = 0.95   # per-step link-utilisation decay
MAX_NODE_QUEUE_DEPTH = 100  # per-node packet buffer cap; overflow → true drop
INIT_PACKETS   = 60     # packets injected at episode reset
INJECT_EVERY   = 10     # inject N_INJECT_BATCH new packets every N steps
N_INJECT_BATCH = 20
DEST_BUCKETS = ("access", "dist", "core")
N_DEST_BUCKETS = len(DEST_BUCKETS)
K_PATHS = 3          # K shortest paths per access-destination for per-packet routing
SHAPING_WEIGHT = 0.4  # potential-based hop-progress reward shaping weight
DEFAULT_FLOW_DURATION_STEPS = 12

REAL_SERVICE_PROVIDER_TOPOLOGY = """BS1 S8 1000 1 0
BS2 S13 1000 1 0
BS3 S22 1000 1 0
BS4 S31 1000 1 0
BS5 S39 1000 1 0
BS6 S48 1000 1 0
BS7 S57 1000 1 0
MECS1 S8 1000 1 0
MECS2 S13 1000 1 0
MECS3 S22 1000 1 0
MECS4 S31 1000 1 0
MECS5 S39 1000 1 0
MECS6 S48 1000 1 0
MECS7 S57 1000 1 0
CS1 S12 1000 1 0
CS2 S21 1000 1 0
CS3 S30 1000 1 0
CS4 S38 1000 1 0
CS5 S47 1000 1 0
CS6 S56 1000 1 0
CS7 S65 1000 1 0
S1 S2 500 1 0
S1 S3 500 1 0
S1 S8 300 1 0
S2 S4 500 1 0
S3 S4 500 1 0
S4 S5 500 1 0
S4 S6 500 1 0
S5 S7 500 1 0
S6 S7 500 1 0
S7 S12 700 1 0
S8 S13 300 1 0
S8 S22 300 1 0
S9 S10 500 1 0
S9 S14 500 1 0
S10 S11 500 1 0
S11 S20 500 1 0
S12 S21 700 1 0
S12 S30 700 1 0
S13 S14 300 1 0
S13 S31 300 1 0
S14 S15 500 1 0
S14 S16 500 1 0
S15 S17 500 1 0
S16 S17 500 1 0
S17 S18 500 1 0
S17 S19 500 1 0
S18 S20 500 1 0
S19 S20 500 1 0
S20 S21 700 1 0
S21 S38 700 1 0
S22 S23 300 1 0
S22 S39 300 1 0
S23 S24 500 1 0
S23 S25 500 1 0
S24 S26 500 1 0
S25 S26 500 1 0
S26 S27 500 1 0
S26 S28 500 1 0
S27 S29 500 1 0
S28 S29 500 1 0
S29 S30 700 1 0
S30 S47 700 1 0
S31 S32 300 1 0
S31 S48 300 1 0
S32 S33 500 1 0
S33 S34 500 1 0
S34 S35 500 1 0
S35 S36 500 1 0
S36 S37 500 1 0
S37 S38 700 1 0
S38 S56 700 1 0
S39 S40 300 1 0
S39 S48 300 1 0
S40 S41 500 1 0
S40 S42 500 1 0
S41 S43 500 1 0
S42 S43 500 1 0
S43 S44 500 1 0
S43 S45 500 1 0
S44 S46 500 1 0
S45 S46 500 1 0
S46 S47 700 1 0
S47 S56 700 1 0
S48 S49 300 1 0
S48 S57 300 1 0
S49 S50 500 1 0
S49 S51 500 1 0
S50 S52 500 1 0
S51 S52 500 1 0
S52 S53 500 1 0
S52 S54 500 1 0
S53 S55 500 1 0
S54 S55 500 1 0
S55 S56 700 1 0
S56 S65 700 1 0
S57 S58 300 1 0
S58 S59 500 1 0
S58 S60 500 1 0
S59 S61 500 1 0
S60 S61 500 1 0
S61 S62 500 1 0
S61 S63 500 1 0
S62 S64 500 1 0
S63 S64 500 1 0
S64 S65 700 1 0"""


def _natural_key(name: str) -> Tuple[str, int]:
    match = re.match(r'^([A-Za-z]+)(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


class NetworkTopology:
    """Hierarchical service-provider topology (65 nodes, 3 tiers)."""

    def __init__(self, topology_type: str = "service_provider",
                 n_nodes: int = 65, seed: int = 42,
                 topology_config: Optional[Dict] = None):
        self.topology_type = topology_type
        self.n_nodes = n_nodes
        self._seed = seed
        self.topology_config = topology_config or {}
        random.seed(seed)
        np.random.seed(seed)

        self.graph = self._build()
        self.hosts = list(self.graph.nodes())

        # Shortest-path cache — used by the Shortest Path baseline ONLY (not by RL agents).
        self.path_cache: Dict[Tuple, List] = {}
        for src, dst in itertools.permutations(self.hosts, 2):
            try:
                self.path_cache[(src, dst)] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self.path_cache[(src, dst)] = []

        # K-shortest-paths cache — used by MADDPG per-destination routing.
        # Only (src, access_dst) pairs: core/dist nodes are not traffic endpoints.
        # For degree-1 nodes (access endpoints with one uplink) all K paths share
        # the same first hop, so dedup at the second hop (index=2) to give agents
        # K genuinely different end-to-end routes.
        self.kpath_cache: Dict[Tuple[str, str], List[List[str]]] = {}
        for src in self.hosts:
            _src_degree = self.graph.degree(src)
            _dedup_idx = 2 if _src_degree == 1 else 1
            for dst in self.access_nodes:
                if src != dst:
                    self.kpath_cache[(src, dst)] = self._build_distinct_kpaths(
                        self.graph, src, dst, K_PATHS, dedup_hop_idx=_dedup_idx
                    )

        # Flow-mode fixed-duration reservations (disabled unless configured).
        self._flow_hold_steps = 0
        self._flow_reservation_buckets: Dict[Tuple[str, str], deque] = {}
        self._flow_reserved: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _build_distinct_kpaths(graph, src: str, dst: str, k: int,
                               dedup_hop_idx: int = 1) -> List[List[str]]:
        """Return up to k paths with distinct hops at dedup_hop_idx, shortest first.

        dedup_hop_idx=1 (default): deduplicate by first hop — best for nodes with
          degree ≥ 2 where paths fan out immediately.
        dedup_hop_idx=2: deduplicate by second hop — use for degree-1 access nodes
          whose single uplink means all paths share the same first hop but diverge
          at the adjacent switch, giving K genuinely different end-to-end routes.

        Oversamples by 4× to find alternatives; pads with the shortest path if
        fewer than k distinct hops exist at the chosen dedup position.
        """
        try:
            seen_fh: set = set()
            distinct: List[List[str]] = []
            for p in itertools.islice(
                nx.shortest_simple_paths(graph, src, dst), k * 4
            ):
                fh = p[dedup_hop_idx] if len(p) > dedup_hop_idx else None
                if fh is not None and fh not in seen_fh:
                    seen_fh.add(fh)
                    distinct.append(p)
                    if len(distinct) == k:
                        break
            while distinct and len(distinct) < k:
                distinct.append(distinct[0])
            return distinct
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ── topology builders ────────────────────────────────────────────────────

    def _build(self) -> nx.Graph:
        builders = {
            "service_provider": self._sp_topology,
            "service_provider_real": self._real_service_provider_topology,
            "grid":             self._grid_topology,
            "random":           self._random_topology,
        }
        if self.topology_type not in builders:
            raise ValueError(f"Unknown topology: {self.topology_type}")
        return builders[self.topology_type]()

    def _sp_topology(self) -> nx.Graph:
        G = nx.Graph()
        nodes = [f"H{i}" for i in range(self.n_nodes)]
        for n in nodes:
            G.add_node(n)

        core_size = max(3, int(0.10 * self.n_nodes))   #  6 core nodes
        dist_size = int(0.30 * self.n_nodes)            # 19 distribution nodes
        # remaining 40 = access nodes

        core_nodes   = nodes[:core_size]
        dist_nodes   = nodes[core_size:core_size + dist_size]
        access_nodes = nodes[core_size + dist_size:]

        # Core: full mesh — high capacity
        for i, j in itertools.combinations(core_nodes, 2):
            G.add_edge(i, j, capacity=10.0, utilization=0.0)

        # Distribution: connect to 1–2 core nodes
        for d in dist_nodes:
            G.add_edge(d, random.choice(core_nodes), capacity=8.0, utilization=0.0)
            if random.random() < 0.3:
                others = [n for n in dist_nodes if n != d]
                if others:
                    peer = random.choice(others)
                    if not G.has_edge(d, peer):
                        G.add_edge(d, peer, capacity=5.0, utilization=0.0)

        # Access: connect to 1 distribution node; 10% also direct to core
        for a in access_nodes:
            G.add_edge(a, random.choice(dist_nodes), capacity=2.0, utilization=0.0)
            if random.random() < 0.10:
                c = random.choice(core_nodes)
                if not G.has_edge(a, c):
                    G.add_edge(a, c, capacity=1.0, utilization=0.0)

        self.core_nodes   = core_nodes
        self.dist_nodes   = dist_nodes
        self.access_nodes = access_nodes
        # Compute topology max degree — used to size the state/action vectors.
        self.max_degree = max(d for _, d in G.degree())
        return G

    def _real_service_provider_topology(self) -> nx.Graph:
        G = nx.Graph()
        topology_text = self._load_real_topology_text()
        switch_nodes = set()
        endpoint_nodes = set()
        edge_rows: List[Tuple[str, str, float]] = []

        for raw_line in topology_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            left, right = parts[0], parts[1]
            capacity = float(parts[2])
            edge_rows.append((left, right, capacity))
            for node in (left, right):
                if node.startswith('S'):
                    switch_nodes.add(node)
                else:
                    endpoint_nodes.add(node)

        ordered_switches = sorted(switch_nodes, key=_natural_key)
        ordered_endpoints = sorted(endpoint_nodes, key=_natural_key)
        for node in ordered_switches + ordered_endpoints:
            G.add_node(node)

        max_capacity = max((cap for _, _, cap in edge_rows), default=1.0)
        for left, right, raw_capacity in edge_rows:
            G.add_edge(
                left,
                right,
                capacity=raw_capacity,
                capacity_scale=max(raw_capacity / max_capacity, 1e-6),
                utilization=0.0,
            )

        access_nodes = ordered_endpoints
        dist_nodes = sorted(
            {nb for access in access_nodes for nb in G.neighbors(access) if nb in switch_nodes},
            key=_natural_key,
        )
        dist_set = set(dist_nodes)
        core_nodes = [node for node in ordered_switches if node not in dist_set]

        self.core_nodes = core_nodes
        self.dist_nodes = dist_nodes
        self.access_nodes = access_nodes
        self.max_degree = max(d for _, d in G.degree()) if G.number_of_nodes() else 0
        return G

    def _load_real_topology_text(self) -> str:
        candidate = self.topology_config.get('file') or self.topology_config.get('path')
        if candidate:
            path = Path(candidate)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path.read_text(encoding='utf-8')
        return REAL_SERVICE_PROVIDER_TOPOLOGY

    def _grid_topology(self) -> nx.Graph:
        side = int(np.sqrt(self.n_nodes))
        G = nx.grid_2d_graph(side, side)
        mapping = {n: f"H{i}" for i, n in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            G[u][v].update(capacity=1.0, utilization=0.0)
        self.core_nodes = self.dist_nodes = []
        self.access_nodes = list(G.nodes())
        self.max_degree = max(d for _, d in G.degree())
        return G

    def _random_topology(self) -> nx.Graph:
        G = nx.erdos_renyi_graph(self.n_nodes, 0.1, seed=self._seed)
        mapping = {n: f"H{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            G[u][v].update(capacity=random.uniform(1.0, 5.0), utilization=0.0)
        self.core_nodes = self.dist_nodes = []
        self.access_nodes = list(G.nodes())
        self.max_degree = max(d for _, d in G.degree())
        return G

    # ── path-cache helpers ───────────────────────────────────────────────────

    def refresh_path_cache(self):
        """Recompute shortest-path cache after topology changes (e.g. link failures).
        Called by NetworkEngine._inject_failures so the Shortest Path baseline re-routes
        correctly rather than following stale pre-failure paths.
        """
        self.path_cache.clear()
        for src, dst in itertools.permutations(self.hosts, 2):
            try:
                self.path_cache[(src, dst)] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self.path_cache[(src, dst)] = []
        self.kpath_cache.clear()
        for src in self.hosts:
            for dst in self.access_nodes:
                if src != dst:
                    self.kpath_cache[(src, dst)] = self._build_distinct_kpaths(
                        self.graph, src, dst, K_PATHS
                    )

    # ── edge helpers ─────────────────────────────────────────────────────────

    def get_neighbors(self, node: str) -> List[str]:
        return list(self.graph.neighbors(node))

    def get_util(self, u: str, v: str) -> float:
        return self.graph[u][v]['utilization'] if self.graph.has_edge(u, v) else 0.0

    def get_capacity(self, u: str, v: str) -> float:
        return self.graph[u][v].get('capacity', 1.0) if self.graph.has_edge(u, v) else 1.0

    def get_capacity_scale(self, u: str, v: str) -> float:
        return self.graph[u][v].get('capacity_scale', 1.0) if self.graph.has_edge(u, v) else 1.0

    def set_util(self, u: str, v: str, val: float):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['utilization'] = float(np.clip(val, 0.0, 1.0))

    def avail_bw(self, u: str, v: str) -> float:
        return max(0.0, 1.0 - self.get_util(u, v)) if self.graph.has_edge(u, v) else 0.0

    def reset_utils(self):
        for u, v in self.graph.edges():
            self.graph[u][v]['utilization'] = 0.0

    def decay_utils(self):
        for u, v in self.graph.edges():
            self.set_util(u, v, self.get_util(u, v) * UTIL_DECAY)

    @staticmethod
    def _edge_key(u: str, v: str) -> Tuple[str, str]:
        return (u, v) if u <= v else (v, u)

    def configure_flow_reservations(self, hold_steps: int):
        self._flow_hold_steps = max(1, int(hold_steps))
        self._flow_reservation_buckets.clear()
        self._flow_reserved.clear()
        for u, v in self.graph.edges():
            key = self._edge_key(u, v)
            self._flow_reservation_buckets[key] = deque([0.0] * self._flow_hold_steps)
            self._flow_reserved[key] = 0.0

    def clear_flow_reservations(self):
        for key, buckets in self._flow_reservation_buckets.items():
            self._flow_reservation_buckets[key] = deque([0.0] * len(buckets))
            self._flow_reserved[key] = 0.0

    def add_flow_reservation(self, u: str, v: str, amount: float):
        if self._flow_hold_steps <= 0 or not self.graph.has_edge(u, v):
            return
        key = self._edge_key(u, v)
        if key not in self._flow_reservation_buckets:
            return
        val = max(0.0, float(amount))
        self._flow_reserved[key] = self._flow_reserved.get(key, 0.0) + val
        self._flow_reservation_buckets[key][-1] += val

    def get_flow_reserved(self, u: str, v: str) -> float:
        key = self._edge_key(u, v)
        return self._flow_reserved.get(key, 0.0)

    def decay_flow_reservations(self):
        if self._flow_hold_steps <= 0:
            return
        for key, buckets in self._flow_reservation_buckets.items():
            expired = buckets.popleft()
            self._flow_reserved[key] = max(0.0, self._flow_reserved.get(key, 0.0) - expired)
            buckets.append(0.0)

    def sync_utils_from_flow_reservations(self):
        if self._flow_hold_steps <= 0:
            return
        for u, v in self.graph.edges():
            self.set_util(u, v, self.get_flow_reserved(u, v))


class NetworkEngine:
    """
    Hop-by-hop routing engine.

    At every timestep each of the 65 agents receives the queue of packets
    sitting at its node and forwards them through the chosen next-hop
    (argmax of the action vector over its neighbours).
    """

    def __init__(self, topology_type: str = "service_provider",
                 n_nodes: int = 65, seed: int = 42,
                 reward_config: Optional[Dict[str, float]] = None,
                 topology_config: Optional[Dict] = None,
                 traffic_config: Optional[Dict] = None):
        self.topology_type = topology_type
        self.n_nodes = n_nodes
        self.topo_seed = seed
        self.topology = NetworkTopology(
            topology_type,
            n_nodes,
            seed,
            topology_config=topology_config,
        )
        self.n_nodes = len(self.topology.hosts)
        self.time_step = 0
        self.offered_load_factor = 1.0
        self.packet_queue: Dict[str, List[Dict]] = defaultdict(list)
        self._episode_stats = self._blank_stats()
        traffic_cfg = traffic_config or {}
        self.traffic_mode = str(traffic_cfg.get('mode', 'packet')).lower()
        if self.traffic_mode not in ('packet', 'flow'):
            self.traffic_mode = 'packet'
        self.packet_size = float(traffic_cfg.get('packet_size', PACKET_SIZE))
        self.flow_packet_size = float(traffic_cfg.get('flow_packet_size', self.packet_size))
        self.flow_hold_steps = int(max(1, traffic_cfg.get('flow_hold_steps', 12)))
        self.flow_duration_steps = int(max(1, traffic_cfg.get('flow_duration_steps', DEFAULT_FLOW_DURATION_STEPS)))
        self.flow_count_base = int(max(1, traffic_cfg.get('flow_count', len(self.topology.access_nodes))))
        self.episode_steps = int(max(1, traffic_cfg.get('episode_steps', 128)))
        self._flow_schedule: Dict[int, List[Dict]] = defaultdict(list)
        self._active_flows: List[Dict] = []
        self._flow_seq = 0
        if self.traffic_mode == 'flow':
            self.topology.configure_flow_reservations(self.flow_hold_steps)
        base_reward = {
            'delivery_weight': ALPHA,
            'max_util_penalty': BETA,
            'var_util_penalty': GAMMA,
            'drop_penalty': 0.6,
            'backlog_penalty': 0.05,
            'congestion_threshold': CONGESTION_THRESHOLD,
            'progress_shaping_weight': SHAPING_WEIGHT,
        }
        self.reward_cfg = {**base_reward, **(reward_config or {})}

        # Expose topology max degree so calling code can read actor_dims once.
        self.max_neighbors: int = self.topology.max_degree
        self.n_destinations: int = len(self.topology.access_nodes)
        self.n_actions: int = K_PATHS * self.n_destinations  # per-destination action matrix (n_dest × K_PATHS)
        # Per-destination index: maps access node name → row index in the action matrix
        self._access_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.topology.access_nodes)}

        # Node-tier lookup — used by Paper-2 attack surface analysis
        self._tier: Dict[str, str] = {}
        for n in self.topology.core_nodes:   self._tier[n] = 'core'
        for n in self.topology.dist_nodes:   self._tier[n] = 'dist'
        for n in self.topology.access_nodes: self._tier[n] = 'access'

        # ── Performance caches ───────────────────────────────────────────────
        # Cached global mean link utilisation — updated once per step() call
        # and read by every get_state() call, avoiding 86x redundant edge sweeps.
        self._step_mean_util: float = 0.0

        # Preallocated zero-state returned for non-observed hosts so get_state()
        # is only called for the trainable subset when obs_indices is supplied.
        self._empty_state: np.ndarray = np.zeros(self.state_dims, dtype=np.float32)

        # Access nodes are now PE (ingress) agents: they inject and forward their
        # own traffic.  Delivery is counted at the last transit switch when
        # chosen == pkt['dst'], so access nodes never receive transit packets in
        # their queues — no sink exclusion is needed.
        self._sink_host_indices: set = set()
        # Fast membership test for host endpoints (BS*, MECS*, CS*).  Used in
        # step() to skip k-path selection at the host level and defer it to the
        # dist switch (correct MPLS PE model).
        self._endpoint_host_set: set = set(self.topology.access_nodes)
        # Kept for API compatibility; unused for injection (access_nodes used directly).
        self._non_sink_hosts: list = list(self.topology.hosts)

    # ── public API ───────────────────────────────────────────────────────────

    def get_all_hosts(self) -> List[str]:
        return self.topology.hosts

    def get_trainable_hosts(self, filter_mode: str = 'all') -> List[str]:
        """Return the subset of hosts that should be active RL agents.

        filter_mode:
          'all'           – all topology nodes (default, legacy behaviour)
          'switches_only' – only switch nodes (S-prefixed)
          'degree_ge_3'   – only switches with degree ≥ 3 (true branching points)
          'access_nodes'  – only edge/PE endpoints (BS*, MECS*, CS*); PE-router model
                            where only ingress nodes make path decisions
        """
        all_hosts = self.topology.hosts
        if filter_mode == 'all':
            return all_hosts
        switch_hosts = [h for h in all_hosts if h.startswith('S')]
        if filter_mode == 'switches_only':
            return switch_hosts
        if filter_mode == 'degree_ge_3':
            return [h for h in switch_hosts
                    if len(self.topology.get_neighbors(h)) >= 3]
        if filter_mode == 'access_nodes':
            # PE-router model: only traffic ingress/egress endpoints act as agents.
            # Transit (core/dist) switches forward packets without RL decisions.
            return list(self.topology.access_nodes)
        if filter_mode == 'dist_nodes':
            # PE-switch model: only the dist switches (ingress PE routers) act as
            # agents — they make all k-path decisions on behalf of the single-homed
            # endpoints attached to them.  Correct architectural mapping for
            # source-based k-path routing in the SP topology.
            return list(self.topology.dist_nodes)
        raise ValueError(f"Unknown trainable_host_filter: {filter_mode!r}")

    def get_number_neighbors(self, host: str) -> int:
        return len(self.topology.get_neighbors(host))

    def get_tier(self, host: str) -> str:
        """Return 'core' | 'dist' | 'access' — used by attack-surface analysis."""
        return self._tier.get(host, 'access')

    def get_adjacency_indices(self) -> List[List[int]]:
        """Return per-agent neighbor indices into the get_all_hosts() ordering."""
        hosts = self.topology.hosts
        host_idx = {h: idx for idx, h in enumerate(hosts)}
        return [
            [host_idx[nb] for nb in self.topology.get_neighbors(host) if nb in host_idx]
            for host in hosts
        ]

    def reset(self):
        self.topology.reset_utils()
        if self.traffic_mode == 'flow':
            self.topology.clear_flow_reservations()
        self.packet_queue = defaultdict(list)
        self.time_step = 0
        self.offered_load_factor = 1.0
        self._step_mean_util = 0.0
        self._episode_stats = self._blank_stats()
        self._flow_schedule = defaultdict(list)
        self._active_flows = []
        self._flow_seq = 0
        if self.traffic_mode == 'flow':
            self._build_flow_schedule()
        else:
            self._inject_packets(int(max(1, round(INIT_PACKETS * self.offered_load_factor))))

    def reset_with_load(self, offered_load_factor: float = 1.0):
        """Reset environment state with a configurable offered-load multiplier."""
        self.topology.reset_utils()
        if self.traffic_mode == 'flow':
            self.topology.clear_flow_reservations()
        self.packet_queue = defaultdict(list)
        self.time_step = 0
        self.offered_load_factor = float(max(0.1, offered_load_factor))
        self._step_mean_util = 0.0
        self._episode_stats = self._blank_stats()
        self._flow_schedule = defaultdict(list)
        self._active_flows = []
        self._flow_seq = 0
        if self.traffic_mode == 'flow':
            self._build_flow_schedule()
        else:
            self._inject_packets(int(max(1, round(INIT_PACKETS * self.offered_load_factor))))

    @property
    def state_dims(self) -> int:
        """Total state-vector size: max_neighbors + destination slots + fixed slots.

        Fixed slots:
          queue_depth, dest_diversity, timestep,
          mean_adj_util, max_adj_util, var_adj_util,
          per-destination K-path first-hop utils (n_destinations × K_PATHS),
          mean hops remaining (1)
        Removed from previous version: bucket fractions (3) and global mean util (1).
        """
        return self.max_neighbors + self.n_destinations + 7 - K_PATHS + K_PATHS * self.n_destinations

    def get_state(self, host: str) -> np.ndarray:
        """
        Observation for one agent.

        Slot layout:
          [0 : MAX_NEIGHBORS]           available bandwidth to each neighbour
                                        (zero-padded when node has fewer neighbours)
          [MAX_NEIGHBORS]               normalised queue depth
          [MAX_NEIGHBORS+1]             destination diversity in queue
          [MAX_NEIGHBORS+2]             normalised timestep
          [MAX_NEIGHBORS+3 :
           MAX_NEIGHBORS+3+N_DEST]      one-hot presence flag per endpoint destination
          [.. + 0]                      mean adjacent link utilisation
          [.. + 1]                      max adjacent link utilisation
          [.. + 2]                      variance of adjacent link utilisations
          [.. + 3 : .. + 3+K_PATHS]    max-link utilisation per k-path index
                                        (action-aligned: slot k corresponds to action k;
                                         bottleneck util over all hops, not just first hop)
          [.. + 3+K_PATHS]             mean hops remaining in queue (normalised [0,1])
                                        guides anti-loop learning

          Total = max_neighbors + n_destinations + 7 + K_PATHS
        """
        mn   = self.max_neighbors
        nbrs = self.topology.get_neighbors(host)
        state = []

        # MAX_NEIGHBORS bandwidth slots (zero-padded)
        for i in range(mn):
            state.append(self.topology.avail_bw(host, nbrs[i]) if i < len(nbrs) else 0.0)

        # queue summary — 3 slots
        q = self.packet_queue[host]
        dsts_in_queue = {p['dst'] for p in q}
        state.append(min(len(q), 20) / 20.0)
        state.append(len(dsts_in_queue) / max(1, len(self.topology.hosts)))
        state.append(self.time_step / 256.0)

        # destination indicator — N_DEST slots (access nodes only; traffic endpoints in SP network)
        for h in self.topology.access_nodes:
            state.append(1.0 if h in dsts_in_queue else 0.0)

        # link statistics — 3 slots (removed: bucket fractions, global mean util)
        utils = [self.topology.get_util(host, nb) for nb in nbrs]
        if utils:
            state.append(float(np.mean(utils)))
            state.append(float(np.max(utils)))
            state.append(float(np.var(utils)) if len(utils) > 1 else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0])

        # Per-destination K-path bottleneck utilisation — n_destinations * K_PATHS slots
        # Layout: for each destination d, K_PATHS consecutive utils [util_k0, util_k1, util_k2]
        # Each slot is the MAX utilisation across all hops of path k (bottleneck signal),
        # giving the actor a direct gradient to prefer the least-loaded path.
        for dst in self.topology.access_nodes:
            paths = self.topology.kpath_cache.get((host, dst), [])
            for k in range(K_PATHS):
                if k < len(paths) and len(paths[k]) >= 2:
                    path = paths[k]
                    util = max(
                        self.topology.get_util(path[j], path[j + 1])
                        for j in range(len(path) - 1)
                    )
                    state.append(util)
                else:
                    state.append(0.0)

        # mean hops remaining in queue — 1 slot (anti-loop signal)
        _q = self.packet_queue[host]
        _mean_hops = float(np.mean([p.get('hops', 0) for p in _q])) if _q else 0.0
        state.append(max(0.0, 1.0 - _mean_hops / TTL_INIT))

        # guarantee exactly state_dims elements
        total = self.state_dims
        state = state[:total]
        while len(state) < total:
            state.append(0.0)
        return np.array(state, dtype=np.float32)

    def get_central_state(self, trainable_hosts: List[str]) -> np.ndarray:
        """Global state vector <B, D> for centralised critics.

        B ∈ [0,1]^|E|  — normalised available bandwidth on every graph edge.
        D ∈ [0,1]^|A|  — modal destination index (normalised) per trainable agent.

        Total dims = |E| + len(trainable_hosts) = 106 + 32 = 138 for the
        service_provider_real topology with degree_ge_3 filter.
        """
        # B: available bandwidth on all edges (ordered by graph.edges())
        b = np.array(
            [self.topology.avail_bw(u, v) for u, v in self.topology.graph.edges()],
            dtype=np.float32,
        )

        # D: modal destination per trainable agent (which destination is most
        # common in each agent's queue), normalised by n_destinations - 1.
        access_to_idx: Dict[str, int] = {
            a: i for i, a in enumerate(self.topology.access_nodes)
        }
        n_dest = len(self.topology.access_nodes)
        d: List[float] = []
        for host in trainable_hosts:
            q = self.packet_queue[host]
            if q:
                counts: Dict[str, int] = {}
                for pkt in q:
                    dst = pkt['dst']
                    counts[dst] = counts.get(dst, 0) + 1
                modal = max(counts, key=counts.get)
                idx = access_to_idx.get(modal, 0)
            else:
                idx = 0
            d.append(float(idx) / max(1, n_dest - 1))

        return np.concatenate([b, np.array(d, dtype=np.float32)])

    def step(self, actions: List[np.ndarray],
             obs_indices=None) -> Tuple[List[np.ndarray], List[float], Dict]:
        """Execute one hop for every agent simultaneously.

        obs_indices: optional list of topology host indices for which
        get_state() should be called.  Non-listed indices receive a
        pre-allocated zero vector, avoiding wasted state computation for
        nodes that are not MADDPG agents (e.g. degree-1 endpoints and
        degree-2 pass-through switches when filter_mode='degree_ge_3').
        """
        self.time_step += 1
        if self.traffic_mode == 'flow':
            self._inject_scheduled_flow_packets()
        hosts = self.topology.hosts
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        step_sent = step_delivered = step_dropped = 0
        rewards = []
        _agent_drop_stats = []  # (pkt_drop_count, total) per active agent; None if inactive
        _agent_active = []  # True for agents that processed packets this step
        _agent_bottleneck: List[Optional[float]] = []  # mean path-bottleneck util; None if inactive
        reward_components = {
            'fwd_success_term': 0.0,
            'shaping_term': 0.0,
            'chosen_util_term': 0.0,
            'extreme_cong_term': 0.0,
            'var_util_term': 0.0,
        }

        for i, host in enumerate(hosts):
            # Endpoint (sink) nodes only receive traffic; they never forward
            # packets. Skip them early to avoid neighbor/queue lookups.
            if i in self._sink_host_indices:
                _agent_drop_stats.append(None)
                _agent_bottleneck.append(None)
                _agent_active.append(False)
                continue

            action = actions[i] if i < len(actions) else np.array([1.0, 0.0, 0.0])
            nbrs   = self.topology.get_neighbors(host)
            q      = self.packet_queue[host]

            if not nbrs or not q:
                _agent_drop_stats.append(None)
                _agent_bottleneck.append(None)
                _agent_active.append(False)
                continue

            # Per-destination MPLS path selection: action encodes an (n_dest × K_PATHS)
            # routing matrix.  Each destination independently selects its path class
            # (k=0 → shortest, k=1/2 → alternatives) via argmax on its row.
            _n_dest = len(self.topology.access_nodes)
            _expected = K_PATHS * _n_dest
            if len(action) >= _expected:
                _action_matrix = action[:_expected].reshape(_n_dest, K_PATHS)
            else:
                # Fallback for undersized actions (non-trainable transit nodes get zeros)
                _pi = int(np.argmax(action[:K_PATHS])) if len(action) >= K_PATHS else 0
                _action_matrix = np.zeros((_n_dest, K_PATHS), dtype=np.float32)
                _action_matrix[:, _pi] = 1.0

            # Compute mean bottleneck util for chosen paths (pre-action network state).
            # Only considers destinations with packets currently in queue.
            _active_dsts = {pkt['dst'] for pkt in q}
            _bn_list: List[float] = []
            for _d_idx, _dst in enumerate(self.topology.access_nodes):
                if _dst not in _active_dsts:
                    continue
                _ck = int(np.argmax(_action_matrix[_d_idx]))
                _bpaths = self.topology.kpath_cache.get((host, _dst), [])
                if _ck < len(_bpaths) and len(_bpaths[_ck]) >= 2:
                    _bpath = _bpaths[_ck]
                    _bn_list.append(max(
                        self.topology.get_util(_bpath[j], _bpath[j + 1])
                        for j in range(len(_bpath) - 1)
                    ))
            _agent_bottleneck_val = float(np.mean(_bn_list)) if _bn_list else 0.0

            # forward packets
            delivered = dropped = 0
            pkt_delivered_count = 0   # packets delivered by this agent this step
            pkt_drop_count = 0        # packets dropped by this agent this step
            for pkt in q:
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
                step_sent += 1
                dst     = pkt['dst']
                path_k_pkt = pkt.get('path_k')
                pkt_src    = pkt.get('src', host)

                if path_k_pkt is None:
                    if host in self._endpoint_host_set:
                        # Host endpoint (BS*, MECS*, CS*): single uplink to dist
                        # switch.  Forward without selecting path_k so the PE
                        # switch makes the k-path decision (MPLS PE model).
                        chosen = nbrs[0] if nbrs else None
                    else:
                        # Ingress PE switch: select k-path on behalf of the
                        # source endpoint.  kpath_cache is keyed from pkt_src
                        # (the originating endpoint), not from this switch.
                        dst_idx = self._access_to_idx.get(dst, 0)
                        path_k_pkt = int(np.argmax(_action_matrix[dst_idx]))
                        pkt['path_k'] = path_k_pkt
                        paths  = self.topology.kpath_cache.get((pkt_src, dst), [])
                        chosen = self._next_hop_for_transit(paths, path_k_pkt, host, nbrs)
                else:
                    # Transit: follow the established globally-consistent path.
                    paths  = self.topology.kpath_cache.get((pkt_src, dst), [])
                    chosen = self._next_hop_for_transit(paths, path_k_pkt, host, nbrs)

                if chosen is None:
                    # Last-resort fallback — should only occur after link failures.
                    _sp = self.topology.path_cache.get((host, dst), [])
                    chosen = _sp[1] if len(_sp) >= 2 and _sp[1] in nbrs else nbrs[0]

                required_bw = self._required_bw(host, chosen)
                if self.topology.avail_bw(host, chosen) < required_bw:
                    self._episode_stats['capacity_block_events'] += 1
                    if self.traffic_mode == 'flow':
                        dropped += 1
                        step_dropped += 1
                        pkt_drop_count += 1
                        self._episode_stats['drop_link_congestion'] += 1
                        self._episode_stats['dropped_delay_samples'].append(
                            max(0, self.time_step - pkt_created)
                        )
                        self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    else:
                        # Link congested: buffer packet at current node rather than drop.
                        # TTL and hop count are unchanged — waiting is not a hop.
                        # Only drop on genuine buffer overflow (MAX_NODE_QUEUE_DEPTH).
                        step_sent -= 1  # this was not a forwarding event
                        if len(next_queue[host]) < MAX_NODE_QUEUE_DEPTH:
                            next_queue[host].append(pkt)
                        else:
                            # Buffer overflow: genuine drop
                            dropped += 1
                            step_dropped += 1
                            pkt_drop_count += 1
                            self._episode_stats['drop_overflow'] += 1
                            self._episode_stats['dropped_delay_samples'].append(
                                max(0, self.time_step - pkt_created)
                            )
                            self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    continue
                cur = self.topology.get_util(host, chosen)
                cap_scale = self.topology.get_capacity_scale(host, chosen)
                self._register_link_load(host, chosen, required_bw, cur, cap_scale)

                if chosen == pkt['dst']:
                    delivered += 1
                    step_delivered += 1
                    pkt_delivered_count += 1
                    final_hops = pkt_hops + 1
                    self._episode_stats['delivered_delay_samples'].append(
                        max(0, self.time_step - pkt_created)
                    )
                    self._episode_stats['delivered_hop_samples'].append(final_hops)
                elif pkt['ttl'] > 0:
                    next_queue[chosen].append({
                        'dst': pkt['dst'],
                        'src': pkt.get('src', host),   # preserve injection source
                        'path_k': pkt.get('path_k'),   # preserve MPLS-like path label
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                        'ingress': host,
                    })
                else:
                    dropped += 1
                    step_dropped += 1   # TTL expired
                    pkt_drop_count += 1
                    self._episode_stats['drop_ttl'] += 1
                    self._episode_stats['dropped_delay_samples'].append(
                        max(0, self.time_step - pkt_created)
                    )
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops + 1)

            # collect per-agent stats for deferred global reward computation
            _agent_drop_stats.append((pkt_drop_count, len(q)))
            _agent_bottleneck.append(_agent_bottleneck_val)
            _agent_active.append(True)

        # ---------------------------------------------------------------
        # Per-agent reward: path-bottleneck + local drop penalty + variance
        # ---------------------------------------------------------------
        # r = (1-path_alpha) * f(min_adj_bw) - path_alpha * bottleneck_util
        #     - drop_penalty * drop_rate - var_util_penalty * network_var
        #
        # path_alpha (default PATH_ALPHA=0.7) makes the bottleneck term dominant
        # so agents learn to prefer least-loaded paths, not just uncongested
        # local links.  The variance penalty is a global load-balance incentive.
        drop_penalty    = float(self.reward_cfg.get('drop_penalty',    BETA))
        path_alpha      = float(self.reward_cfg.get('path_alpha',      PATH_ALPHA))
        var_util_pen_w  = float(self.reward_cfg.get('var_util_penalty', GAMMA))

        # Global variance penalty — computed once, applied to all active agents.
        _util_vals_pre = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        var_pen = var_util_pen_w * float(np.var(_util_vals_pre)) if len(_util_vals_pre) > 1 else 0.0

        for host, stats, bn in zip(hosts, _agent_drop_stats, _agent_bottleneck):
            if stats is None:
                rewards.append(0.0)
            else:
                drop_count, total = stats
                # f(min available BW across adjacent links) — local congestion signal
                adj_bws = [self.topology.avail_bw(host, nb)
                           for nb in self.topology.get_neighbors(host)]
                min_bw = float(min(adj_bws)) if adj_bws else 0.0
                if min_bw >= 0.4:
                    bw_reward = 1.0
                elif min_bw >= 0.2:
                    bw_reward = 0.2
                elif min_bw >= 0.05:
                    bw_reward = -1.0
                else:
                    bw_reward = -5.0
                drop_term = drop_penalty * drop_count / max(1, total)
                bn_val = bn if bn is not None else 0.0
                r = (1.0 - path_alpha) * bw_reward - path_alpha * bn_val - drop_term - var_pen
                rewards.append(r)
                reward_components['fwd_success_term'] += bw_reward
                reward_components['chosen_util_term'] += bn_val
                reward_components['var_util_term']    += var_pen

        # bookkeeping
        self._advance_link_loads_one_step()
        self.packet_queue = next_queue
        if self.traffic_mode != 'flow' and self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))

        self._episode_stats['packets_sent']      += step_sent
        self._episode_stats['packets_delivered'] += step_delivered
        self._episode_stats['packets_dropped']   += step_dropped
        self._episode_stats['total_reward']      += sum(rewards)
        self._episode_stats['steps'] += 1
        for k, v in reward_components.items():
            self._episode_stats['reward_component_sums'][k] += v

        current_backlog = sum(len(q) for q in self.packet_queue.values())
        self._episode_stats['backlog_sum'] += current_backlog
        self._episode_stats['backlog_peak'] = max(
            self._episode_stats['backlog_peak'], current_backlog
        )
        queue_sizes = [len(q) for q in self.packet_queue.values()]
        self._episode_stats['step_max_node_queue'].append(max(queue_sizes) if queue_sizes else 0)
        self._episode_stats['step_active_queues'].append(sum(1 for size in queue_sizes if size > 0))

        util_values = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        mean_util = float(np.mean(util_values)) if util_values else 0.0
        max_util = float(np.max(util_values)) if util_values else 0.0
        # Cache for get_state() — avoids 86× redundant edge sweeps per timestep.
        self._step_mean_util = mean_util
        self._episode_stats['step_mean_utils'].append(mean_util)
        self._episode_stats['step_max_utils'].append(max_util)

        # Build next-state observations.  When obs_indices is provided only
        # compute states for those hosts; the rest get the preallocated empty
        # vector (they are never read by the MADDPG training loop).
        if obs_indices is not None:
            _obs_set = set(obs_indices)
            next_states = [
                self.get_state(hosts[i]) if i in _obs_set else self._empty_state
                for i in range(len(hosts))
            ]
        else:
            next_states = [self.get_state(h) for h in hosts]
        util_var = float(np.var(util_values)) if util_values else 0.0
        info = {
            'packets_sent':      step_sent,
            'packets_delivered': step_delivered,
            'packets_dropped':   step_dropped,
            'delivery_rate':     step_delivered / max(1, step_sent) * 100,
            'packet_loss_rate':  step_dropped / max(1, step_sent) * 100,
            'network_utilization': mean_util,
            'util_variance':     util_var,
            'max_link_utilization': max_util,
            'backlog_packets': current_backlog,
            'reward_components': {
                'fwd_success_term':  reward_components['fwd_success_term'],
                'shaping_term':      reward_components['shaping_term'],
                'chosen_util_term':  reward_components['chosen_util_term'],
                'extreme_cong_term': reward_components['extreme_cong_term'],
                'var_util_term':     reward_components['var_util_term'],
            },
        }
        return next_states, rewards, info

    def _required_bw(self, u: str, v: str) -> float:
        cap_scale = max(self.topology.get_capacity_scale(u, v), 1e-6)
        load_unit = self.flow_packet_size if self.traffic_mode == 'flow' else self.packet_size
        return load_unit / cap_scale

    def _register_link_load(self, u: str, v: str, required_bw: float,
                            cur_util: Optional[float] = None,
                            cap_scale: Optional[float] = None):
        if self.traffic_mode == 'flow':
            self.topology.add_flow_reservation(u, v, required_bw)
            # Keep utilization observations aligned with active reservations.
            self.topology.set_util(u, v, self.topology.get_flow_reserved(u, v))
            return

        if cur_util is None:
            cur_util = self.topology.get_util(u, v)
        if cap_scale is None:
            cap_scale = max(self.topology.get_capacity_scale(u, v), 1e-6)
        self.topology.set_util(u, v, cur_util + self.packet_size / cap_scale)

    def _advance_link_loads_one_step(self):
        if self.traffic_mode == 'flow':
            self.topology.decay_flow_reservations()
            self.topology.sync_utils_from_flow_reservations()
        else:
            self.topology.decay_utils()

    def _select_kpath_next_hops(self, action: np.ndarray, host: str, nbrs: List[str]) -> Dict[str, str]:
        """Select next-hop per access destination using K-shortest-path action vector.

        Action layout: [path0_pref, path1_pref, path2_pref]  (length = K_PATHS)
        A single path index = argmax(action) is selected and applied uniformly
        to ALL destinations at this timestep, matching the paper definition
        A_i ∈ {0, …, K-1}.

        Returns: dict mapping each access destination → chosen next-hop neighbor.
        """
        result: Dict[str, str] = {}
        if not nbrs:
            return result

        # One path-index choice for the entire timestep (paper: A_i ∈ {0,…,K-1})
        path_idx = int(np.argmax(action)) if len(action) >= K_PATHS else 0

        for dst in self.topology.access_nodes:
            paths = self.topology.kpath_cache.get((host, dst), [])

            # Try chosen path first, then alternatives, then path_cache fallback
            chosen = None
            path_order = [path_idx] + [j for j in range(len(paths)) if j != path_idx]
            for pi in path_order:
                if pi < len(paths) and len(paths[pi]) >= 2:
                    nxt = paths[pi][1]
                    if nxt in nbrs:
                        chosen = nxt
                        break

            if chosen is None:
                sp = self.topology.path_cache.get((host, dst), [])
                chosen = sp[1] if len(sp) >= 2 and sp[1] in nbrs else nbrs[0]

            result[dst] = chosen
        return result

    def _next_hop_for_ingress(self, paths: List[List[str]], path_k: int,
                              nbrs: List[str]) -> Optional[str]:
        """Return the first next-hop on path path_k for an ingress packet.

        Tries path_k first, then alternative indices.  Returns None if no
        valid neighbour is reachable on any precomputed path.
        """
        path_order = [path_k] + [j for j in range(len(paths)) if j != path_k]
        for pi in path_order:
            if pi < len(paths) and len(paths[pi]) >= 2:
                nxt = paths[pi][1]
                if nxt in nbrs:
                    return nxt
        return None

    def _next_hop_for_transit(self, paths: List[List[str]], path_k: int,
                              host: str, nbrs: List[str]) -> Optional[str]:
        """Return next-hop for a transit packet on globally-consistent path path_k.

        Locates host's current position in paths[path_k] and returns the next
        node.  Structurally loop-free: paths are simple paths (no repeated
        nodes), so a packet can never revisit a node along this path.
        Falls back to alternative path indices when host is absent from path_k
        (e.g. after a link failure diverted the packet). Returns None if no
        valid neighbour is found.
        """
        path_order = [path_k] + [j for j in range(len(paths)) if j != path_k]
        for pi in path_order:
            if pi >= len(paths):
                continue
            path = paths[pi]
            try:
                pos = path.index(host)
                if pos + 1 < len(path) and path[pos + 1] in nbrs:
                    return path[pos + 1]
            except ValueError:
                pass
        return None

    def get_episode_stats(self) -> Dict:
        s = self._episode_stats
        sent = max(1, s['packets_sent'])
        injected = max(1, s['injected_count'])
        steps = max(1, s['steps'])

        delivered_delay = np.array(s['delivered_delay_samples'], dtype=np.float32)
        delivered_hops = np.array(s['delivered_hop_samples'], dtype=np.float32)
        step_mean_utils = np.array(s['step_mean_utils'], dtype=np.float32)
        step_max_utils = np.array(s['step_max_utils'], dtype=np.float32)
        step_max_node_queue = np.array(s['step_max_node_queue'], dtype=np.float32)
        step_active_queues = np.array(s['step_active_queues'], dtype=np.float32)
        per_agent_steps = max(1, steps * len(self.topology.hosts))

        return {
            'total_reward': s['total_reward'],
            'packets_sent': s['packets_sent'],
            'packets_injected': s['injected_count'],
            'packets_delivered': s['packets_delivered'],
            'packets_dropped': s['packets_dropped'],
            # hop_delivery_frac: fraction of *forwarding events* that are deliveries
            # (kept for transparency but not the primary paper metric)
            'hop_delivery_frac': s['packets_delivered'] / sent * 100,
            # end_to_end_pdr: fraction of *injected* packets successfully delivered
            # NOTE: penalised by in-transit backlog at episode end; use resolved_pdr
            # for a metric that excludes truncated in-flight packets.
            'end_to_end_pdr': s['packets_delivered'] / injected * 100,
            # resolved_pdr: PDR over packets with a final outcome only (delivered or
            # dropped). Excludes packets still in-transit at episode end, making this
            # independent of episode length and routing latency.
            'resolved_pdr': s['packets_delivered'] / max(1, s['packets_delivered'] + s['packets_dropped']) * 100,
            # true_loss_rate: unique dropped packets / unique injected packets.
            # Unlike packet_loss_rate (per-hop), this is the true per-packet loss.
            'true_loss_rate': s['packets_dropped'] / injected * 100,
            # packet_loss_rate: fraction of forwarding events ending in a drop
            # NOTE: denominator is hop-forwarding events (~hops × injected), so
            # this severely underestimates per-packet loss. Kept for legacy.
            'packet_loss_rate': s['packets_dropped'] / sent * 100,
            'goodput_per_step': s['packets_delivered'] / steps,
            'backlog_end': sum(len(q) for q in self.packet_queue.values()),
            'backlog_avg': s['backlog_sum'] / steps,
            'backlog_peak': s['backlog_peak'],
            'delay_mean': float(np.mean(delivered_delay)) if delivered_delay.size else 0.0,
            'delay_p95': float(np.percentile(delivered_delay, 95)) if delivered_delay.size else 0.0,
            'hops_mean': float(np.mean(delivered_hops)) if delivered_hops.size else 0.0,
            'hops_p95': float(np.percentile(delivered_hops, 95)) if delivered_hops.size else 0.0,
            'util_mean': float(np.nanmean(step_mean_utils)) if step_mean_utils.size else 0.0,
            'util_p95': float(np.nanpercentile(step_mean_utils, 95)) if step_mean_utils.size else 0.0,
            'util_max': float(np.nanmax(step_max_utils)) if step_max_utils.size else 0.0,
            'overload_step_fraction': float(
                np.mean(step_max_utils >= 0.80)
            ) if step_max_utils.size else 0.0,
            'drop_ttl': int(s['drop_ttl']),
            'drop_overflow': int(s['drop_overflow']),
            'drop_link_congestion': int(s['drop_link_congestion']),
            'drop_no_path': int(s['drop_no_path']),
            'capacity_block_events': int(s['capacity_block_events']),
            'drop_ttl_rate': s['drop_ttl'] / injected * 100,
            'drop_overflow_rate': s['drop_overflow'] / injected * 100,
            'drop_link_congestion_rate': s['drop_link_congestion'] / injected * 100,
            'drop_no_path_rate': s['drop_no_path'] / injected * 100,
            'capacity_block_per_step': s['capacity_block_events'] / steps,
            'capacity_block_per_injected': s['capacity_block_events'] / injected,
            'max_node_queue_peak': int(np.max(step_max_node_queue)) if step_max_node_queue.size else 0,
            'max_node_queue_avg': float(np.mean(step_max_node_queue)) if step_max_node_queue.size else 0.0,
            'active_queues_avg': float(np.mean(step_active_queues)) if step_active_queues.size else 0.0,
            'traffic_mode': self.traffic_mode,
            'reward_components': {
                k: float(v / per_agent_steps)
                for k, v in s['reward_component_sums'].items()
            },
            'reward_config': self.reward_cfg,
        }

    def get_link_utilization_distribution(self) -> np.ndarray:
        edges = list(self.topology.graph.edges())
        if not edges:
            return np.array([0.0], dtype=np.float32)
        return np.array(
            [self.topology.get_util(u, v) for u, v in edges],
            dtype=np.float32
        )

    # ── OSPF / shortest-path baseline ────────────────────────────────────────

    def ospf_step(self) -> Dict:
        """
        One step under deterministic shortest-path routing.
        Used ONLY during Paper-1 evaluation to produce the OSPF baseline.
        RL agents play no role here.
        """
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        sent = delivered = dropped = 0

        for host, q in list(self.packet_queue.items()):
            for pkt in q:
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
                sent += 1
                path = self.topology.path_cache.get((host, pkt['dst']), [])
                if len(path) < 2:
                    dropped += 1
                    self._episode_stats['drop_no_path'] += 1
                    self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    continue
                nxt = path[1]
                required_bw = self._required_bw(host, nxt)
                if self.topology.avail_bw(host, nxt) < required_bw:
                    self._episode_stats['capacity_block_events'] += 1
                    if self.traffic_mode == 'flow':
                        # Flow mode: immediate drop — same semantics as MADDPG step().
                        # Ensures OSPF/SP baseline is evaluated under identical conditions.
                        dropped += 1
                        self._episode_stats['drop_link_congestion'] += 1
                        self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                        self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    else:
                        # Packet mode: buffer at current node — TTL and hop count unchanged.
                        if len(next_queue[host]) < MAX_NODE_QUEUE_DEPTH:
                            next_queue[host].append(pkt)
                        else:
                            dropped += 1
                            self._episode_stats['drop_overflow'] += 1
                            self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                            self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    continue
                cur = self.topology.get_util(host, nxt)
                cap_scale = self.topology.get_capacity_scale(host, nxt)
                self._register_link_load(host, nxt, required_bw, cur, cap_scale)
                if nxt == pkt['dst']:
                    delivered += 1
                    self._episode_stats['delivered_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['delivered_hop_samples'].append(pkt_hops + 1)
                elif pkt['ttl'] > 0:
                    next_queue[nxt].append({
                        'dst': pkt['dst'],
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                    })
                else:
                    dropped += 1
                    self._episode_stats['drop_ttl'] += 1
                    self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops + 1)

        self._advance_link_loads_one_step()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))
        self.time_step += 1

        # Update episode-level stats so get_episode_stats() reflects OSPF runs.
        self._episode_stats['packets_sent']      += sent
        self._episode_stats['packets_delivered'] += delivered
        self._episode_stats['packets_dropped']   += dropped
        self._episode_stats['steps']             += 1
        current_backlog = sum(len(q) for q in self.packet_queue.values())
        self._episode_stats['backlog_sum']  += current_backlog
        self._episode_stats['backlog_peak'] = max(self._episode_stats['backlog_peak'], current_backlog)
        queue_sizes = [len(q) for q in self.packet_queue.values()]
        self._episode_stats['step_max_node_queue'].append(max(queue_sizes) if queue_sizes else 0)
        self._episode_stats['step_active_queues'].append(sum(1 for size in queue_sizes if size > 0))
        util_values = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        _mean_util = float(np.mean(util_values)) if util_values else 0.0
        _max_util  = float(np.max(util_values))  if util_values else 0.0
        self._episode_stats['step_mean_utils'].append(_mean_util)
        self._episode_stats['step_max_utils'].append(_max_util)

        return {
            'packets_sent':      sent,
            'packets_delivered': delivered,
            'packets_dropped':   dropped,
            'packet_loss_rate':  dropped / max(1, sent) * 100,
        }

    def ospf_queue_level_step(self) -> Dict:
        """
        Queue-level OSPF-like baseline.

        One next-hop is selected per host queue per step using shortest-path
        affinity to queued destinations. This baseline matches MADDPG's
        queue-level decision granularity while still using shortest-path priors.
        """
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        sent = delivered = dropped = 0

        for host, q in list(self.packet_queue.items()):
            nbrs = self.topology.get_neighbors(host)
            if not nbrs:
                for pkt in q:
                    self._episode_stats['dropped_delay_samples'].append(
                        max(0, self.time_step - int(pkt.get('created_at', 0)))
                    )
                    self._episode_stats['dropped_hop_samples'].append(int(pkt.get('hops', 0)))
                    self._episode_stats['drop_no_path'] += 1
                dropped += len(q)
                sent += len(q)
                continue

            chosen = self._select_queue_level_next_hop(host, q, nbrs)
            for pkt in q:
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
                sent += 1
                required_bw = self._required_bw(host, chosen)
                if self.topology.avail_bw(host, chosen) < required_bw:
                    dropped += 1
                    self._episode_stats['capacity_block_events'] += 1
                    self._episode_stats['drop_overflow'] += 1
                    self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    continue
                cur = self.topology.get_util(host, chosen)
                cap_scale = self.topology.get_capacity_scale(host, chosen)
                self._register_link_load(host, chosen, required_bw, cur, cap_scale)
                if chosen == pkt['dst']:
                    delivered += 1
                    self._episode_stats['delivered_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['delivered_hop_samples'].append(pkt_hops + 1)
                elif pkt['ttl'] > 0:
                    next_queue[chosen].append({
                        'dst': pkt['dst'],
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                    })
                else:
                    dropped += 1
                    self._episode_stats['drop_ttl'] += 1
                    self._episode_stats['dropped_delay_samples'].append(max(0, self.time_step - pkt_created))
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops + 1)

        self._advance_link_loads_one_step()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))
        self.time_step += 1

        # Update episode-level stats so get_episode_stats() reflects OSPF runs.
        self._episode_stats['packets_sent']      += sent
        self._episode_stats['packets_delivered'] += delivered
        self._episode_stats['packets_dropped']   += dropped
        self._episode_stats['steps']             += 1
        current_backlog = sum(len(q) for q in self.packet_queue.values())
        self._episode_stats['backlog_sum']  += current_backlog
        self._episode_stats['backlog_peak'] = max(self._episode_stats['backlog_peak'], current_backlog)
        queue_sizes = [len(q) for q in self.packet_queue.values()]
        self._episode_stats['step_max_node_queue'].append(max(queue_sizes) if queue_sizes else 0)
        self._episode_stats['step_active_queues'].append(sum(1 for size in queue_sizes if size > 0))
        util_values = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        _mean_util = float(np.mean(util_values)) if util_values else 0.0
        _max_util  = float(np.max(util_values))  if util_values else 0.0
        self._episode_stats['step_mean_utils'].append(_mean_util)
        self._episode_stats['step_max_utils'].append(_max_util)

        return {
            'packets_sent':      sent,
            'packets_delivered': delivered,
            'packets_dropped':   dropped,
            'packet_loss_rate':  dropped / max(1, sent) * 100,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _inject_packets(self, n: int):
        # Inject at access (PE) nodes — traffic enters the SP backbone only at its
        # edge endpoints, mirroring real MPLS/SR-TE traffic engineering where PE
        # routers originate customer traffic and P routers only transit.
        srcs = self.topology.access_nodes
        dsts = self.topology.access_nodes
        for _ in range(n):
            src = random.choice(srcs)
            dst = random.choice([h for h in dsts if h != src])
            self.packet_queue[src].append({
                'dst': dst,
                'src': src,       # injection source — used for globally-consistent path lookup
                'path_k': None,   # path label set by the ingress agent (MPLS-like LSP)
                'ttl': TTL_INIT,
                'created_at': self.time_step,
                'hops': 0,
                'ingress': None,
            })
        self._episode_stats['injected_count'] += n

    def _build_flow_schedule(self):
        """Create an episode traffic matrix as a schedule of access-to-access flows."""
        access_nodes = list(self.topology.access_nodes)
        if len(access_nodes) < 2:
            return

        self._flow_schedule = defaultdict(list)
        self._active_flows = []
        self._flow_seq = 0

        flow_count = max(1, int(round(self.flow_count_base * self.offered_load_factor)))
        duration = max(1, int(self.flow_duration_steps))
        latest_start = max(1, self.episode_steps - duration + 1)

        for _ in range(flow_count):
            src = random.choice(access_nodes)
            dst_choices = [h for h in access_nodes if h != src]
            if not dst_choices:
                continue
            dst = random.choice(dst_choices)
            start_step = random.randint(1, latest_start)
            flow = {
                'flow_id': self._flow_seq,
                'src': src,
                'dst': dst,
                'start_step': start_step,
                'end_step': min(self.episode_steps, start_step + duration - 1),
            }
            self._flow_schedule[start_step].append(flow)
            self._flow_seq += 1

    def _inject_scheduled_flow_packets(self):
        """Inject one packet per active flow for the current step."""
        if not self._flow_schedule and not self._active_flows:
            return

        current_step = self.time_step
        scheduled = self._flow_schedule.pop(current_step, [])
        if scheduled:
            self._active_flows.extend(scheduled)

        next_active: List[Dict] = []
        injected = 0
        for flow in self._active_flows:
            if flow['start_step'] <= current_step <= flow['end_step']:
                self.packet_queue[flow['src']].append({
                    'dst': flow['dst'],
                    'src': flow['src'],
                    'path_k': None,
                    'ttl': TTL_INIT,
                    'created_at': current_step,
                    'hops': 0,
                    'ingress': None,
                    'flow_id': flow['flow_id'],
                })
                injected += 1
            if flow['end_step'] > current_step:
                next_active.append(flow)

        self._active_flows = next_active
        self._episode_stats['injected_count'] += injected

    def _select_queue_level_next_hop(self, host: str, queue: List[Dict], nbrs: List[str]) -> str:
        """Pick one next-hop for the whole queue using shortest-path affinity."""
        if not queue or not nbrs:
            return nbrs[0] if nbrs else host

        scores = np.zeros(len(nbrs), dtype=np.float64)
        for pkt in queue:
            dst = pkt['dst']
            for i, nb in enumerate(nbrs):
                path = self.topology.path_cache.get((nb, dst), [])
                if not path:
                    continue
                # Prefer neighbors that keep packets closer to destination.
                scores[i] += 1.0 / max(1, len(path) - 1)

        best_idx = int(np.argmax(scores))
        return nbrs[best_idx]

    @staticmethod
    def _blank_stats() -> Dict:
        return {
            'total_reward': 0.0,
            'packets_sent': 0,
            'packets_injected': 0,
            'injected_count': 0,
            'packets_delivered': 0,
            'packets_dropped': 0,
            'steps': 0,
            'backlog_sum': 0,
            'backlog_peak': 0,
            'delivered_delay_samples': [],
            'delivered_hop_samples': [],
            'dropped_delay_samples': [],
            'dropped_hop_samples': [],
            'drop_ttl': 0,
            'drop_overflow': 0,
            'drop_link_congestion': 0,
            'drop_no_path': 0,
            'capacity_block_events': 0,
            'step_mean_utils': [],
            'step_max_utils': [],
            'step_max_node_queue': [],
            'step_active_queues': [],
            'reward_component_sums': {
                'fwd_success_term':  0.0,
                'shaping_term':      0.0,
                'chosen_util_term':  0.0,
                'extreme_cong_term': 0.0,
                'var_util_term':     0.0,
            },
        }


class NetworkEnv:
    """Thin wrapper — matches the interface used by StandaloneExperimentRunner."""

    def __init__(self, engine: NetworkEngine):
        self.engine = engine

    def reset(self) -> List[np.ndarray]:
        self.engine.reset()
        return [self.engine.get_state(h) for h in self.engine.get_all_hosts()]

    def step(self, actions: List[np.ndarray], obs_indices=None):
        return self.engine.step(actions, obs_indices=obs_indices)

    def get_stats(self) -> Dict:
        return self.engine.get_episode_stats()