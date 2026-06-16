# Complete init file to make modules importable
"""
MADDPG Clean Implementation Package
Self-contained MADDPG implementation with corrected adversarial attack framework
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .maddpg_implementation import MADDPG, Agent, ActorNetwork, CriticNetwork, ReplayBuffer
from .network_environment import NetworkEngine, NetworkEnv, NetworkTopology

__all__ = [
    'MADDPG', 
    'Agent', 
    'ActorNetwork', 
    'CriticNetwork', 
    'ReplayBuffer',
    'NetworkEngine', 
    'NetworkEnv', 
    'NetworkTopology', 
    'FlowManager'
]