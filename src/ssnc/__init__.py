"""
SSNC Module - Self-Structuring Neural Classifier

This module provides the complete SSNC implementation with:
- SSNCNeuron: Enhanced neuron with multi-component utility
- NeuronMetrics: Tracking metrics for neurons
- AdaptiveObserver: Network state analysis and regime detection
- AutonomousSSNCController: Self-learning plasticity controller
- CompleteSSNC_MLP: Main network class (imported from parent for now)
"""

# Core classes
from src.ssnc.neuron import SSNCNeuron
from src.ssnc.metrics import NeuronMetrics
from src.ssnc.observer import AdaptiveObserver
from src.ssnc.controller import (
    AutonomousSSNCController,
    AnomalyDetector,
    ExperienceBuffer,
    ActionBandit,
    Action,
    NetworkState
)

# Main network class
from src.ssnc.network import CompleteSSNC_MLP

__all__ = [
    'SSNCNeuron',
    'NeuronMetrics', 
    'AdaptiveObserver',
    'AutonomousSSNCController',
    'AnomalyDetector',
    'ExperienceBuffer',
    'ActionBandit',
    'Action',
    'NetworkState',
    'CompleteSSNC_MLP'
]
