"""
Neuron Metrics Module.

This module defines the NeuronMetrics dataclass used to track the state and history
of individual SSNC neurons.
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque

@dataclass
class NeuronMetrics:
    """Comprehensive neuron tracking with multi-component utility."""
    neuron_id: int
    layer_id: int
    age: int = 0
    
    # Soft deletion
    marked_for_deletion: bool = False
    marked_cycles: int = 0
    soft_deleted: bool = False
    soft_deleted_at: Optional[int] = None
    gate_value: float = 1.0
    
    # Hard deletion (Phase 7)
    hard_deleted: bool = False
    hard_deleted_at: Optional[int] = None
    
    # Creation tracking (Phase 8)
    parent_id: Optional[int] = None
    created_at: Optional[int] = None
    creation_method: str = "initial"  # "initial", "duplicate", "split"
    # New Phase 11: Stability
    birth_step: int = 0
    probation_period: int = 500
    
    # === ENHANCED UTILITY METRICS ===
    # Multi-component utility tracking
    activation_ema: float = 0.0          # EMA of activation magnitude
    gradient_magnitude: float = 0.0       # EMA of gradient magnitude
    weight_norm: float = 0.0              # L2 norm of associated weights
    correct_contribution: int = 0         # Count of correct predictions contributed to
    total_contribution: int = 0           # Total predictions contributed to
    
    # Peak utility tracking (for resurrection decisions)
    max_utility: float = 0.0              # Historical peak utility
    
    # Dead ReLU detection
    is_dead_relu: bool = False            # True if stuck at zero activation
    dead_relu_count: int = 0              # Consecutive zero activation samples
    
    # Utility history
    utility_history: deque = None
    
    def __post_init__(self):
        if self.utility_history is None:
            self.utility_history = deque(maxlen=100)
    
    def compute_composite_utility(self, weights: tuple = (0.4, 0.3, 0.1, 0.2)) -> float:
        """
        Compute weighted multi-component utility.
        
        Args:
            weights: (w_activation, w_gradient, w_weight, w_task) weights
        
        Returns:
            Composite utility score in [0, 1]
        """
        w_act, w_grad, w_weight, w_task = weights
        
        # Task importance = correct / total (if any contributions)
        task_importance = 0.0
        if self.total_contribution > 0:
            task_importance = self.correct_contribution / self.total_contribution
        
        # Dead ReLU penalty: if dead, utility is near zero regardless
        if self.is_dead_relu:
            return 0.01  # Not exactly zero (allow for re-initialization logic)
        
        utility = (
            w_act * min(1.0, self.activation_ema) +
            w_grad * min(1.0, self.gradient_magnitude) +
            w_weight * min(1.0, self.weight_norm / 10.0) +  # Normalize weight norm
            w_task * task_importance
        )
        
        # Update max utility
        if utility > self.max_utility:
            self.max_utility = utility
        
        return max(0.0, min(1.0, utility))
