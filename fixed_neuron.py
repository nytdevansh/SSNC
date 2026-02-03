"""
Fixed neuron.py - Proper utility scaling with sigmoid
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class SSNCNeuron:
    """Enhanced neuron pointer / metadata with multi-component utility."""
    neuron_id: int
    layer_id: int
    utility: float = 0.5
    age: int = 0
    creation_method: str = "initial"
    parent_id: Optional[int] = None
    probation_period: int = 500
    hard_deleted: bool = False
    
    # Activation tracking
    activation_history: List[float] = field(default_factory=list)
    avg_activation: float = 0.0
    
    # Gradient tracking
    avg_gradient: float = 0.0
    
    # Dead ReLU detection
    dead_relu_count: int = 0
    is_dead_relu: bool = False
    DEAD_RELU_THRESHOLD: int = 50
    
    # Weight norm
    weight_norm: float = 0.0
    
    # Task contribution
    correct_predictions: int = 0
    total_predictions: int = 0
    
    # Per-class tracking
    class_contributions: Dict[int, int] = field(default_factory=dict)
    prediction_confidence_sum: float = 0.0
    prediction_confidence_count: int = 0
    
    # Peak utility for resurrection
    max_utility: float = 0.0
    
    # Utility component weights
    utility_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.1, 0.2)
    
    def observe_activation(self, value: float):
        """Update activation statistics and dead ReLU detection."""
        if self.hard_deleted:
            return
        
        # EMA update
        momentum = 0.95
        self.avg_activation = (momentum * self.avg_activation) + ((1 - momentum) * abs(value))
        
        # Dead ReLU detection
        if abs(value) < 1e-6:
            self.dead_relu_count += 1
            if self.dead_relu_count >= self.DEAD_RELU_THRESHOLD:
                self.is_dead_relu = True
        else:
            self.dead_relu_count = 0
            self.is_dead_relu = False

    def observe_gradient(self, value: float):
        """Update gradient magnitude statistics."""
        if self.hard_deleted:
            return
        
        momentum = 0.95
        self.avg_gradient = (momentum * self.avg_gradient) + ((1 - momentum) * abs(value))

    def observe_error(self, value: float):
        """Legacy method - now routes to observe_gradient."""
        self.observe_gradient(value)
    
    def record_prediction(self, was_correct: bool, class_id: int = None, confidence: float = None):
        """Track contribution to predictions for task importance."""
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
            if class_id is not None:
                self.class_contributions[class_id] = self.class_contributions.get(class_id, 0) + 1
        
        if confidence is not None:
            self.prediction_confidence_sum += confidence
            self.prediction_confidence_count += 1
    
    def get_avg_confidence(self) -> float:
        """Return average prediction confidence."""
        if self.prediction_confidence_count == 0:
            return 0.5
        return self.prediction_confidence_sum / self.prediction_confidence_count
    
    def get_class_importance(self, class_id: int) -> float:
        """Return contribution ratio for a specific class."""
        class_correct = self.class_contributions.get(class_id, 0)
        if self.correct_predictions == 0:
            return 0.0
        return class_correct / self.correct_predictions

    def compute_utility(self):
        """
        Compute multi-component utility with proper scaling.
        
        FIXED VERSION - Uses sigmoid to map raw utilities to 0-1 range.
        """
        if self.hard_deleted:
            self.utility = 0.0
            return
        
        # Dead ReLU special case
        if self.is_dead_relu:
            self.utility = 0.01
            return
        
        w_act, w_grad, w_weight, w_task = self.utility_weights
        
        # Task importance
        task_importance = 0.0
        if self.total_predictions > 0:
            task_importance = self.correct_predictions / self.total_predictions
        
        # Compute weighted utility (raw range typically 0.0 - 0.1 for deep networks)
        raw_utility = (
            w_act * min(1.0, self.avg_activation) +
            w_grad * min(1.0, self.avg_gradient) +
            w_weight * min(1.0, self.weight_norm / 10.0) +
            w_task * task_importance
        )
        
        # PROPER SCALING using sigmoid
        # Maps raw range [0.0, 0.1] to approximately [0.0, 1.0]
        # Center at 0.05 (typical value), steep slope for separation
        try:
            scaled = 1.0 / (1.0 + np.exp(-20.0 * (raw_utility - 0.05)))
            self.utility = float(max(0.0, min(1.0, scaled)))
        except:
            # Fallback if numpy not available or overflow
            # Simple linear scaling as backup
            self.utility = min(1.0, max(0.0, raw_utility * 15.0))
        
        # Track peak
        if self.utility > self.max_utility:
            self.max_utility = self.utility
