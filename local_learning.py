"""
SSNC PHASE 4: Local Neuron Learning

Neurons now adapt their own weights based on local signals and utility.

Key principle: Small, bounded updates tied to utility.
High utility neurons learn more aggressively.
Low utility neurons learn conservatively (or not at all).

CRITICAL SAFETY:
- All updates are clipped
- Learning rates are bounded
- We can disable local learning easily
"""

import torch
import numpy as np
from typing import Optional
from ssnc_neuron import SSNCNeuron


class SSNCNeuronWithLearning(SSNCNeuron):
    """
    Extended SSNC neuron that can learn locally.
    
    Adds to base SSNCNeuron:
    - Local weight updates
    - Utility-modulated learning rate
    - Update history tracking
    - Safety mechanisms
    """
    
    def __init__(
        self,
        neuron_id: int,
        layer_id: int,
        activation_window: int = 100,
        error_window: int = 50,
        momentum: float = 0.9,
        # New PHASE 4 parameters
        local_lr: float = 0.001,
        utility_modulation: bool = True,
        max_update_norm: float = 0.01
    ):
        """
        Args:
            local_lr: Base learning rate for local updates
            utility_modulation: If True, scale lr by utility
            max_update_norm: Maximum allowed update magnitude (safety)
        """
        super().__init__(neuron_id, layer_id, activation_window, error_window, momentum)
        
        # Local learning parameters
        self.local_lr = local_lr
        self.utility_modulation = utility_modulation
        self.max_update_norm = max_update_norm
        
        # Update tracking
        self.weight_updates = []  # History of updates applied
        self.update_norms = []    # Magnitude of each update
        self.learning_enabled = True  # Can disable locally
        
        # Safety counters
        self.n_updates_applied = 0
        self.n_updates_clipped = 0
    
    def compute_local_learning_rate(self) -> float:
        """
        Compute effective learning rate based on utility.
        
        Returns:
            Effective learning rate (can be zero for low-utility neurons)
        """
        if not self.learning_enabled:
            return 0.0
        
        base_lr = self.local_lr
        
        if self.utility_modulation:
            # Scale by utility
            # High utility (0.5+) → learn at full rate
            # Medium utility (0.2-0.5) → learn at reduced rate
            # Low utility (<0.2) → barely learn or not at all
            
            if self.utility > 0.5:
                scale = 1.0
            elif self.utility > 0.2:
                # Linear scaling between 0.2 and 0.5
                scale = (self.utility - 0.2) / 0.3
            else:
                # Very low utility - minimal learning
                scale = 0.1
            
            effective_lr = base_lr * scale
        else:
            effective_lr = base_lr
        
        return effective_lr
    
    def compute_local_gradient(
        self,
        activation: float,
        error_signal: float,
        input_activations: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute local weight update using only local information.
        
        Uses a simplified Hebbian-like rule:
        Δw ∝ error_signal * input_activations
        
        This is a local approximation of backprop.
        
        Args:
            activation: This neuron's current activation
            error_signal: Downstream error signal
            input_activations: Activations from previous layer (optional)
            
        Returns:
            Weight update vector (or None if inputs not available)
        """
        if input_activations is None:
            return None
        
        # Hebbian-like update
        # Δw = lr * error * inputs
        gradient = error_signal * input_activations
        
        return gradient
    
    def apply_local_update(
        self,
        gradient: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply local weight update with safety checks.
        
        Args:
            gradient: Computed gradient
            weights: Current weights (incoming to this neuron)
            
        Returns:
            Updated weights
        """
        # Compute effective learning rate
        lr = self.compute_local_learning_rate()
        
        if lr == 0.0:
            return weights  # No update
        
        # Compute update
        update = -lr * gradient  # Negative for gradient descent
        
        # Safety: clip update norm
        update_norm = np.linalg.norm(update)
        
        if update_norm > self.max_update_norm:
            # Clip to max norm
            update = update * (self.max_update_norm / update_norm)
            self.n_updates_clipped += 1
        
        # Apply update
        new_weights = weights + update
        
        # Track for monitoring
        self.weight_updates.append(update)
        self.update_norms.append(np.linalg.norm(update))
        self.n_updates_applied += 1
        
        # Keep history bounded
        if len(self.weight_updates) > 100:
            self.weight_updates.pop(0)
            self.update_norms.pop(0)
        
        return new_weights
    
    def get_learning_stats(self) -> dict:
        """Get statistics about local learning."""
        if not self.update_norms:
            return {
                'n_updates': 0,
                'mean_update_norm': 0.0,
                'std_update_norm': 0.0,
                'n_clipped': 0,
                'clip_rate': 0.0,
                'effective_lr': self.compute_local_learning_rate(),
                'learning_enabled': self.learning_enabled
            }
        
        return {
            'n_updates': self.n_updates_applied,
            'mean_update_norm': float(np.mean(self.update_norms)),
            'std_update_norm': float(np.std(self.update_norms)),
            'max_update_norm': float(np.max(self.update_norms)),
            'n_clipped': self.n_updates_clipped,
            'clip_rate': self.n_updates_clipped / max(self.n_updates_applied, 1),
            'effective_lr': self.compute_local_learning_rate(),
            'learning_enabled': self.learning_enabled
        }
    
    def disable_learning(self):
        """Disable local learning for this neuron."""
        self.learning_enabled = False
    
    def enable_learning(self):
        """Enable local learning for this neuron."""
        self.learning_enabled = True
    
    def reset_learning_stats(self):
        """Reset learning statistics."""
        self.weight_updates = []
        self.update_norms = []
        self.n_updates_applied = 0
        self.n_updates_clipped = 0


class LocalLearningManager:
    """
    Manages local learning for a layer of neurons.
    
    Handles:
    - Collecting input activations
    - Computing local gradients
    - Applying updates to PyTorch weights
    - Tracking statistics
    """
    
    def __init__(self, neurons: list, enable_learning: bool = True):
        """
        Args:
            neurons: List of SSNCNeuronWithLearning instances
            enable_learning: Whether to enable local learning
        """
        self.neurons = neurons
        self.enable_learning = enable_learning
        
        # Cached inputs for local gradient computation
        self.cached_inputs = None
    
    def cache_inputs(self, inputs: torch.Tensor):
        """
        Cache layer inputs for local gradient computation.
        
        Args:
            inputs: Input tensor (batch_size, input_dim)
        """
        # Average over batch for local learning
        self.cached_inputs = inputs.mean(dim=0).detach().cpu().numpy()
    
    def apply_local_updates(
        self,
        layer: torch.nn.Linear,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ):
        """
        Apply local learning updates to layer weights.
        
        Args:
            layer: PyTorch Linear layer
            activations: Neuron activations (batch_size, n_neurons)
            gradients: Error gradients (batch_size, n_neurons)
        """
        if not self.enable_learning or self.cached_inputs is None:
            return
        
        # Get current weights
        weights = layer.weight.data.cpu().numpy()  # Shape: (n_neurons, input_dim)
        
        # Average activations and gradients over batch
        batch_acts = activations.mean(dim=0).detach().cpu().numpy()
        batch_grads = gradients.mean(dim=0).detach().cpu().numpy()
        
        # Update each neuron
        for neuron_idx, neuron in enumerate(self.neurons):
            if not neuron.learning_enabled:
                continue
            
            # Get this neuron's weights
            neuron_weights = weights[neuron_idx]
            
            # Compute local gradient
            local_grad = neuron.compute_local_gradient(
                activation=batch_acts[neuron_idx],
                error_signal=batch_grads[neuron_idx],
                input_activations=self.cached_inputs
            )
            
            if local_grad is not None:
                # Apply update
                new_weights = neuron.apply_local_update(
                    gradient=local_grad,
                    weights=neuron_weights
                )
                
                # Write back to layer
                weights[neuron_idx] = new_weights
        
        # Update layer weights
        layer.weight.data = torch.from_numpy(weights).float().to(layer.weight.device)
    
    def get_layer_learning_stats(self) -> dict:
        """Get aggregated learning statistics for layer."""
        all_stats = [n.get_learning_stats() for n in self.neurons]
        
        if not all_stats:
            return {}
        
        return {
            'total_updates': sum(s['n_updates'] for s in all_stats),
            'mean_update_norm': np.mean([s['mean_update_norm'] for s in all_stats]),
            'total_clipped': sum(s['n_clipped'] for s in all_stats),
            'avg_effective_lr': np.mean([s['effective_lr'] for s in all_stats]),
            'neurons_learning': sum(1 for s in all_stats if s['learning_enabled'])
        }


if __name__ == "__main__":
    print("SSNC PHASE 4: Local Neuron Learning")
    print("="*60)
    
    # Create a learning neuron
    neuron = SSNCNeuronWithLearning(
        neuron_id=0,
        layer_id=0,
        local_lr=0.001
    )
    
    print(f"Created learning neuron: {neuron}")
    print(f"Base learning rate: {neuron.local_lr}")
    print(f"Utility modulation: {neuron.utility_modulation}")
    print()
    
    # Simulate learning with different utilities
    print("Simulating learning with different utility levels:")
    print("-" * 60)
    
    for utility_level in [0.1, 0.3, 0.6]:
        neuron.utility = utility_level
        effective_lr = neuron.compute_local_learning_rate()
        print(f"Utility={utility_level:.1f} → Effective LR={effective_lr:.6f} "
              f"(scale={effective_lr/neuron.local_lr:.2f}x)")
    
    print()
    
    # Test local update
    print("Testing local weight update:")
    print("-" * 60)
    
    neuron.utility = 0.5  # Medium-high utility
    
    # Simulate inputs, activation, error
    weights = np.random.randn(10) * 0.1
    inputs = np.random.randn(10)
    activation = 0.5
    error = -0.1
    
    # Compute gradient
    gradient = neuron.compute_local_gradient(activation, error, inputs)
    print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
    
    # Apply update
    new_weights = neuron.apply_local_update(gradient, weights)
    weight_change = np.linalg.norm(new_weights - weights)
    print(f"Weight change norm: {weight_change:.6f}")
    print(f"Update was clipped: {neuron.n_updates_clipped > 0}")
    
    # Get stats
    stats = neuron.get_learning_stats()
    print(f"\nLearning statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ PHASE 4 local learning implementation complete!")
    print("Next: Integrate with SSNC-MLP")
