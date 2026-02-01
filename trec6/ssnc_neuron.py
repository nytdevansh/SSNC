"""
SSNC PHASE 2: SSNC-Neuron with Local State

Each neuron tracks its own statistics and computes local signals.
In PHASE 2: OBSERVATION ONLY - no learning, no structural changes.

Key principle: Neurons know themselves before they act.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Dict


class SSNCNeuron:
    """
    A neuron that maintains local state and tracks its own utility.
    
    PHASE 2: Observe only (no learning yet)
    PHASE 4: Add local learning
    PHASE 6+: Respond to observer commands
    
    Each neuron tracks:
    - Activation statistics (mean, variance)
    - Inactivity counter
    - Downstream error correlation
    - Age (timesteps since creation)
    - Utility score (importance metric)
    """
    
    def __init__(
        self,
        neuron_id: int,
        layer_id: int,
        activation_window: int = 100,
        error_window: int = 50,
        momentum: float = 0.9
    ):
        """
        Args:
            neuron_id: Unique ID within layer
            layer_id: Which layer this neuron belongs to
            activation_window: Window for activation tracking
            error_window: Window for error correlation
            momentum: EMA momentum for statistics
        """
        self.neuron_id = neuron_id
        self.layer_id = layer_id
        
        # === Local State ===
        
        # Activation statistics
        self.activation_mean = 0.0
        self.activation_var = 0.0
        self.activation_history = deque(maxlen=activation_window)
        
        # Inactivity tracking
        self.inactivity_counter = 0
        self.inactivity_threshold = 1e-4  # Consider "active" if |activation| > threshold
        
        # Error correlation buffer (for downstream utility)
        self.error_buffer = deque(maxlen=error_window)
        self.error_correlation = 0.0
        
        # Age tracking
        self.age = 0  # Timesteps since creation
        
        # Utility (computed from above signals)
        self.utility = 0.0
        self.utility_history = deque(maxlen=50)
        
        # Update tracking
        self.n_updates = 0
        self.momentum = momentum
        
        # Status flags
        self.is_active = True  # For soft deletion in later phases
        self.is_marked_for_deletion = False
    
    def observe_activation(self, activation: float):
        """
        Update neuron's view of its own activation.
        
        Called during forward pass to track activation patterns.
        
        Args:
            activation: This neuron's output value (scalar)
        """
        # Track raw history
        self.activation_history.append(activation)
        
        # Update statistics with EMA
        if self.n_updates == 0:
            self.activation_mean = activation
            self.activation_var = 0.0
        else:
            delta = activation - self.activation_mean
            self.activation_mean += (1 - self.momentum) * delta
            self.activation_var = (
                self.momentum * self.activation_var +
                (1 - self.momentum) * delta * delta
            )
        
        # Update inactivity counter
        if abs(activation) < self.inactivity_threshold:
            self.inactivity_counter += 1
        else:
            self.inactivity_counter = 0  # Reset on activity
        
        # Increment counters
        self.n_updates += 1
        self.age += 1
    
    def observe_error(self, error_signal: float):
        """
        Update neuron's view of downstream error.
        
        This is used to estimate utility - how much does this neuron
        contribute to reducing loss?
        
        Args:
            error_signal: Gradient or error from downstream
        """
        self.error_buffer.append(error_signal)
        
        # Compute correlation between activation and error
        # High correlation = neuron is useful for reducing error
        if len(self.activation_history) > 10 and len(self.error_buffer) > 10:
            # Get recent windows
            recent_acts = list(self.activation_history)[-10:]
            recent_errs = list(self.error_buffer)[-10:]
            
            # Compute correlation
            acts_array = np.array(recent_acts)
            errs_array = np.array(recent_errs)
            
            if np.std(acts_array) > 1e-6 and np.std(errs_array) > 1e-6:
                correlation = np.corrcoef(acts_array, errs_array)[0, 1]
                
                # Handle NaN (can happen with perfect correlation of zero signals)
                if not np.isnan(correlation):
                    # Smooth correlation with EMA
                    if self.error_correlation == 0.0:
                        self.error_correlation = abs(correlation)
                    else:
                        self.error_correlation = (
                            0.9 * self.error_correlation +
                            0.1 * abs(correlation)
                        )
    
    def compute_utility(self) -> float:
        """
        Compute neuron's utility score.
        
        Utility combines:
        - Activation magnitude (is neuron responsive?)
        - Activation variance (does neuron differentiate inputs?)
        - Error correlation (does neuron help reduce loss?)
        - Recency (newer neurons get grace period)
        
        Returns:
            Utility score in [0, 1] (approximately)
        """
        # Component 1: Activation magnitude (normalized)
        # Neurons that fire more strongly are potentially more useful
        activation_magnitude = abs(self.activation_mean)
        activation_score = np.tanh(activation_magnitude)  # Saturate at 1
        
        # Component 2: Activation variance
        # Neurons that vary their output are more useful than constant ones
        variance_score = np.tanh(np.sqrt(self.activation_var))
        
        # Component 3: Error correlation
        # Neurons correlated with error reduction are useful
        error_score = self.error_correlation
        
        # Component 4: Inactivity penalty
        # Neurons that haven't fired recently are less useful
        if self.inactivity_counter > 50:
            inactivity_penalty = np.exp(-self.inactivity_counter / 100.0)
        else:
            inactivity_penalty = 1.0
        
        # Component 5: Age bonus (grace period for young neurons)
        if self.age < 1000:
            age_bonus = 1.0 + 0.2 * (1000 - self.age) / 1000.0
        else:
            age_bonus = 1.0
        
        # Combine components (weighted average)
        utility = (
            0.2 * activation_score +
            0.2 * variance_score +
            0.4 * error_score +
            0.2 * (activation_score * variance_score)  # Interaction term
        ) * inactivity_penalty * age_bonus
        
        # Smooth utility
        self.utility = utility
        self.utility_history.append(utility)
        
        return utility
    
    def get_state_dict(self) -> Dict:
        """
        Get complete state for logging/checkpointing.
        
        Returns:
            Dictionary with all neuron state
        """
        return {
            'neuron_id': self.neuron_id,
            'layer_id': self.layer_id,
            'activation_mean': self.activation_mean,
            'activation_var': self.activation_var,
            'inactivity_counter': self.inactivity_counter,
            'error_correlation': self.error_correlation,
            'age': self.age,
            'utility': self.utility,
            'n_updates': self.n_updates,
            'is_active': self.is_active,
            'is_marked_for_deletion': self.is_marked_for_deletion,
            # Statistics
            'mean_utility': np.mean(self.utility_history) if self.utility_history else 0.0,
            'std_utility': np.std(self.utility_history) if self.utility_history else 0.0,
        }
    
    def reset(self):
        """Reset all statistics (for new tasks or ablations)."""
        self.activation_mean = 0.0
        self.activation_var = 0.0
        self.activation_history.clear()
        self.inactivity_counter = 0
        self.error_buffer.clear()
        self.error_correlation = 0.0
        self.utility = 0.0
        self.utility_history.clear()
        self.n_updates = 0
        # Don't reset age - that's permanent
    
    def __repr__(self) -> str:
        return (
            f"SSNCNeuron(id={self.neuron_id}, layer={self.layer_id}, "
            f"utility={self.utility:.3f}, age={self.age}, "
            f"inactive={self.inactivity_counter})"
        )


class SSNCNeuronLayer(nn.Module):
    """
    Manages a layer of SSNC neurons.
    
    Wraps a standard PyTorch layer with SSNC neuron tracking.
    """
    
    def __init__(self, layer_id: int, n_neurons: int):
        """
        Args:
            layer_id: Which layer in the network
            n_neurons: Number of neurons in this layer
        """
        super().__init__()
        self.layer_id = layer_id
        self.n_neurons = n_neurons
        
        # Create SSNC neurons
        self.neurons = [
            SSNCNeuron(neuron_id=i, layer_id=layer_id)
            for i in range(n_neurons)
        ]
    
    def observe_forward(self, activations: torch.Tensor):
        """
        Update all neurons with their activations.
        
        Args:
            activations: Tensor of shape (batch_size, n_neurons)
        """
        # Average over batch for per-neuron statistics
        batch_mean_acts = activations.mean(dim=0).detach().cpu().numpy()
        
        for neuron_idx, activation in enumerate(batch_mean_acts):
            self.neurons[neuron_idx].observe_activation(float(activation))
    
    def observe_backward(self, gradients: torch.Tensor):
        """
        Update all neurons with downstream error signals.
        
        Args:
            gradients: Tensor of shape (batch_size, n_neurons)
        """
        # Average over batch
        batch_mean_grads = gradients.mean(dim=0).detach().cpu().numpy()
        
        for neuron_idx, gradient in enumerate(batch_mean_grads):
            self.neurons[neuron_idx].observe_error(float(gradient))
    
    def compute_all_utilities(self):
        """Compute utility for all neurons."""
        for neuron in self.neurons:
            neuron.compute_utility()
    
    def get_layer_summary(self) -> Dict:
        """
        Get summary statistics for the layer.
        
        Returns:
            Dict with aggregate statistics
        """
        utilities = [n.utility for n in self.neurons]
        activations = [n.activation_mean for n in self.neurons]
        variances = [n.activation_var for n in self.neurons]
        inactive_counts = [n.inactivity_counter for n in self.neurons]
        
        return {
            'layer_id': self.layer_id,
            'n_neurons': self.n_neurons,
            'utilities': {
                'mean': np.mean(utilities),
                'std': np.std(utilities),
                'min': np.min(utilities),
                'max': np.max(utilities),
                'distribution': utilities
            },
            'activations': {
                'mean': np.mean(activations),
                'std': np.std(activations),
            },
            'variances': {
                'mean': np.mean(variances),
                'std': np.std(variances),
            },
            'inactive_neurons': sum(1 for c in inactive_counts if c > 50),
            'max_inactivity': max(inactive_counts) if inactive_counts else 0
        }
    
    def get_all_neuron_states(self) -> list:
        """Get state dict for all neurons."""
        return [neuron.get_state_dict() for neuron in self.neurons]


if __name__ == "__main__":
    print("SSNC PHASE 2: SSNC-Neuron Local State Tracking")
    print("="*60)
    
    # Create a test neuron
    neuron = SSNCNeuron(neuron_id=0, layer_id=1)
    print(f"\nCreated: {neuron}")
    
    # Simulate some activations
    print("\nSimulating activation sequence...")
    np.random.seed(42)
    
    for t in range(100):
        # Simulate activation (varying over time)
        act = 0.5 * np.sin(t / 10.0) + 0.1 * np.random.randn()
        neuron.observe_activation(act)
        
        # Simulate error signal (correlated with activation)
        err = -0.3 * act + 0.05 * np.random.randn()
        neuron.observe_error(err)
        
        # Compute utility periodically
        if t % 10 == 0:
            utility = neuron.compute_utility()
            if t % 50 == 0:
                print(f"  t={t:3d}: utility={utility:.3f}, "
                      f"mean_act={neuron.activation_mean:.3f}, "
                      f"inactive={neuron.inactivity_counter}")
    
    # Final state
    print(f"\nFinal state:")
    state = neuron.get_state_dict()
    for key, value in state.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test a layer
    print("\n" + "="*60)
    print("Testing SSNCNeuronLayer...")
    
    layer = SSNCNeuronLayer(layer_id=0, n_neurons=10)
    
    # Simulate batch of activations
    activations = torch.randn(32, 10)  # Batch of 32
    layer.observe_forward(activations)
    
    # Simulate gradients
    gradients = torch.randn(32, 10)
    layer.observe_backward(gradients)
    
    # Compute utilities
    layer.compute_all_utilities()
    
    # Get summary
    summary = layer.get_layer_summary()
    print(f"\nLayer summary:")
    print(f"  Mean utility: {summary['utilities']['mean']:.4f}")
    print(f"  Utility std: {summary['utilities']['std']:.4f}")
    print(f"  Inactive neurons: {summary['inactive_neurons']}")
    
    print("\n✅ PHASE 2 neuron implementation complete!")
    print("Next: Integrate with InstrumentedMLP")
