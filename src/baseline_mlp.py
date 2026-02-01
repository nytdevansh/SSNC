"""
SSNC PHASE 1: Baseline Instrumented MLP

A standard MLP with comprehensive logging hooks for observability.
No SSNC logic yet - just a normal model we can watch carefully.

Key features:
- Simple MLP architecture (configurable depth/width)
- Per-neuron activation tracking
- Per-layer norm tracking
- Smoothed loss tracking
- Easy to freeze/unfreeze for later phases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class InstrumentedMLP(nn.Module):
    """
    Standard MLP with hooks for detailed observation.
    
    This is our baseline - trains normally, but we can see everything.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Choose smooth activation (required for SSNC later)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        else:
            raise ValueError(f"Use smooth activations only: tanh or softplus")
        
        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        # Statistics tracking (per neuron, per layer)
        self.stats = NeuronStatistics(hidden_dims)
        
        # Loss history for smoothing
        self.loss_history = []
        self.loss_window = 100
        
        # Register forward hooks for automatic tracking
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward hooks on each hidden layer to track activations.
        This happens automatically during forward pass.
        """
        for layer_idx, layer in enumerate(self.layers[:-1]):  # Exclude output layer
            def make_hook(idx):
                def hook(module, input, output):
                    # Track activation statistics for this layer
                    with torch.no_grad():
                        self.stats.update(idx, output)
                return hook
            
            layer.register_forward_hook(make_hook(layer_idx))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with automatic stat tracking via hooks."""
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x
    
    def update_loss(self, loss: float):
        """Track loss with smoothing."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.loss_window:
            self.loss_history.pop(0)
    
    def get_smoothed_loss(self) -> Optional[float]:
        """Get exponentially smoothed loss."""
        if not self.loss_history:
            return None
        return np.mean(self.loss_history)
    
    def get_layer_stats(self) -> Dict:
        """Get current statistics for all layers."""
        return self.stats.get_summary()
    
    def freeze_weights(self):
        """Freeze all weights (for PHASE 2+)."""
        for param in self.parameters():
            param.requires_grad = False
        print("🔒 All weights frozen")
    
    def unfreeze_weights(self):
        """Unfreeze all weights."""
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 All weights unfrozen")
    
    def reset_stats(self):
        """Reset all statistics (useful between experiments)."""
        self.stats.reset()
        self.loss_history = []


class NeuronStatistics:
    """
    Tracks per-neuron and per-layer statistics.
    
    For each layer we track:
    - Mean activation per neuron
    - Variance per neuron
    - Layer-wide output norm
    
    Uses exponential moving averages for stability.
    """
    
    def __init__(self, hidden_dims: List[int], momentum: float = 0.9):
        self.hidden_dims = hidden_dims
        self.momentum = momentum
        
        # Per-neuron statistics
        # Format: {layer_idx: {neuron_idx: {'mean': ..., 'var': ...}}}
        self.neuron_stats = defaultdict(lambda: defaultdict(dict))
        
        # Per-layer statistics
        # Format: {layer_idx: {'output_norm': ...}}
        self.layer_stats = defaultdict(dict)
        
        # Initialize storage
        for layer_idx, n_neurons in enumerate(hidden_dims):
            for neuron_idx in range(n_neurons):
                self.neuron_stats[layer_idx][neuron_idx] = {
                    'mean': 0.0,
                    'var': 0.0,
                    'n_updates': 0
                }
            
            self.layer_stats[layer_idx] = {
                'output_norm': 0.0,
                'n_updates': 0
            }
    
    def update(self, layer_idx: int, activations: torch.Tensor):
        """
        Update statistics for a layer given a batch of activations.
        
        Args:
            layer_idx: Which layer (0-indexed)
            activations: Tensor of shape (batch_size, n_neurons)
        """
        if layer_idx >= len(self.hidden_dims):
            return  # Skip output layer
        
        with torch.no_grad():
            # Convert to numpy for statistics
            acts = activations.detach().cpu().numpy()
            batch_size = acts.shape[0]
            
            # Update per-neuron statistics
            for neuron_idx in range(acts.shape[1]):
                neuron_acts = acts[:, neuron_idx]
                
                # Compute batch statistics
                batch_mean = np.mean(neuron_acts)
                batch_var = np.var(neuron_acts)
                
                # Exponential moving average
                stats = self.neuron_stats[layer_idx][neuron_idx]
                
                if stats['n_updates'] == 0:
                    # First update - just use batch stats
                    stats['mean'] = batch_mean
                    stats['var'] = batch_var
                else:
                    # EMA update
                    stats['mean'] = (
                        self.momentum * stats['mean'] + 
                        (1 - self.momentum) * batch_mean
                    )
                    stats['var'] = (
                        self.momentum * stats['var'] + 
                        (1 - self.momentum) * batch_var
                    )
                
                stats['n_updates'] += 1
            
            # Update layer-wide statistics
            layer_norm = np.linalg.norm(acts) / batch_size
            
            layer_stats = self.layer_stats[layer_idx]
            if layer_stats['n_updates'] == 0:
                layer_stats['output_norm'] = layer_norm
            else:
                layer_stats['output_norm'] = (
                    self.momentum * layer_stats['output_norm'] + 
                    (1 - self.momentum) * layer_norm
                )
            
            layer_stats['n_updates'] += 1
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics for all layers.
        
        Returns dict with:
        - per_layer: {layer_idx: {'mean_activation', 'mean_variance', 'output_norm'}}
        - per_neuron: {layer_idx: {neuron_idx: {'mean', 'var'}}}
        """
        summary = {
            'per_layer': {},
            'per_neuron': {}
        }
        
        for layer_idx in range(len(self.hidden_dims)):
            # Aggregate neuron stats for layer summary
            means = [
                self.neuron_stats[layer_idx][n]['mean']
                for n in range(self.hidden_dims[layer_idx])
            ]
            variances = [
                self.neuron_stats[layer_idx][n]['var']
                for n in range(self.hidden_dims[layer_idx])
            ]
            
            summary['per_layer'][layer_idx] = {
                'mean_activation': np.mean(means),
                'std_activation': np.std(means),
                'mean_variance': np.mean(variances),
                'output_norm': self.layer_stats[layer_idx]['output_norm'],
                'n_neurons': self.hidden_dims[layer_idx]
            }
            
            # Copy neuron-level stats
            summary['per_neuron'][layer_idx] = {
                n: dict(self.neuron_stats[layer_idx][n])
                for n in range(self.hidden_dims[layer_idx])
            }
        
        return summary
    
    def reset(self):
        """Reset all statistics."""
        for layer_idx in range(len(self.hidden_dims)):
            for neuron_idx in range(self.hidden_dims[layer_idx]):
                self.neuron_stats[layer_idx][neuron_idx] = {
                    'mean': 0.0,
                    'var': 0.0,
                    'n_updates': 0
                }
            
            self.layer_stats[layer_idx] = {
                'output_norm': 0.0,
                'n_updates': 0
            }


def print_model_stats(model: InstrumentedMLP):
    """Pretty print current model statistics."""
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    
    # Loss
    smoothed_loss = model.get_smoothed_loss()
    if smoothed_loss is not None:
        print(f"Smoothed Loss: {smoothed_loss:.6f}")
    else:
        print("Smoothed Loss: N/A (no training yet)")
    
    # Layer stats
    stats = model.get_layer_stats()
    print("\nPer-Layer Summary:")
    print("-" * 60)
    
    for layer_idx, layer_stats in stats['per_layer'].items():
        print(f"\nLayer {layer_idx}:")
        print(f"  Neurons: {layer_stats['n_neurons']}")
        print(f"  Mean Activation: {layer_stats['mean_activation']:.4f} "
              f"(±{layer_stats['std_activation']:.4f})")
        print(f"  Mean Variance: {layer_stats['mean_variance']:.4f}")
        print(f"  Output Norm: {layer_stats['output_norm']:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("SSNC PHASE 1: Baseline Instrumented MLP")
    print("="*60)
    
    # Create a small test model
    model = InstrumentedMLP(
        input_dim=784,  # MNIST
        hidden_dims=[128, 64],
        output_dim=10,
        activation="tanh"
    )
    
    print(f"\nModel architecture:")
    print(f"  Input: {model.input_dim}")
    print(f"  Hidden: {model.hidden_dims}")
    print(f"  Output: {model.output_dim}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass with dummy data...")
    dummy_input = torch.randn(32, 784)  # Batch of 32
    output = model(dummy_input)
    print(f"✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
    
    # Check that hooks are working
    print("\nChecking statistics tracking...")
    print_model_stats(model)
    
    # Test freezing
    print("Testing weight freezing...")
    model.freeze_weights()
    model.unfreeze_weights()
    
    print("\n✅ PHASE 1 baseline model ready!")
    print("Next: Train this model to convergence, then move to PHASE 2")
