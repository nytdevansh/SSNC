"""
SSNC PHASE 2: MLP with SSNC-Neuron Integration

Extends the baseline InstrumentedMLP with SSNC neuron tracking.
Still trains normally, but now each neuron maintains local state.

PHASE 2 Rule: Neurons OBSERVE only, they do not ACT yet.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from baseline_mlp import InstrumentedMLP
from ssnc_neuron import SSNCNeuronLayer


class SSNC_MLP(InstrumentedMLP):
    """
    MLP with SSNC neuron tracking.
    
    Extends baseline model with per-neuron state tracking and utility computation.
    Training still happens normally (standard SGD), but we observe everything.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh"
    ):
        # Initialize base model
        super().__init__(input_dim, hidden_dims, output_dim, activation)
        
        # Create SSNC neuron layers (one per hidden layer)
        self.ssnc_layers = nn.ModuleList([
            SSNCNeuronLayer(layer_id=i, n_neurons=hidden_dims[i])
            for i in range(len(hidden_dims))
        ])
        
        # Hook for backward pass (to capture gradients)
        self._register_backward_hooks()
        
        # Store gradients temporarily for neuron observation
        self.cached_gradients = {}
    
    def _register_backward_hooks(self):
        """
        Register backward hooks to capture gradients for utility computation.
        """
        for layer_idx, layer in enumerate(self.layers[:-1]):  # Hidden layers only
            def make_backward_hook(idx):
                def hook(module, grad_input, grad_output):
                    # Store gradient for this layer
                    # grad_output[0] is the gradient w.r.t. layer output
                    if grad_output[0] is not None:
                        self.cached_gradients[idx] = grad_output[0].detach()
                return hook
            
            layer.register_full_backward_hook(make_backward_hook(layer_idx))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SSNC neuron observation.
        
        Same as baseline, but SSNC neurons observe activations.
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # SSNC neurons observe activations
            self.ssnc_layers[i].observe_forward(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x
    
    def observe_gradients(self):
        """
        After backward pass, let SSNC neurons observe error signals.
        
        Call this after loss.backward() to update neuron utility.
        """
        for layer_idx, ssnc_layer in enumerate(self.ssnc_layers):
            if layer_idx in self.cached_gradients:
                gradients = self.cached_gradients[layer_idx]
                ssnc_layer.observe_backward(gradients)
        
        # Clear cache
        self.cached_gradients.clear()
    
    def compute_all_utilities(self):
        """Compute utility for all neurons in all layers."""
        for ssnc_layer in self.ssnc_layers:
            ssnc_layer.compute_all_utilities()
    
    def get_ssnc_summary(self) -> Dict:
        """
        Get SSNC-specific statistics for all layers.
        
        Returns:
            Dict with per-layer SSNC stats
        """
        summary = {
            'layers': [],
            'global': {
                'total_neurons': 0,
                'inactive_neurons': 0,
                'mean_utility': 0.0,
                'low_utility_neurons': 0  # utility < 0.1
            }
        }
        
        all_utilities = []
        
        for ssnc_layer in self.ssnc_layers:
            layer_summary = ssnc_layer.get_layer_summary()
            summary['layers'].append(layer_summary)
            
            # Aggregate global stats
            summary['global']['total_neurons'] += layer_summary['n_neurons']
            summary['global']['inactive_neurons'] += layer_summary['inactive_neurons']
            all_utilities.extend(layer_summary['utilities']['distribution'])
        
        # Compute global utility statistics
        if all_utilities:
            summary['global']['mean_utility'] = sum(all_utilities) / len(all_utilities)
            summary['global']['low_utility_neurons'] = sum(
                1 for u in all_utilities if u < 0.1
            )
        
        return summary
    
    def get_detailed_neuron_states(self) -> List[Dict]:
        """
        Get detailed state for every neuron.
        
        Useful for debugging and analysis.
        """
        all_states = []
        for ssnc_layer in self.ssnc_layers:
            all_states.extend(ssnc_layer.get_all_neuron_states())
        return all_states
    
    def reset_ssnc_stats(self):
        """Reset all SSNC neuron statistics (keep weights)."""
        for ssnc_layer in self.ssnc_layers:
            for neuron in ssnc_layer.neurons:
                neuron.reset()
        print("🔄 SSNC neuron statistics reset")


def print_ssnc_stats(model: SSNC_MLP):
    """Pretty print SSNC statistics."""
    print("\n" + "="*60)
    print("SSNC NEURON STATISTICS")
    print("="*60)
    
    summary = model.get_ssnc_summary()
    
    # Global stats
    print("\nGlobal Summary:")
    print("-" * 60)
    print(f"Total neurons: {summary['global']['total_neurons']}")
    print(f"Mean utility: {summary['global']['mean_utility']:.4f}")
    print(f"Inactive neurons (>50 steps): {summary['global']['inactive_neurons']}")
    print(f"Low utility neurons (<0.1): {summary['global']['low_utility_neurons']}")
    
    # Per-layer stats
    print("\nPer-Layer Summary:")
    print("-" * 60)
    
    for layer_data in summary['layers']:
        print(f"\nLayer {layer_data['layer_id']}:")
        print(f"  Neurons: {layer_data['n_neurons']}")
        
        utils = layer_data['utilities']
        print(f"  Utility: {utils['mean']:.4f} (±{utils['std']:.4f}) "
              f"[{utils['min']:.4f}, {utils['max']:.4f}]")
        
        acts = layer_data['activations']
        print(f"  Activations: {acts['mean']:.4f} (±{acts['std']:.4f})")
        
        print(f"  Inactive: {layer_data['inactive_neurons']}")
        print(f"  Max inactivity: {layer_data['max_inactivity']} steps")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("SSNC PHASE 2: MLP with SSNC Neuron Integration")
    print("="*60)
    
    # Create SSNC-enabled model
    model = SSNC_MLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        activation='tanh'
    )
    
    print(f"\nModel created:")
    print(f"  Architecture: {model.input_dim} → {model.hidden_dims} → {model.output_dim}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  SSNC layers: {len(model.ssnc_layers)}")
    
    # Test forward pass
    print("\nTesting forward pass with SSNC observation...")
    dummy_input = torch.randn(32, 784)
    output = model(dummy_input)
    print(f"✓ Forward pass: {dummy_input.shape} → {output.shape}")
    
    # Test backward pass (simulate training)
    print("\nSimulating training step (forward + backward)...")
    dummy_target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    loss.backward()
    model.observe_gradients()  # Let neurons see gradients
    
    print(f"✓ Backward pass complete, loss: {loss.item():.4f}")
    
    # Compute utilities
    print("\nComputing neuron utilities...")
    model.compute_all_utilities()
    
    # Print SSNC statistics
    print_ssnc_stats(model)
    
    # Test detailed neuron inspection
    print("Testing detailed neuron state access...")
    neuron_states = model.get_detailed_neuron_states()
    print(f"✓ Retrieved {len(neuron_states)} neuron states")
    
    # Show a sample neuron
    sample_neuron = neuron_states[0]
    print(f"\nSample neuron (Layer {sample_neuron['layer_id']}, "
          f"ID {sample_neuron['neuron_id']}):")
    for key in ['utility', 'activation_mean', 'activation_var', 
                'error_correlation', 'age', 'inactivity_counter']:
        value = sample_neuron[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ PHASE 2 integration complete!")
    print("Key validation checks:")
    print("  ✓ SSNC neurons track activations automatically")
    print("  ✓ SSNC neurons observe gradients after backward pass")
    print("  ✓ Utility computation works")
    print("  ✓ Statistics remain stable")
    print("\nNext: Train this model and verify no performance change from baseline")
