"""
SSNC PHASE 4 & 5: Integrated MLP with Local Learning and Observer

Combines:
- PHASE 2: Neuron observation
- PHASE 3: Utility validation
- PHASE 4: Local learning
- PHASE 5: Observer monitoring

This is the first "complete" SSNC system with all components working together.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from ssnc_mlp import SSNC_MLP
from archives.local_learning import SSNCNeuronWithLearning, LocalLearningManager
from observer import SSNCObserver, StructuralDecision


class SSNC_MLP_Advanced(SSNC_MLP):
    """
    Advanced SSNC-MLP with local learning and observer.
    
    New capabilities:
    - Neurons learn locally (utility-modulated)
    - Observer monitors and makes decisions
    - Hybrid learning (SGD + local updates)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        # PHASE 4 parameters
        enable_local_learning: bool = False,
        local_lr: float = 0.0001,
        # PHASE 5 parameters
        enable_observer: bool = False,
        observer_readonly: bool = True
    ):
        """
        Args:
            enable_local_learning: Turn on PHASE 4 local learning
            local_lr: Base learning rate for local updates
            enable_observer: Turn on PHASE 5 observer
            observer_readonly: If True, observer only logs decisions
        """
        # Initialize base SSNC-MLP
        super().__init__(input_dim, hidden_dims, output_dim, activation)
        
        # PHASE 4: Replace neurons with learning-capable versions
        self.enable_local_learning = enable_local_learning
        
        if enable_local_learning:
            self._upgrade_neurons_to_learning(local_lr)
        
        # PHASE 5: Create observer
        self.enable_observer = enable_observer
        self.observer = None
        
        if enable_observer:
            self.observer = SSNCObserver(readonly=observer_readonly)
        
        # Observation frequency
        self.observe_every = 10  # Observer checks every N steps
        self.step_count = 0
        
        # Cache for activations during forward pass (used with local learning)
        self.cached_activations = {}
    
    def _upgrade_neurons_to_learning(self, local_lr: float):
        """Replace standard neurons with learning-capable neurons."""
        print(f"🔄 Upgrading neurons to learning-capable (local_lr={local_lr})...")
        
        new_ssnc_layers = []
        
        for layer_idx, ssnc_layer in enumerate(self.ssnc_layers):
            # Create new learning neurons
            learning_neurons = [
                SSNCNeuronWithLearning(
                    neuron_id=neuron.neuron_id,
                    layer_id=neuron.layer_id,
                    activation_window=neuron.activation_history.maxlen,
                    error_window=neuron.error_buffer.maxlen,
                    momentum=neuron.momentum,
                    local_lr=local_lr,
                    utility_modulation=True
                )
                for neuron in ssnc_layer.neurons
            ]
            
            # Copy existing state
            for old_neuron, new_neuron in zip(ssnc_layer.neurons, learning_neurons):
                new_neuron.activation_mean = old_neuron.activation_mean
                new_neuron.activation_var = old_neuron.activation_var
                new_neuron.utility = old_neuron.utility
                new_neuron.age = old_neuron.age
            
            # Replace neurons
            ssnc_layer.neurons = learning_neurons
            
            # Create learning manager
            ssnc_layer.learning_manager = LocalLearningManager(
                neurons=learning_neurons,
                enable_learning=self.enable_local_learning
            )
            
            new_ssnc_layers.append(ssnc_layer)
        
        self.ssnc_layers = nn.ModuleList(new_ssnc_layers)
        print(f"✓ {sum(len(layer.neurons) for layer in self.ssnc_layers)} neurons upgraded")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with local learning capability.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Clear activation cache at start of forward pass
        if self.enable_local_learning:
            self.cached_activations.clear()
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            # Cache inputs for local learning
            if self.enable_local_learning:
                self.ssnc_layers[i].learning_manager.cache_inputs(x)
            
            x = layer(x)
            x = self.activation(x)
            
            # Cache activations for this layer (needed for local learning updates)
            if self.enable_local_learning:
                self.cached_activations[i] = x.detach().clone()
            
            # SSNC neurons observe
            self.ssnc_layers[i].observe_forward(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        return x
    
    def observe_gradients(self):
        """
        Enhanced gradient observation with local learning.
        """
        # Base observation (from PHASE 2)
        for layer_idx, ssnc_layer in enumerate(self.ssnc_layers):
            if layer_idx in self.cached_gradients:
                gradients = self.cached_gradients[layer_idx]
                ssnc_layer.observe_backward(gradients)
                
                # PHASE 4: Apply local learning if enabled
                if self.enable_local_learning and hasattr(ssnc_layer, 'learning_manager'):
                    # Get cached activations and apply local updates
                    if layer_idx in self.cached_activations:
                        activations = self.cached_activations[layer_idx]
                        ssnc_layer.learning_manager.apply_local_updates(
                            layer=self.layers[layer_idx],
                            activations=activations,
                            gradients=gradients
                        )
        
        self.cached_gradients.clear()
    
    def apply_local_learning(self):
        """
        Apply local learning updates to all layers.
        
        Note: Local learning is now applied directly in observe_gradients,
        so this method is kept for backward compatibility.
        """
        pass
    
    def observer_cycle(self):
        """
        Run observer monitoring cycle.
        
        Call this periodically during training.
        """
        if not self.enable_observer or self.observer is None:
            return None
        
        self.step_count += 1
        
        if self.step_count % self.observe_every != 0:
            return None
        
        # Get complete model state
        model_state = self._get_model_state_for_observer()
        
        # Run observation
        report = self.observer.observe(model_state)
        
        return report
    
    def _get_model_state_for_observer(self) -> Dict:
        """Package model state for observer."""
        # Get SSNC summary
        ssnc_summary = self.get_ssnc_summary()
        
        # Add learning statistics if available
        if self.enable_local_learning:
            learning_stats = []
            for layer in self.ssnc_layers:
                if hasattr(layer, 'learning_manager'):
                    learning_stats.append(layer.learning_manager.get_layer_learning_stats())
            ssnc_summary['learning_stats'] = learning_stats
        
        return ssnc_summary
    
    def get_learning_summary(self) -> Dict:
        """Get summary of local learning activity."""
        if not self.enable_local_learning:
            return {'enabled': False}
        
        summary = {
            'enabled': True,
            'layers': []
        }
        
        for layer_idx, ssnc_layer in enumerate(self.ssnc_layers):
            if hasattr(ssnc_layer, 'learning_manager'):
                layer_stats = ssnc_layer.learning_manager.get_layer_learning_stats()
                layer_stats['layer_id'] = layer_idx
                summary['layers'].append(layer_stats)
        
        return summary
    
    def get_observer_summary(self) -> Optional[Dict]:
        """Get summary of observer activity."""
        if not self.enable_observer or self.observer is None:
            return None
        
        return {
            'observations': self.observer.observation_count,
            'health_score': self.observer.health_score,
            'health_trend': self.observer.get_health_trend(),
            'decisions': self.observer.get_decision_summary(),
            'should_intervene': self.observer.should_intervene()
        }


def print_advanced_stats(model: SSNC_MLP_Advanced):
    """Print comprehensive statistics including learning and observer."""
    print("\n" + "="*60)
    print("ADVANCED SSNC STATISTICS")
    print("="*60)
    
    # Standard SSNC stats
    summary = model.get_ssnc_summary()
    print("\nNeuron Statistics:")
    print(f"  Total neurons: {summary['global']['total_neurons']}")
    print(f"  Mean utility: {summary['global']['mean_utility']:.4f}")
    print(f"  Inactive: {summary['global']['inactive_neurons']}")
    
    # Local learning stats
    if model.enable_local_learning:
        learning_summary = model.get_learning_summary()
        print("\nLocal Learning Statistics:")
        
        for layer_stats in learning_summary['layers']:
            print(f"  Layer {layer_stats['layer_id']}:")
            print(f"    Updates applied: {layer_stats.get('total_updates', 0)}")
            print(f"    Mean update norm: {layer_stats.get('mean_update_norm', 0):.6f}")
            print(f"    Neurons learning: {layer_stats.get('neurons_learning', 0)}")
    
    # Observer stats
    if model.enable_observer:
        observer_summary = model.get_observer_summary()
        if observer_summary:
            print("\nObserver Statistics:")
            print(f"  Observations: {observer_summary['observations']}")
            print(f"  Health score: {observer_summary['health_score']:.3f}")
            print(f"  Health trend: {observer_summary['health_trend']['trend']}")
            print(f"  Total decisions: {observer_summary['decisions']['total_decisions']}")
            
            should_act, reason = observer_summary['should_intervene']
            print(f"  Should intervene: {should_act} ({reason})")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("SSNC PHASE 4 & 5: Advanced MLP Integration")
    print("="*60)
    print()
    
    # Create models with different configurations
    
    # Config 1: Only local learning
    print("Configuration 1: Local learning only")
    model_learning = SSNC_MLP_Advanced(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        enable_local_learning=True,
        local_lr=0.0001,
        enable_observer=False
    )
    print(f"✓ Model created with local learning")
    print()
    
    # Config 2: Only observer
    print("Configuration 2: Observer only")
    model_observer = SSNC_MLP_Advanced(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        enable_local_learning=False,
        enable_observer=True,
        observer_readonly=True
    )
    print(f"✓ Model created with observer (read-only)")
    print()
    
    # Config 3: Both (full SSNC)
    print("Configuration 3: Full SSNC (learning + observer)")
    model_full = SSNC_MLP_Advanced(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        enable_local_learning=True,
        local_lr=0.0001,
        enable_observer=True,
        observer_readonly=True
    )
    print(f"✓ Full SSNC model created")
    print()
    
    # Test full model
    print("Testing full SSNC model:")
    print("-" * 60)
    
    # Forward pass
    dummy_input = torch.randn(32, 784)
    output = model_full(dummy_input)
    print(f"✓ Forward pass: {dummy_input.shape} → {output.shape}")
    
    # Backward pass (simulate)
    dummy_target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)
    loss.backward()
    
    # Observe gradients
    model_full.observe_gradients()
    print(f"✓ Gradient observation complete")
    
    # Compute utilities
    model_full.compute_all_utilities()
    print(f"✓ Utilities computed")
    
    # Run observer cycle
    report = model_full.observer_cycle()
    if report:
        print(f"✓ Observer cycle complete")
        print(f"  Health: {report['health']['overall_score']:.3f}")
        print(f"  Issues: {len(report['issues'])}")
        print(f"  Decisions: {len(report['decisions'])}")
    
    # Print stats
    print()
    print_advanced_stats(model_full)
    
    print("✅ PHASE 4 & 5 integration complete!")
    print()
    print("Available configurations:")
    print("  1. Standard (PHASE 2): Observation only")
    print("  2. + Local Learning (PHASE 4): Neurons adapt locally")
    print("  3. + Observer (PHASE 5): Global monitoring and decisions")
    print("  4. Full SSNC: All components working together")
    print()
    print("Next: Train with hybrid learning (SGD + local) and observer monitoring")
