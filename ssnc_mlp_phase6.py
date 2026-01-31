"""
SSNC PHASE 6: MLP with Soft Deletion

Extends PHASE 4+5 model with Observer execution authority.
Observer can now soft-delete neurons (with recovery mechanism).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

from ssnc_mlp_advanced import SSNC_MLP_Advanced, print_advanced_stats
from soft_deletion import SoftDeletionManager
from observer import SSNCObserver


class SSNC_MLP_Phase6(SSNC_MLP_Advanced):
    """
    SSNC-MLP with soft deletion enabled.
    
    New capability: Observer can gate low-utility neurons.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        # PHASE 4
        enable_local_learning: bool = True,
        local_lr: float = 0.0001,
        # PHASE 5
        enable_observer: bool = True,
        # PHASE 6 - NEW
        enable_soft_deletion: bool = True,
        deletion_utility_threshold: float = 0.05,
        marking_cycles_required: int = 10,
        max_deletions_per_cycle: int = 5
    ):
        """
        Args:
            enable_soft_deletion: Enable PHASE 6 soft deletion
            deletion_utility_threshold: Utility below this gets marked
            marking_cycles_required: Cycles before actual soft deletion
            max_deletions_per_cycle: Budget for new deletions
        """
        # Initialize base (PHASE 4+5)
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            enable_local_learning=enable_local_learning,
            local_lr=local_lr,
            enable_observer=enable_observer,
            observer_readonly=(not enable_soft_deletion)  # Active if deletion enabled
        )
        
        # PHASE 6: Soft deletion manager
        self.enable_soft_deletion = enable_soft_deletion
        self.deletion_manager = None
        
        if enable_soft_deletion:
            self.deletion_manager = SoftDeletionManager(
                n_layers=len(hidden_dims),
                neurons_per_layer=hidden_dims,
                deletion_utility_threshold=deletion_utility_threshold,
                marking_cycles_required=marking_cycles_required,
                max_deletions_per_cycle=max_deletions_per_cycle
            )
            print(f"✓ Soft deletion enabled (threshold={deletion_utility_threshold})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with soft deletion gating.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            # Cache inputs for local learning
            if self.enable_local_learning:
                self.ssnc_layers[i].learning_manager.cache_inputs(x)
            
            x = layer(x)
            x = self.activation(x)
            
            # PHASE 6: Apply soft deletion gating
            if self.enable_soft_deletion and self.deletion_manager is not None:
                x = self.deletion_manager.apply_gating(x, layer_id=i)
            
            # SSNC neurons observe
            self.ssnc_layers[i].observe_forward(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        return x
    
    def observer_cycle(self):
        """
        Enhanced observer cycle with soft deletion execution.
        """
        if not self.enable_observer or self.observer is None:
            return None
        
        self.step_count += 1
        
        if self.step_count % self.observe_every != 0:
            return None
        
        # Get model state
        model_state = self._get_model_state_for_observer()
        
        # Run observation
        report = self.observer.observe(model_state)
        
        # PHASE 6: Execute soft deletion decisions
        if self.enable_soft_deletion and self.deletion_manager is not None:
            deletion_actions = self._execute_deletion_decisions(model_state, report)
            report['deletion_actions'] = deletion_actions
        
        return report
    
    def _execute_deletion_decisions(
        self,
        model_state: Dict,
        observer_report: Dict
    ) -> Dict:
        """
        Execute soft deletion based on observer decisions and current utilities.
        
        Returns:
            Actions taken
        """
        # Collect utilities for all neurons
        utilities = {}
        
        for layer_idx, ssnc_layer in enumerate(self.ssnc_layers):
            for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                utilities[(layer_idx, neuron_idx)] = neuron.utility
        
        # Update deletion manager
        actions = self.deletion_manager.update_cycle(
            utilities=utilities,
            observer_decisions=observer_report['decisions']
        )
        
        return actions
    
    def get_deletion_summary(self) -> Optional[Dict]:
        """Get summary of deletion activity."""
        if not self.enable_soft_deletion or self.deletion_manager is None:
            return None
        
        stats = self.deletion_manager.get_statistics()
        candidates = self.deletion_manager.get_deletion_candidates()
        
        return {
            'statistics': stats,
            'candidates': candidates,
            'n_candidates': len(candidates)
        }


def print_phase6_stats(model: SSNC_MLP_Phase6):
    """Print comprehensive statistics including soft deletion."""
    print("\n" + "="*60)
    print("PHASE 6 SSNC STATISTICS (WITH SOFT DELETION)")
    print("="*60)
    
    # Standard stats
    summary = model.get_ssnc_summary()
    print("\nNeuron Statistics:")
    print(f"  Total neurons: {summary['global']['total_neurons']}")
    print(f"  Mean utility: {summary['global']['mean_utility']:.4f}")
    print(f"  Inactive: {summary['global']['inactive_neurons']}")
    
    # Local learning
    if model.enable_local_learning:
        learning_summary = model.get_learning_summary()
        print("\nLocal Learning:")
        total_updates = sum(
            l.get('total_updates', 0) 
            for l in learning_summary.get('layers', [])
        )
        print(f"  Total updates: {total_updates:,}")
    
    # Observer
    if model.enable_observer:
        observer_summary = model.get_observer_summary()
        if observer_summary:
            print("\nObserver:")
            print(f"  Observations: {observer_summary['observations']}")
            print(f"  Health score: {observer_summary['health_score']:.3f}")
            print(f"  Decisions: {observer_summary['decisions']['total_decisions']}")
    
    # PHASE 6: Soft deletion
    if model.enable_soft_deletion:
        deletion_summary = model.get_deletion_summary()
        if deletion_summary:
            stats = deletion_summary['statistics']
            print("\nSoft Deletion Statistics:")
            print(f"  Deletion cycles: {stats['cycle_count']}")
            print(f"  Currently marked: {stats['marked_for_deletion']}")
            print(f"  Currently soft-deleted: {stats['soft_deleted']}")
            print(f"  Total soft-deleted: {stats['total_soft_deleted']}")
            print(f"  Total recovered: {stats['total_recovered']}")
            print(f"  Recovery rate: {stats['recovery_rate']:.1%}")
            print(f"  Avg gate value: {stats['avg_gate_value']:.3f}")
            
            if deletion_summary['n_candidates'] > 0:
                print(f"\n  Top Deletion Candidates:")
                for i, cand in enumerate(deletion_summary['candidates'][:5]):
                    status = "GATED" if cand['soft_deleted'] else "MARKED"
                    print(f"    {status}: Layer {cand['layer']}, "
                          f"Neuron {cand['neuron']}, "
                          f"Gate={cand['gate_value']:.2f}, "
                          f"Utility={cand['last_utility']:.3f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print("SSNC PHASE 6: MLP with Soft Deletion")
    print("="*60)
    print()
    
    # Create Phase 6 model
    print("Creating Phase 6 model with soft deletion...")
    model = SSNC_MLP_Phase6(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        enable_local_learning=True,
        local_lr=0.0001,
        enable_observer=True,
        enable_soft_deletion=True,
        deletion_utility_threshold=0.05,
        marking_cycles_required=5  # Short for testing
    )
    
    print(f"✓ Model created")
    print(f"  Local learning: {model.enable_local_learning}")
    print(f"  Observer: {model.enable_observer}")
    print(f"  Soft deletion: {model.enable_soft_deletion}")
    print(f"  Observer readonly: {model.observer.readonly}")
    print()
    
    # Simulate training steps
    print("Simulating training with soft deletion...")
    print("-" * 60)
    
    for step in range(50):
        # Forward pass
        dummy_input = torch.randn(32, 784)
        output = model(dummy_input)
        
        # Simulate backward
        dummy_target = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)
        loss.backward()
        
        # SSNC observation
        model.observe_gradients()
        model.compute_all_utilities()
        
        # Observer cycle (includes soft deletion)
        if step % 10 == 0:
            report = model.observer_cycle()
            if report and 'deletion_actions' in report:
                actions = report['deletion_actions']
                
                if actions['newly_gated']:
                    print(f"\nStep {step}: Soft-deleted {len(actions['newly_gated'])} neurons")
                    for item in actions['newly_gated']:
                        print(f"  Layer {item['layer']}, Neuron {item['neuron']}: "
                              f"utility={item['utility']:.3f}")
                
                if actions['recovered']:
                    print(f"\nStep {step}: Recovered {len(actions['recovered'])} neurons")
    
    # Final statistics
    print("\n" + "="*60)
    print("Final Statistics:")
    print_phase6_stats(model)
    
    print("\n✅ PHASE 6 integration complete!")
    print("Next: Train on MNIST with soft deletion enabled")
