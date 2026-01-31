"""
SSNC PHASE 6: Soft Deletion

Observer can now ACT - but safely, with soft deletion (gating).

Soft deletion = neuron output is gated (multiplied by small value)
- Neuron still exists (weights preserved)
- Can recover if utility improves
- Gradual, reversible process

CRITICAL SAFETY:
- Long hysteresis (many cycles before deletion)
- Slow gating (gradual reduction, not instant)
- Recovery mechanism (ungating if utility improves)
- Budget limit (max deletions per cycle)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DeletionState:
    """Tracks deletion state for a neuron."""
    neuron_id: int
    layer_id: int
    
    # Deletion tracking
    marked_for_deletion: bool = False
    marked_cycles: int = 0  # How long marked
    soft_deleted: bool = False
    gate_value: float = 1.0  # 1.0 = active, 0.0 = fully gated
    
    # Recovery tracking
    recovery_attempts: int = 0
    last_utility: float = 0.0
    
    # Age tracking
    age: int = 0  # Total cycles since neuron creation
    
    # History
    deletion_timestamp: Optional[int] = None
    recovery_timestamp: Optional[int] = None


class SoftDeletionManager:
    """
    Manages soft deletion of neurons.
    
    PHASE 6: Soft deletion only - neurons are gated, not removed.
    PHASE 7: Will add hard deletion (actual removal).
    """
    
    def __init__(
        self,
        n_layers: int,
        neurons_per_layer: List[int],
        # Deletion thresholds
        deletion_utility_threshold: float = 0.15,
        marking_cycles_required: int = 5,  # Must be low utility for 10 cycles
        # Gating parameters
        gating_rate: float = 0.1,  # How fast to gate (per cycle)
        min_gate_value: float = 0.01,  # Don't fully gate (keep some signal)
        # Recovery parameters
        recovery_utility_threshold: float = 0.15,  # Utility to trigger recovery
        ungating_rate: float = 0.2,  # How fast to ungate (faster than gating)
        # Budget
        max_deletions_per_cycle: int = 10
    ):
        """
        Args:
            marking_cycles_required: Neuron must be low-utility for this many cycles
            gating_rate: How much to reduce gate_value per cycle (0.1 = 10% reduction)
            min_gate_value: Minimum gate value (don't fully silence)
            recovery_utility_threshold: Utility needed to trigger recovery
            ungating_rate: How fast to restore gate_value during recovery
            max_deletions_per_cycle: Max new soft deletions per cycle
        """
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        
        # Thresholds
        self.deletion_utility_threshold = deletion_utility_threshold
        self.marking_cycles_required = marking_cycles_required
        self.gating_rate = gating_rate
        self.min_gate_value = min_gate_value
        self.recovery_utility_threshold = recovery_utility_threshold
        self.ungating_rate = ungating_rate
        self.max_deletions_per_cycle = max_deletions_per_cycle
        
        # Track deletion state for all neurons
        self.deletion_states: Dict[tuple, DeletionState] = {}
        
        for layer_id in range(n_layers):
            for neuron_id in range(neurons_per_layer[layer_id]):
                key = (layer_id, neuron_id)
                self.deletion_states[key] = DeletionState(
                    neuron_id=neuron_id,
                    layer_id=layer_id
                )
        
        # Statistics
        self.cycle_count = 0
        self.total_soft_deleted = 0
        self.total_recovered = 0
        self.deletion_history = []
    
    def update_cycle(
        self,
        utilities: Dict[tuple, float],
        observer_decisions: List
    ) -> Dict:
        """
        Main update cycle for soft deletion.
        
        Args:
            utilities: {(layer_id, neuron_id): utility}
            observer_decisions: Decisions from observer
            
        Returns:
            Report of actions taken
        """
        self.cycle_count += 1
        
        actions = {
            'newly_marked': [],
            'newly_gated': [],
            'gating_progress': [],
            'recovered': [],
            'cycle': self.cycle_count
        }
        
        # Step 1: Update marking based on utilities
        for (layer_id, neuron_id), utility in utilities.items():
            state = self.deletion_states[(layer_id, neuron_id)]
            state.last_utility = utility
            
            # Check if should be marked
            if utility < self.deletion_utility_threshold:
                if not state.marked_for_deletion:
                    state.marked_for_deletion = True
                    state.marked_cycles = 1
                    actions['newly_marked'].append((layer_id, neuron_id, utility))
                else:
                    state.marked_cycles += 1
            else:
                # Utility recovered - unmark
                if state.marked_for_deletion and not state.soft_deleted:
                    state.marked_for_deletion = False
                    state.marked_cycles = 0
        
        # Step 2: Apply soft deletion (with budget)
        newly_deleted_count = 0
        
        for key, state in self.deletion_states.items():
            layer_id, neuron_id = key
            
            # Start soft deletion if marked long enough
            if (state.marked_for_deletion and 
                state.marked_cycles >= self.marking_cycles_required and
                not state.soft_deleted and
                newly_deleted_count < self.max_deletions_per_cycle):
                
                state.soft_deleted = True
                state.deletion_timestamp = self.cycle_count
                self.total_soft_deleted += 1
                newly_deleted_count += 1
                
                actions['newly_gated'].append({
                    'layer': layer_id,
                    'neuron': neuron_id,
                    'utility': state.last_utility,
                    'marked_cycles': state.marked_cycles
                })
            
            # Continue gating if already soft deleted
            if state.soft_deleted:
                if state.gate_value > self.min_gate_value:
                    # Gradual gating
                    state.gate_value *= (1.0 - self.gating_rate)
                    state.gate_value = max(state.gate_value, self.min_gate_value)
                    
                    actions['gating_progress'].append({
                        'layer': layer_id,
                        'neuron': neuron_id,
                        'gate_value': state.gate_value
                    })
        
        # Step 3: Recovery mechanism
        for key, state in self.deletion_states.items():
            layer_id, neuron_id = key
            
            if state.soft_deleted:
                # Check if utility has recovered
                if state.last_utility > self.recovery_utility_threshold:
                    # Ungate gradually
                    if state.gate_value < 1.0:
                        state.gate_value = min(
                            state.gate_value * (1.0 + self.ungating_rate),
                            1.0
                        )
                        
                        # If fully recovered, mark as recovered
                        if state.gate_value >= 0.99:
                            state.soft_deleted = False
                            state.marked_for_deletion = False
                            state.marked_cycles = 0
                            state.recovery_timestamp = self.cycle_count
                            state.recovery_attempts += 1
                            self.total_recovered += 1
                            
                            actions['recovered'].append({
                                'layer': layer_id,
                                'neuron': neuron_id,
                                'utility': state.last_utility,
                                'cycles_deleted': self.cycle_count - state.deletion_timestamp
                            })
        
        # Record history
        self.deletion_history.append(actions)
        
        return actions
    
    def get_gate_values(self, layer_id: int) -> torch.Tensor:
        """
        Get gate values for a layer as a tensor.
        
        Returns:
            Tensor of shape (n_neurons,) with gate values
        """
        n_neurons = self.neurons_per_layer[layer_id]
        gate_values = torch.ones(n_neurons)
        
        for neuron_id in range(n_neurons):
            state = self.deletion_states[(layer_id, neuron_id)]
            gate_values[neuron_id] = state.gate_value
        
        return gate_values
    
    def apply_gating(
        self,
        activations: torch.Tensor,
        layer_id: int
    ) -> torch.Tensor:
        """
        Apply gating to layer activations.
        
        Args:
            activations: Tensor of shape (batch_size, n_neurons)
            layer_id: Which layer
            
        Returns:
            Gated activations
        """
        gate_values = self.get_gate_values(layer_id).to(activations.device)
        
        # Broadcast multiply: (batch_size, n_neurons) * (n_neurons,)
        return activations * gate_values.unsqueeze(0)
    
    def get_statistics(self) -> Dict:
        """Get deletion statistics."""
        # Count current states
        marked_count = sum(
            1 for state in self.deletion_states.values()
            if state.marked_for_deletion
        )
        soft_deleted_count = sum(
            1 for state in self.deletion_states.values()
            if state.soft_deleted
        )
        
        # Average gate value for soft-deleted neurons
        deleted_neurons = [
            state for state in self.deletion_states.values()
            if state.soft_deleted
        ]
        avg_gate_value = (
            np.mean([s.gate_value for s in deleted_neurons])
            if deleted_neurons else 1.0
        )
        
        return {
            'cycle_count': self.cycle_count,
            'marked_for_deletion': marked_count,
            'soft_deleted': soft_deleted_count,
            'total_soft_deleted': self.total_soft_deleted,
            'total_recovered': self.total_recovered,
            'avg_gate_value': avg_gate_value,
            'recovery_rate': (
                self.total_recovered / max(self.total_soft_deleted, 1)
            )
        }
    
    def get_deletion_candidates(self) -> List[Dict]:
        """Get list of neurons marked for deletion."""
        candidates = []
        
        for key, state in self.deletion_states.items():
            if state.marked_for_deletion or state.soft_deleted:
                candidates.append({
                    'layer': state.layer_id,
                    'neuron': state.neuron_id,
                    'marked': state.marked_for_deletion,
                    'soft_deleted': state.soft_deleted,
                    'marked_cycles': state.marked_cycles,
                    'gate_value': state.gate_value,
                    'last_utility': state.last_utility
                })
        
        return candidates
    
    def reset_neuron(self, layer_id: int, neuron_id: int):
        """Manually reset a neuron's deletion state."""
        state = self.deletion_states[(layer_id, neuron_id)]
        state.marked_for_deletion = False
        state.marked_cycles = 0
        state.soft_deleted = False
        state.gate_value = 1.0


if __name__ == "__main__":
    print("SSNC PHASE 6: Soft Deletion Manager")
    print("="*60)
    
    # Create manager
    manager = SoftDeletionManager(
        n_layers=2,
        neurons_per_layer=[256, 128],
        deletion_utility_threshold=0.05,
        marking_cycles_required=5,  # Short for demo
        gating_rate=0.2,  # Fast for demo
        max_deletions_per_cycle=3
    )
    
    print(f"Manager created:")
    print(f"  Layers: {manager.n_layers}")
    print(f"  Total neurons: {sum(manager.neurons_per_layer)}")
    print(f"  Deletion threshold: {manager.deletion_utility_threshold}")
    print(f"  Marking cycles required: {manager.marking_cycles_required}")
    print()
    
    # Simulate utilities over time
    print("Simulating deletion cycle:")
    print("-" * 60)
    
    for cycle in range(10):
        # Create mock utilities
        utilities = {}
        
        for layer_id in range(2):
            for neuron_id in range(manager.neurons_per_layer[layer_id]):
                # Most neurons have normal utility
                utility = 0.15 + 0.1 * np.random.randn()
                
                # A few have consistently low utility
                if neuron_id < 5:
                    utility = 0.02 + 0.01 * np.random.randn()
                
                utilities[(layer_id, neuron_id)] = max(0.0, utility)
        
        # Update cycle
        actions = manager.update_cycle(utilities, [])
        
        # Print summary
        print(f"\nCycle {cycle + 1}:")
        if actions['newly_marked']:
            print(f"  Newly marked: {len(actions['newly_marked'])}")
        if actions['newly_gated']:
            print(f"  Newly gated: {len(actions['newly_gated'])}")
            for item in actions['newly_gated']:
                print(f"    Layer {item['layer']}, Neuron {item['neuron']}: "
                      f"utility={item['utility']:.3f}")
        if actions['gating_progress']:
            print(f"  Gating progress: {len(actions['gating_progress'])} neurons")
        if actions['recovered']:
            print(f"  Recovered: {len(actions['recovered'])}")
    
    # Final statistics
    print("\n" + "="*60)
    stats = manager.get_statistics()
    print("Final Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Show candidates
    print("\nDeletion Candidates:")
    candidates = manager.get_deletion_candidates()
    print(f"  Total: {len(candidates)}")
    for i, cand in enumerate(candidates[:5]):  # Show first 5
        print(f"  [{i+1}] Layer {cand['layer']}, Neuron {cand['neuron']}: "
              f"gate={cand['gate_value']:.3f}, utility={cand['last_utility']:.3f}")
    
    print("\n✅ PHASE 6 soft deletion implementation complete!")
    print("Next: Integrate with SSNC-MLP and train with deletion enabled")
