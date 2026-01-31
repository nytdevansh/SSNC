"""
ULTRA-DEBUGGED SSNC - Shows exactly what's happening

This version prints EVERYTHING so we can see where the process breaks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class DebugSSNCNeuron:
    """Ultra-simple neuron with verbose debugging."""
    
    def __init__(self, neuron_id: int, layer_id: int):
        self.neuron_id = neuron_id
        self.layer_id = layer_id
        self.age = 0
        self.utility = 0.0
        self.activation_history = deque(maxlen=50)
        self.error_history = deque(maxlen=50)
    
    def observe_activation(self, value: float):
        self.activation_history.append(value)
        self.age += 1
    
    def observe_error(self, value: float):
        self.error_history.append(value)
    
    def compute_utility(self) -> float:
        if len(self.activation_history) < 10:
            self.utility = 0.0
            return 0.0
        
        acts = np.array(list(self.activation_history))
        magnitude = np.abs(np.mean(acts))
        variance = np.std(acts)
        
        if len(self.error_history) >= 10:
            errs = np.array(list(self.error_history))
            if np.std(acts) > 1e-6 and np.std(errs) > 1e-6:
                correlation = abs(np.corrcoef(acts, errs)[0, 1])
                if not np.isnan(correlation):
                    self.utility = magnitude * variance * correlation
                    return self.utility
        
        self.utility = magnitude * variance
        return self.utility


class DebugSSNC_MLP(nn.Module):
    """SSNC with extensive debugging output."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        deletion_threshold: float = 0.10,
        marking_cycles: int = 5,
        grace_period: int = 20  # SHORTER for testing
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.deletion_threshold = deletion_threshold
        self.marking_cycles = marking_cycles
        self.grace_period = grace_period
        
        # Build network
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.activation = nn.ReLU()
        
        # SSNC neurons - DIRECT STORAGE
        self.ssnc_neurons = []
        for layer_idx, n_neurons in enumerate(hidden_dims):
            layer_neurons = [
                DebugSSNCNeuron(neuron_id=i, layer_id=layer_idx)
                for i in range(n_neurons)
            ]
            self.ssnc_neurons.append(layer_neurons)
        
        # Deletion state
        self.neuron_states = {}
        for layer_idx, n_neurons in enumerate(hidden_dims):
            for neuron_idx in range(n_neurons):
                self.neuron_states[(layer_idx, neuron_idx)] = {
                    'marked': False,
                    'marked_cycles': 0,
                    'soft_deleted': False,
                    'gate_value': 1.0
                }
        
        self.cycle_count = 0
        self.total_soft_deleted = 0
        self.cached_gradients = {}
        
        # Hooks
        for layer_idx, layer in enumerate(self.layers[:-1]):
            def make_hook(idx):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self.cached_gradients[idx] = grad_output[0].detach()
                return hook
            layer.register_full_backward_hook(make_hook(layer_idx))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer_idx, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Observe
            batch_mean = x.mean(dim=0).detach().cpu().numpy()
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                neuron.observe_activation(float(batch_mean[neuron_idx]))
            
            # Apply gates
            gates = torch.ones(x.shape[1], device=x.device)
            for neuron_idx in range(x.shape[1]):
                key = (layer_idx, neuron_idx)
                if key in self.neuron_states:
                    gates[neuron_idx] = self.neuron_states[key]['gate_value']
            x = x * gates.unsqueeze(0)
        
        x = self.layers[-1](x)
        return x
    
    def observe_gradients(self):
        for layer_idx in self.cached_gradients:
            if layer_idx >= len(self.ssnc_neurons):
                continue
            
            grads = self.cached_gradients[layer_idx]
            batch_mean = grads.mean(dim=0).detach().cpu().numpy()
            
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                neuron.observe_error(float(batch_mean[neuron_idx]))
        
        self.cached_gradients.clear()
    
    def compute_all_utilities(self):
        for layer_neurons in self.ssnc_neurons:
            for neuron in layer_neurons:
                neuron.compute_utility()
    
    def plasticity_cycle(self, verbose: bool = False) -> Dict:
        """Plasticity cycle with detailed debugging."""
        self.cycle_count += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PLASTICITY CYCLE {self.cycle_count}")
            print('='*60)
        
        actions = {'soft_marked': [], 'soft_gated': []}
        
        # Collect utilities
        utilities = {}
        for layer_idx, layer_neurons in enumerate(self.ssnc_neurons):
            for neuron in layer_neurons:
                key = (layer_idx, neuron.neuron_id)
                utilities[key] = neuron.utility
        
        if verbose:
            print(f"\nUtilities collected: {len(utilities)}")
            util_values = list(utilities.values())
            print(f"  Mean: {np.mean(util_values):.6f}")
            print(f"  Std: {np.std(util_values):.6f}")
            print(f"  Range: [{np.min(util_values):.6f}, {np.max(util_values):.6f}]")
            below_thresh = sum(1 for u in util_values if u < self.deletion_threshold)
            print(f"  Below threshold ({self.deletion_threshold}): {below_thresh}")
        
        # Check ages
        if verbose:
            ages = [n.age for layer in self.ssnc_neurons for n in layer]
            print(f"\nNeuron ages:")
            print(f"  Min: {min(ages)}, Max: {max(ages)}, Mean: {np.mean(ages):.1f}")
            mature = sum(1 for a in ages if a >= self.grace_period)
            print(f"  Mature (>= {self.grace_period}): {mature}")
        
        # Update marking
        marked_this_cycle = 0
        for key, utility in utilities.items():
            state = self.neuron_states[key]
            neuron = self.ssnc_neurons[key[0]][key[1]]
            
            # Grace period
            if neuron.age < self.grace_period:
                continue
            
            # Mark low utility
            if utility < self.deletion_threshold:
                if not state['marked']:
                    state['marked'] = True
                    state['marked_cycles'] = 1
                    actions['soft_marked'].append(key)
                    marked_this_cycle += 1
                else:
                    state['marked_cycles'] += 1
            else:
                if state['marked'] and not state['soft_deleted']:
                    state['marked'] = False
                    state['marked_cycles'] = 0
        
        if verbose:
            print(f"\nMarking phase:")
            print(f"  Newly marked: {marked_this_cycle}")
            total_marked = sum(1 for s in self.neuron_states.values() if s['marked'])
            print(f"  Total currently marked: {total_marked}")
            
            # Show marked neurons with cycles
            if total_marked > 0:
                marked_info = [
                    (k, s['marked_cycles']) 
                    for k, s in self.neuron_states.items() 
                    if s['marked']
                ][:5]
                print(f"  Sample marked (first 5):")
                for key, cycles in marked_info:
                    print(f"    {key}: {cycles} cycles")
        
        # Apply soft deletion
        deletion_budget = 3
        deleted_this_cycle = 0
        
        for key, state in self.neuron_states.items():
            if deletion_budget <= 0:
                break
            
            if (state['marked'] and
                state['marked_cycles'] >= self.marking_cycles and
                not state['soft_deleted']):
                
                state['soft_deleted'] = True
                self.total_soft_deleted += 1
                deletion_budget -= 1
                deleted_this_cycle += 1
                
                actions['soft_gated'].append({
                    'layer': key[0],
                    'neuron': key[1],
                    'utility': utilities[key],
                    'cycles': state['marked_cycles']
                })
        
        if verbose:
            print(f"\nDeletion phase:")
            print(f"  Newly deleted: {deleted_this_cycle}")
            total_deleted = sum(1 for s in self.neuron_states.values() if s['soft_deleted'])
            print(f"  Total soft-deleted: {total_deleted}")
            print(f"  Cumulative total: {self.total_soft_deleted}")
        
        # Continue gating
        for key, state in self.neuron_states.items():
            if state['soft_deleted'] and state['gate_value'] > 0.05:
                state['gate_value'] *= 0.9
                state['gate_value'] = max(state['gate_value'], 0.05)
        
        if verbose:
            print('='*60 + '\n')
        
        return actions
    
    def get_statistics(self) -> Dict:
        marked = sum(1 for s in self.neuron_states.values() if s['marked'])
        soft_deleted = sum(1 for s in self.neuron_states.values() if s['soft_deleted'])
        
        all_utilities = []
        for layer_neurons in self.ssnc_neurons:
            for neuron in layer_neurons:
                all_utilities.append(neuron.utility)
        
        return {
            'cycle': self.cycle_count,
            'marked': marked,
            'soft_deleted': soft_deleted,
            'total_soft_deleted': self.total_soft_deleted,
            'utility_mean': float(np.mean(all_utilities)) if all_utilities else 0.0,
            'utility_std': float(np.std(all_utilities)) if all_utilities else 0.0,
            'low_utility_count': sum(1 for u in all_utilities if u < self.deletion_threshold)
        }


if __name__ == "__main__":
    print("="*70)
    print("ULTRA-DEBUGGED SSNC TEST")
    print("="*70)
    
    model = DebugSSNC_MLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        deletion_threshold=0.10,
        marking_cycles=5,
        grace_period=20  # Shorter for testing
    )
    
    print(f"\nModel created:")
    print(f"  Hidden dims: {model.hidden_dims}")
    print(f"  Grace period: {model.grace_period} cycles")
    print(f"  Marking cycles: {model.marking_cycles}")
    print(f"  Deletion threshold: {model.deletion_threshold}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Run cycles
    print(f"\n{'='*70}")
    print("RUNNING 30 CYCLES WITH VERBOSE OUTPUT")
    print('='*70)
    
    for cycle in range(30):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        model.observe_gradients()
        model.compute_all_utilities()
        
        # Verbose every 5 cycles
        verbose = (cycle % 5 == 0 or cycle >= 25)
        actions = model.plasticity_cycle(verbose=verbose)
        
        # Print important events
        if actions['soft_gated'] and not verbose:
            print(f"\nCycle {cycle}: ⚡ DELETED {len(actions['soft_gated'])} neurons!")
    
    # Final stats
    print(f"\n{'='*70}")
    stats = model.get_statistics()
    print("FINAL STATISTICS:")
    print('='*70)
    print(f"Marked: {stats['marked']}")
    print(f"Soft-deleted: {stats['soft_deleted']}")
    print(f"Total deleted: {stats['total_soft_deleted']}")
    print(f"Utility: {stats['utility_mean']:.6f} ± {stats['utility_std']:.6f}")
    print(f"Low utility count: {stats['low_utility_count']}")
    print('='*70)
    
    if stats['total_soft_deleted'] > 0:
        print("\n✅ SUCCESS! Plasticity is working!")
    else:
        print("\n❌ Still broken - see detailed output above")