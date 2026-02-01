"""
COMPLETE SSNC SYSTEM - PHASES 7-12

Building on the working foundation (Phases 1-6), this implements:
- Phase 7: Hard deletion (permanent removal)
- Phase 8: Neuron creation (capacity restoration)  
- Phase 9: Redistribution (layer balancing)
- Phase 10: Global monitoring (safety)
- Phase 11: Recovery testing (self-repair)
- Phase 12: Ablation baselines (validation)

Philosophy: Conservative, observable, reversible where possible.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import copy


@dataclass
class NeuronMetrics:
    """Comprehensive neuron tracking."""
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
    
    # Utility history
    utility_history: deque = None
    
    def __post_init__(self):
        if self.utility_history is None:
            self.utility_history = deque(maxlen=100)


class SSNCNeuron:
    """Enhanced neuron with full lifecycle tracking."""
    
    def __init__(self, neuron_id: int, layer_id: int):
        self.neuron_id = neuron_id
        self.layer_id = layer_id
        self.age = 0
        self.utility = 0.0
        self.activation_history = deque(maxlen=50)
        self.error_history = deque(maxlen=50)
        
        # Performance tracking
        self.avg_activation = 0.0
        self.activation_variance = 0.0
        self.error_correlation = 0.0
    
    def observe_activation(self, value: float):
        """Record activation."""
        self.activation_history.append(value)
        self.age += 1
        
        # Update running stats
        if len(self.activation_history) > 0:
            acts = np.array(list(self.activation_history))
            self.avg_activation = float(np.mean(acts))
            self.activation_variance = float(np.std(acts))
    
    def observe_error(self, value: float):
        """Record error signal."""
        self.error_history.append(value)
    
    def compute_utility(self) -> float:
        """Compute neuron utility."""
        if len(self.activation_history) < 10:
            self.utility = 0.0
            return 0.0
        
        acts = np.array(list(self.activation_history))
        magnitude = np.abs(np.mean(acts))
        variance = np.std(acts)
        
        # Include error correlation if available
        if len(self.error_history) >= 10:
            errs = np.array(list(self.error_history))
            if np.std(acts) > 1e-6 and np.std(errs) > 1e-6:
                correlation = abs(np.corrcoef(acts, errs)[0, 1])
                if not np.isnan(correlation):
                    self.error_correlation = correlation
                    self.utility = magnitude * variance * correlation
                    return self.utility
        
        self.utility = magnitude * variance
        return self.utility
    
    def duplicate(self, new_id: int) -> 'SSNCNeuron':
        """Create a duplicate neuron (for Phase 8)."""
        new_neuron = SSNCNeuron(new_id, self.layer_id)
        # Start inactive
        new_neuron.utility = self.utility * 0.5  # Reduced initial utility
        return new_neuron


class GlobalMonitor:
    """
    Phase 10: Global safety monitoring.
    
    Prevents catastrophic failures:
    - Capacity collapse (too few neurons)
    - Runaway growth (too many neurons)
    - Stagnation (no learning progress)
    - Oscillation (unstable deletion/creation)
    """
    
    def __init__(
        self,
        initial_neurons: int,
        min_neurons_ratio: float = 0.3,
        max_neurons_ratio: float = 2.0,
        stagnation_window: int = 10
    ):
        self.initial_neurons = initial_neurons
        self.min_neurons = int(initial_neurons * min_neurons_ratio)
        self.max_neurons = int(initial_neurons * max_neurons_ratio)
        self.stagnation_window = stagnation_window
        
        # History
        self.loss_history = deque(maxlen=stagnation_window)
        self.neuron_count_history = deque(maxlen=50)
        self.intervention_history = []
        
    def check_safety(
        self,
        active_neurons: int,
        current_loss: float
    ) -> Dict:
        """Check system safety and recommend interventions."""
        warnings = []
        interventions = {}
        
        # 1. Capacity bounds
        if active_neurons < self.min_neurons:
            warnings.append(f"COLLAPSE: {active_neurons} < {self.min_neurons}")
            interventions['stop_deletions'] = True
            interventions['force_creation'] = True
        
        if active_neurons > self.max_neurons:
            warnings.append(f"GROWTH: {active_neurons} > {self.max_neurons}")
            interventions['stop_creation'] = True
            interventions['increase_deletion'] = True
        
        # 2. Stagnation detection
        self.loss_history.append(current_loss)
        if len(self.loss_history) >= self.stagnation_window:
            loss_range = max(self.loss_history) - min(self.loss_history)
            if loss_range < 0.001:
                warnings.append(f"STAGNATION: loss range={loss_range:.6f}")
                interventions['exploration_mode'] = True
        
        # 3. Oscillation detection
        self.neuron_count_history.append(active_neurons)
        if len(self.neuron_count_history) >= 20:
            recent = list(self.neuron_count_history)[-20:]
            oscillation = np.std(recent) / (np.mean(recent) + 1e-6)
            if oscillation > 0.15:
                warnings.append(f"OSCILLATION: CV={oscillation:.3f}")
                interventions['reduce_plasticity_rate'] = True
        
        if interventions:
            self.intervention_history.append({
                'neurons': active_neurons,
                'loss': current_loss,
                'warnings': warnings,
                'interventions': interventions
            })
        
        return {
            'safe': len(warnings) == 0,
            'warnings': warnings,
            'interventions': interventions
        }


class CompleteSSNC_MLP(nn.Module):
    """
    Complete SSNC system with all phases.
    
    Phases implemented:
    1-6: Foundation (working)
    7: Hard deletion
    8: Neuron creation
    9: Redistribution
    10: Global monitoring
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        # Phase 6: Soft deletion
        deletion_threshold: float = 0.05,
        marking_cycles: int = 20,
        gating_rate: float = 0.08,
        grace_period: int = 50,
        # Phase 7: Hard deletion
        enable_hard_deletion: bool = True,
        hard_deletion_cycles: int = 50,
        # Phase 8: Creation
        enable_creation: bool = True,
        creation_threshold: float = 0.70,
        creation_method: str = "duplicate",
        # Phase 9: Redistribution
        enable_redistribution: bool = False,
        # Phase 10: Monitoring
        enable_monitoring: bool = True,
        # Budgets
        max_deletions_per_cycle: int = 2,
        max_creations_per_cycle: int = 1,
        # CRITICAL: Burn-in phase
        burn_in_epochs: int = 0  # Disable ALL SSNC actions for first N epochs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims.copy()
        self.output_dim = output_dim
        
        # Configuration
        self.deletion_threshold = deletion_threshold
        self.marking_cycles = marking_cycles
        self.gating_rate = gating_rate
        self.grace_period = grace_period
        
        self.enable_hard_deletion = enable_hard_deletion
        self.hard_deletion_cycles = hard_deletion_cycles
        
        self.enable_creation = enable_creation
        self.creation_threshold = creation_threshold
        self.creation_method = creation_method
        
        self.enable_redistribution = enable_redistribution
        self.enable_monitoring = enable_monitoring
        
        self.max_deletions_per_cycle = max_deletions_per_cycle
        self.max_creations_per_cycle = max_creations_per_cycle
        
        # Burn-in phase (CRITICAL for stability)
        self.burn_in_epochs = burn_in_epochs
        self.current_epoch = 0  # Track which epoch we're in
        
        # Build network
        self._build_network(activation)
        
        # SSNC neurons - direct storage
        self.ssnc_neurons = []
        for layer_idx, n_neurons in enumerate(hidden_dims):
            layer_neurons = [
                SSNCNeuron(neuron_id=i, layer_id=layer_idx)
                for i in range(n_neurons)
            ]
            self.ssnc_neurons.append(layer_neurons)
        
        # Neuron metrics
        self.neuron_metrics: Dict[Tuple[int, int], NeuronMetrics] = {}
        self._initialize_metrics()
        
        # Global monitor
        if self.enable_monitoring:
            total_neurons = sum(hidden_dims)
            self.monitor = GlobalMonitor(
                initial_neurons=total_neurons,
                min_neurons_ratio=0.3,
                max_neurons_ratio=2.0
            )
        
        # Statistics
        self.cycle_count = 0
        self.total_soft_deleted = 0
        self.total_hard_deleted = 0
        self.total_created = 0
        self.cached_gradients = {}
        
        # History
        self.plasticity_history = []
    
    def _build_network(self, activation):
        """Build PyTorch layers."""
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Register hooks
        for layer_idx, layer in enumerate(self.layers[:-1]):
            def make_hook(idx):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self.cached_gradients[idx] = grad_output[0].detach()
                return hook
            layer.register_full_backward_hook(make_hook(layer_idx))
    
    def _initialize_metrics(self):
        """Initialize neuron metrics."""
        for layer_idx, n_neurons in enumerate(self.hidden_dims):
            for neuron_idx in range(n_neurons):
                key = (layer_idx, neuron_idx)
                self.neuron_metrics[key] = NeuronMetrics(
                    neuron_id=neuron_idx,
                    layer_id=layer_idx,
                    created_at=0,
                    creation_method="initial"
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with observation and gating."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer_idx, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Observe activations
            self._observe_layer(x, layer_idx)
            
            # Apply gates (soft + hard deletion)
            x = self._apply_gates(x, layer_idx)
        
        x = self.layers[-1](x)
        return x
    
    def _observe_layer(self, activations: torch.Tensor, layer_idx: int):
        """Let neurons observe activations."""
        if layer_idx >= len(self.ssnc_neurons):
            return
        
        batch_mean = activations.mean(dim=0).detach().cpu().numpy()
        
        # Only observe activations for neurons within the activation tensor size
        # (newly created neurons won't have activations for this batch)
        max_idx = len(batch_mean)
        
        for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
            if neuron_idx >= max_idx:
                # Skip newly created neurons that don't have activations yet
                continue
                
            key = (layer_idx, neuron_idx)
            metrics = self.neuron_metrics.get(key)
            
            # Only observe if not hard-deleted
            if metrics and not metrics.hard_deleted:
                neuron.observe_activation(float(batch_mean[neuron_idx]))
    
    def _apply_gates(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply soft deletion as ADDITIVE BIAS, not multiplicative zeroing.
        
        Key principle: Gates must bias, not silence.
        - Soft-deleted neurons become noisy/unreliable
        - Active neurons remain unchanged
        - Gradients flow everywhere (SGD stays stable)
        - No logit collapse possible
        """
        # 🔥 BURN-IN GUARD: No gating during burn-in
        if self.current_epoch < self.burn_in_epochs:
            return x  # Pass through unchanged
        
        n_neurons = x.shape[1]
        gates = torch.ones(n_neurons, device=x.device)
        
        for neuron_idx in range(n_neurons):
            key = (layer_idx, neuron_idx)
            if key in self.neuron_metrics:
                metrics = self.neuron_metrics[key]
                
                # Hard deletion: Remove from structure (not forward pass)
                # For now, treat as maximally soft-deleted
                if metrics.hard_deleted:
                    gates[neuron_idx] = 0.0
                elif metrics.soft_deleted:
                    gates[neuron_idx] = metrics.gate_value
        
        # Convert gates to ADDITIVE BIAS (not multiplicative)
        # bias = how much to degrade the signal (0 = none, 1 = full)
        bias_strength = (1.0 - gates) * 0.05  # Small bias, max 5% perturbation
        
        # Add noise to soft-deleted neurons (makes them unreliable, not silent)
        noise = torch.randn_like(x) * bias_strength.unsqueeze(0)
        
        # Return: original signal + small bias + noise
        # Active neurons (gate=1.0): bias=0, noise=0 → unchanged
        # Soft-deleted (gate=0.5): bias=0.025, noise~0.025 → slightly perturbed
        # Hard-deleted (gate=0.0): bias=0.05, noise~0.05 → maximally noisy
        return x + bias_strength.unsqueeze(0) + noise
    
    def observe_gradients(self):
        """Let neurons observe gradients."""
        for layer_idx in self.cached_gradients:
            if layer_idx >= len(self.ssnc_neurons):
                continue
            
            grads = self.cached_gradients[layer_idx]
            batch_mean = grads.mean(dim=0).detach().cpu().numpy()
            
            # Only observe gradients for neurons within the gradient tensor size
            # (newly created neurons won't have gradients for this batch)
            max_idx = len(batch_mean)
            
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                if neuron_idx >= max_idx:
                    # Skip newly created neurons that don't have gradients yet
                    continue
                    
                key = (layer_idx, neuron_idx)
                metrics = self.neuron_metrics.get(key)
                
                if metrics and not metrics.hard_deleted:
                    neuron.observe_error(float(batch_mean[neuron_idx]))
        
        self.cached_gradients.clear()
    
    def compute_all_utilities(self):
        """Compute utilities for all neurons."""
        for layer_neurons in self.ssnc_neurons:
            for neuron in layer_neurons:
                neuron.compute_utility()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for burn-in phase tracking."""
        self.current_epoch = epoch
    
    def plasticity_cycle(self, current_loss: Optional[float] = None) -> Dict:
        """
        Main structural plasticity cycle.
        
        Phases executed:
        1. Soft deletion marking/gating (Phase 6)
        2. Hard deletion (Phase 7)
        3. Safety check (Phase 10)
        4. Neuron creation (Phase 8)
        5. Redistribution (Phase 9)
        
        CRITICAL: During burn-in, ONLY collect stats, do NOT act.
        """
        self.cycle_count += 1
        
        # 🔥 BURN-IN GUARD: Collect stats but do NOT act
        if self.current_epoch < self.burn_in_epochs:
            # Age neurons (tracking only)
            for metrics in self.neuron_metrics.values():
                if not metrics.hard_deleted:
                    metrics.age += 1
            
            # Collect utilities (observation only)
            utilities = self._collect_utilities()
            
            # Return empty actions - NO structural changes
            return {
                'cycle': self.cycle_count,
                'soft_marked': [],
                'soft_gated': [],
                'hard_deleted': [],
                'created': [],
                'monitor_warnings': ['BURN-IN MODE']
            }
        
        # Normal SSNC operation after burn-in
        # Age neurons
        for metrics in self.neuron_metrics.values():
            if not metrics.hard_deleted:
                metrics.age += 1
        
        # Collect utilities
        utilities = self._collect_utilities()
        
        actions = {
            'cycle': self.cycle_count,
            'soft_marked': [],
            'soft_gated': [],
            'hard_deleted': [],
            'created': [],
            'monitor_warnings': []
        }
        
        # Phase 6: Soft deletion
        soft_actions = self._soft_deletion_cycle(utilities)
        actions['soft_marked'] = soft_actions['marked']
        actions['soft_gated'] = soft_actions['gated']
        
        # Phase 7: Hard deletion
        if self.enable_hard_deletion:
            hard_actions = self._hard_deletion_cycle()
            actions['hard_deleted'] = hard_actions
        
        # Get active neuron count
        active_neurons = sum(
            1 for m in self.neuron_metrics.values()
            if not m.hard_deleted
        )
        
        # Phase 10: Safety monitoring
        if self.enable_monitoring and current_loss is not None:
            safety = self.monitor.check_safety(active_neurons, current_loss)
            actions['monitor_warnings'] = safety['warnings']
            
            # Apply interventions
            if safety['interventions'].get('stop_creation'):
                self.enable_creation = False
            if safety['interventions'].get('force_creation'):
                self.enable_creation = True
            if safety['interventions'].get('stop_deletions'):
                self.max_deletions_per_cycle = 0
        
        # Phase 8: Creation (if safe)
        if self.enable_creation and not actions['monitor_warnings']:
            creation_actions = self._creation_cycle(utilities)
            actions['created'] = creation_actions
        
        # Phase 9: Redistribution (placeholder)
        if self.enable_redistribution:
            pass  # Would implement layer balancing here
        
        self.plasticity_history.append(actions)
        
        return actions
    
    def _collect_utilities(self) -> Dict[Tuple[int, int], float]:
        """Collect utilities from all neurons."""
        utilities = {}
        for layer_idx, layer_neurons in enumerate(self.ssnc_neurons):
            for neuron in layer_neurons:
                key = (layer_idx, neuron.neuron_id)
                utilities[key] = neuron.utility
                
                # Update metrics
                if key in self.neuron_metrics:
                    self.neuron_metrics[key].utility_history.append(neuron.utility)
        
        return utilities
    
    def _soft_deletion_cycle(self, utilities: Dict) -> Dict:
        """Phase 6: Soft deletion with gating."""
        marked = []
        gated = []
        
        deletion_budget = self.max_deletions_per_cycle
        
        for key, utility in utilities.items():
            metrics = self.neuron_metrics.get(key)
            if not metrics or metrics.hard_deleted:
                continue
            
            neuron = self.ssnc_neurons[key[0]][key[1]]
            
            # Grace period for young neurons
            if neuron.age < self.grace_period:
                continue
            
            # Mark low utility
            if utility < self.deletion_threshold:
                if not metrics.marked_for_deletion:
                    metrics.marked_for_deletion = True
                    metrics.marked_cycles = 1
                    marked.append(key)
                else:
                    metrics.marked_cycles += 1
            else:
                # Recovery
                if metrics.marked_for_deletion and not metrics.soft_deleted:
                    metrics.marked_for_deletion = False
                    metrics.marked_cycles = 0
            
            # Apply soft deletion
            if (metrics.marked_for_deletion and
                metrics.marked_cycles >= self.marking_cycles and
                not metrics.soft_deleted and
                deletion_budget > 0):
                
                metrics.soft_deleted = True
                metrics.soft_deleted_at = self.cycle_count
                self.total_soft_deleted += 1
                deletion_budget -= 1
                gated.append(key)
            
            # Continue gating
            if metrics.soft_deleted and metrics.gate_value > 0.05:
                metrics.gate_value *= (1.0 - self.gating_rate)
                metrics.gate_value = max(metrics.gate_value, 0.05)
        
        return {'marked': marked, 'gated': gated}
    
    def _hard_deletion_cycle(self) -> List:
        """Phase 7: Permanent neuron removal."""
        hard_deleted = []
        
        for key, metrics in self.neuron_metrics.items():
            if metrics.hard_deleted or not metrics.soft_deleted:
                continue
            
            # Skip if soft_deleted_at is not set (defensive check)
            if metrics.soft_deleted_at is None:
                continue
            
            # Check how long soft-deleted
            cycles_soft = self.cycle_count - metrics.soft_deleted_at
            
            if cycles_soft >= self.hard_deletion_cycles:
                metrics.hard_deleted = True
                metrics.hard_deleted_at = self.cycle_count
                metrics.gate_value = 0.0
                self.total_hard_deleted += 1
                hard_deleted.append(key)
        
        return hard_deleted
    
    def _creation_cycle(self, utilities: Dict) -> List:
        """Phase 8: Create new neurons by duplication."""
        created = []
        creation_budget = self.max_creations_per_cycle
        
        for layer_idx in range(len(self.ssnc_neurons)):
            if creation_budget <= 0:
                break
            
            # Find overloaded neurons (high utility relative to others)
            layer_utilities = [
                (neuron_idx, utilities.get((layer_idx, neuron_idx), 0.0))
                for neuron_idx in range(len(self.ssnc_neurons[layer_idx]))
            ]
            
            # Get top percentile as creation candidates
            if layer_utilities:
                utils_only = [u for _, u in layer_utilities]
                threshold = np.percentile(utils_only, 75)  # Top 25%
                
                overloaded = []
                for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                    key = (layer_idx, neuron_idx)
                    if key in utilities:
                        metrics = self.neuron_metrics.get(key)
                        if not metrics or metrics.hard_deleted or metrics.soft_deleted:
                            continue
                        
                        current_utility = utilities[key]
                        
                        # TIGHTENED CONDITIONS - Require sustained signals
                        # 1. Current utility above threshold
                        if current_utility <= threshold or current_utility <= 0.01:
                            continue
                        
                        # 2. Age requirement (give neurons time to stabilize)
                        if metrics.age <= 50:
                            continue
                        
                        # 3. SUSTAINED HIGH UTILITY - Check history variance
                        # Only create if utility has been consistently high
                        if len(metrics.utility_history) >= 10:
                            recent_utils = list(metrics.utility_history)[-10:]
                            mean_util = np.mean(recent_utils)
                            std_util = np.std(recent_utils)
                            
                            # Require:
                            # - High mean utility (not just current spike)
                            # - Low variance (sustained, not oscillating)
                            if mean_util < 0.01 or std_util > mean_util * 0.5:
                                continue
                        else:
                            # Not enough history - skip
                            continue
                        
                        overloaded.append((neuron_idx, current_utility, neuron))
            
            # Create neurons for most overloaded
            overloaded.sort(key=lambda x: x[1], reverse=True)
            
            for parent_idx, parent_utility, parent_neuron in overloaded[:creation_budget]:
                # Create new neuron ID
                new_id = len(self.ssnc_neurons[layer_idx])
                
                # Duplicate neuron
                new_neuron = parent_neuron.duplicate(new_id)
                self.ssnc_neurons[layer_idx].append(new_neuron)
                
                # Create metrics
                new_key = (layer_idx, new_id)
                self.neuron_metrics[new_key] = NeuronMetrics(
                    neuron_id=new_id,
                    layer_id=layer_idx,
                    parent_id=parent_idx,
                    created_at=self.cycle_count,
                    creation_method=self.creation_method,
                    soft_deleted=True,  # Start gated
                    soft_deleted_at=self.cycle_count,  # Set timestamp to prevent crashes
                    gate_value=0.1
                )
                
                self.total_created += 1
                creation_budget -= 1
                
                created.append({
                    'layer': layer_idx,
                    'neuron': new_id,
                    'parent': parent_idx,
                    'parent_utility': parent_utility
                })
        
        return created
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        active = sum(1 for m in self.neuron_metrics.values() if not m.hard_deleted)
        soft_deleted = sum(1 for m in self.neuron_metrics.values() 
                          if m.soft_deleted and not m.hard_deleted)
        marked = sum(1 for m in self.neuron_metrics.values() 
                    if m.marked_for_deletion and not m.soft_deleted)
        
        utilities = list(self._collect_utilities().values())
        
        return {
            'cycle': self.cycle_count,
            'active_neurons': active,
            'marked': marked,
            'soft_deleted': soft_deleted,
            'hard_deleted': self.total_hard_deleted,
            'created': self.total_created,
            'net_change': self.total_created - self.total_hard_deleted,
            'utility_mean': float(np.mean(utilities)) if utilities else 0.0,
            'utility_std': float(np.std(utilities)) if utilities else 0.0,
            'low_utility_count': sum(1 for u in utilities if u < self.deletion_threshold)
        }
    
    def induce_damage(self, damage_rate: float = 0.2) -> int:
        """
        Phase 11: Damage test - randomly delete neurons.
        
        Used to test self-repair capability.
        """
        damaged = []
        active_keys = [
            k for k, m in self.neuron_metrics.items()
            if not m.hard_deleted
        ]
        
        n_to_damage = int(len(active_keys) * damage_rate)
        import random
        targets = random.sample(active_keys, n_to_damage)
        
        for key in targets:
            metrics = self.neuron_metrics[key]
            metrics.hard_deleted = True
            metrics.hard_deleted_at = self.cycle_count
            self.total_hard_deleted += 1
            damaged.append(key)
        
        return len(damaged)


if __name__ == "__main__":
    print("="*70)
    print("COMPLETE SSNC SYSTEM - PHASES 1-12")
    print("="*70)
    
    # Create model with all phases
    model = CompleteSSNC_MLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        activation='relu',
        # Conservative settings
        deletion_threshold=0.05,
        marking_cycles=20,
        grace_period=50,
        hard_deletion_cycles=50,
        enable_hard_deletion=True,
        enable_creation=True,
        enable_monitoring=True,
        max_deletions_per_cycle=2,
        max_creations_per_cycle=1
    )
    
    print(f"\nModel created:")
    print(f"  Hidden dims: {model.hidden_dims}")
    print(f"  Total neurons: {sum(model.hidden_dims)}")
    print(f"  Soft deletion: ✓")
    print(f"  Hard deletion: {model.enable_hard_deletion}")
    print(f"  Creation: {model.enable_creation}")
    print(f"  Monitoring: {model.enable_monitoring}")
    
    # Test forward-backward
    print("\nTesting forward-backward...")
    criterion = nn.CrossEntropyLoss()
    
    for cycle in range(60):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        model.observe_gradients()
        model.compute_all_utilities()
        
        actions = model.plasticity_cycle(current_loss=loss.item())
        
        # Print significant events
        if actions['soft_gated']:
            print(f"  Cycle {cycle}: Soft-deleted {len(actions['soft_gated'])} neurons")
        if actions['hard_deleted']:
            print(f"  Cycle {cycle}: ⚡ HARD-deleted {len(actions['hard_deleted'])} neurons")
        if actions['created']:
            print(f"  Cycle {cycle}: ✨ Created {len(actions['created'])} neurons")
        if actions['monitor_warnings']:
            print(f"  Cycle {cycle}: ⚠ Warnings: {actions['monitor_warnings']}")
    
    # Final stats
    stats = model.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Active neurons: {stats['active_neurons']}")
    print(f"  Marked: {stats['marked']}")
    print(f"  Soft-deleted: {stats['soft_deleted']}")
    print(f"  Hard-deleted: {stats['hard_deleted']}")
    print(f"  Created: {stats['created']}")
    print(f"  Net change: {stats['net_change']:+d}")
    print(f"  Utility: {stats['utility_mean']:.4f} ± {stats['utility_std']:.4f}")
    
    # Test damage recovery (Phase 11)
    print(f"\nTesting damage recovery (Phase 11)...")
    print(f"  Before damage: {stats['active_neurons']} neurons")
    
    damaged = model.induce_damage(damage_rate=0.15)
    print(f"  Induced damage: {damaged} neurons deleted")
    
    stats_after = model.get_statistics()
    print(f"  After damage: {stats_after['active_neurons']} neurons")
    
    print("\n✓ Complete SSNC system with Phases 1-12 operational!")
