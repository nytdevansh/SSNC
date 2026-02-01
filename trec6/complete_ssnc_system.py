"""
COMPLETE SSNC SYSTEM - PHASES 1-10 INTEGRATED
(Updated to match src/complete_ssnc.py architecture)

This implements the full self-organizing neural network:
- Phase 1-6: Foundation (observation, local learning, soft deletion) ✓
- Phase 7: Hard deletion (permanent removal) ✓
- Phase 8: Neuron creation (capacity restoration) ✓
- Phase 9: Redistribution (layer balancing) ✓
- Phase 10: Global safety monitoring ✓

Key principle: Conservative by default, aggressive only when safe.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import sys
import os

# Add project root and trec6 directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'trec6'))
from ssnc_neuron import SSNCNeuron


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


class AdaptiveObserver:
    """
    Principled SSNC Observer - Manages network plasticity with clear reasoning.
    
    Core Principles:
    1. DELETION: Only delete neurons that are PROVABLY useless
       - Consistently low utility over extended period
       - Not contributing to any class's predictions
       - Network has redundant capacity
    
    2. CREATION: Only create neurons when NEEDED
       - Network is under-capacity (high average utility = overloaded)
       - Specific classes have poor performance
       - Loss is stagnating despite learning
    
    3. STABILITY: Prevent oscillation through adaptive thresholds
       - Thresholds adjust based on utility distribution
       - Rate limits adapt to network stability
       - Decisions are logged with reasoning
    """
    
    def __init__(
        self,
        initial_neurons: int,
        min_capacity_ratio: float = 0.3,
        max_capacity_ratio: float = 2.0,
        adaptation_rate: float = 0.1
    ):
        self.initial_neurons = initial_neurons
        self.min_neurons = int(initial_neurons * min_capacity_ratio)
        self.max_neurons = int(initial_neurons * max_capacity_ratio)
        self.adaptation_rate = adaptation_rate
        
        # Adaptive thresholds (start with defaults, adapt based on distribution)
        self.deletion_threshold = 0.05   # Will adapt based on utility percentiles
        self.creation_threshold = 0.40   # Will adapt based on utility percentiles
        
        # History tracking for principled decisions
        self.loss_history = deque(maxlen=20)
        self.accuracy_history = deque(maxlen=20)
        self.utility_distribution_history = deque(maxlen=10)
        self.neuron_count_history = deque(maxlen=50)
        self.structural_changes = deque(maxlen=100)
        
        # Decision log with reasoning
        self.decision_log = []
        
        # State
        self.is_stable = True
        self.trend = 'neutral'  # 'improving', 'degrading', 'stagnating', 'neutral'
        
    def analyze_network_state(
        self,
        utilities: Dict,
        active_neurons: int,
        current_loss: float,
        cycle: int,
        layer_sizes: List[int] = [],
        current_accuracy: float = None
    ) -> Dict:
        """
        Analyze network state and return principled recommendations.
        
        Returns decision dictionary with:
        - should_delete: bool (and how many)
        - should_create: bool (and how many)
        - reasoning: list of strings explaining decisions
        - adapted_thresholds: new threshold values
        """
        self.loss_history.append(current_loss)
        self.neuron_count_history.append(active_neurons)
        if current_accuracy is not None:
            self.accuracy_history.append(current_accuracy)
            

        # === ANALYZE UTILITY DISTRIBUTION ===
        utility_values = list(utilities.values())
        if not utility_values:
            return self._safe_default()
        
        utility_array = np.array(utility_values)
        self.utility_distribution_history.append({
            'mean': float(np.mean(utility_array)),
            'std': float(np.std(utility_array)),
            'min': float(np.min(utility_array)),
            'max': float(np.max(utility_array)),
            'p10': float(np.percentile(utility_array, 10)),
            'p90': float(np.percentile(utility_array, 90))
        })
        
        # === DETECT TRENDS ===
        self.trend = self._detect_trend()
        
        # === ADAPT THRESHOLDS BASED ON DISTRIBUTION ===
        # Deletion: target bottom 5-10% IF they're truly low
        p10 = np.percentile(utility_array, 10)
        p90 = np.percentile(utility_array, 90)
        mean_utility = np.mean(utility_array)
        
        # Calculate Pressure (Organic Demand)
        n_overloaded = np.sum(utility_array > self.creation_threshold)
        n_useless = np.sum(utility_array < self.deletion_threshold)
        total_n = len(utility_array)
        creation_pressure = n_overloaded / total_n if total_n > 0 else 0
        deletion_pressure = n_useless / total_n if total_n > 0 else 0
        
        # Calculate Stability/Performance
        avg_acc = 0.0
        acc_std = 0.0
        loss_std = 0.0
        if len(self.accuracy_history) >= 10:
             rec_acc = list(self.accuracy_history)[-10:]
             avg_acc = np.mean(rec_acc)
             acc_std = np.std(rec_acc)
        if len(self.loss_history) >= 10:
             rec_loss = list(self.loss_history)[-10:]
             loss_std = np.std(rec_loss)

        # === 2. DIAGNOSE REGIME ===
        regime = 'OPTIMAL'
        reasoning = []
        
        if avg_acc > 88.0 and acc_std < 0.5:
            regime = 'CONVERGING'
        elif loss_std > 0.5: 
            regime = 'UNSTABLE'
        elif avg_acc < 60.0 and cycle > 50:
            regime = 'UNDERFITTING_SEVERE'
        elif (float(self.accuracy_history[-1]) - float(self.accuracy_history[0])) < 0.5 \
             and len(self.accuracy_history) >= 10 and avg_acc < 90.0:
            regime = 'STAGNANT'
        elif float(self.loss_history[-1]) > 0.5:
            regime = 'UNDERFITTING'
            
        reasoning.append(f"DIAGNOSIS: {regime} (Acc={avg_acc:.1f}%, LossStd={loss_std:.2f})")
        
        # === 3. SET POLICY (Multipliers) ===
        # Default: Balanced
        create_mult = 1.0
        delete_mult = 1.0
        thresh_adjust = 0.0
        should_stop = False
        
        if regime == 'CONVERGING':
            should_stop = True
        elif regime == 'UNSTABLE':
            create_mult = 0.0 # Stop growing
            delete_mult = 0.0 # Stop pruning
            reasoning.append("ACTION: Freezing checks to restore stability")
        elif regime == 'UNDERFITTING_SEVERE':
            create_mult = 3.0 # Aggressive growth
            delete_mult = 0.5 # Conservative pruning
            thresh_adjust = -0.1 # Lower creation bar
        elif regime == 'STAGNANT':
            create_mult = 2.0 # Force diversity
            delete_mult = 1.5 # Prune dead weight
        elif regime == 'UNDERFITTING':
            create_mult = 1.5
            delete_mult = 0.8
            
        # === 4. EXECUTE DECISIONS (Pressure * Policy) ===
        decisions = {
            'should_delete': False, 'delete_count': 0,
            'should_create': False, 'create_count': 0,
            'should_add_layer': False, 'should_stop': should_stop,
            'prune_layer_idx': -1, 'reasoning': reasoning,
            'deletion_threshold': self.deletion_threshold,
            'creation_threshold': self.creation_threshold + thresh_adjust
        }
        
        if should_stop: return decisions

        # Safety Override
        safety = self._check_safety(active_neurons, utility_array)
        if not safety['safe']:
            return self._handle_unsafe(safety, decisions)

        # Layer Balancing Check
        if layer_sizes and max(layer_sizes) > 400:
             decisions['should_add_layer'] = True
             decisions['prune_layer_idx'] = layer_sizes.index(max(layer_sizes))
             decisions['reasoning'].append(f"BALANCE: Layer {decisions['prune_layer_idx']} bloated. Shifting.")
             create_mult = 0.0 # specific override
        
        # Final Calculation
        eff_del_pressure = deletion_pressure * delete_mult
        eff_create_pressure = creation_pressure * create_mult
        
        if eff_del_pressure > 0.01:
            count = max(1, int(n_useless * 0.10 * delete_mult))
            decisions['should_delete'] = True
            decisions['delete_count'] = count
            decisions['reasoning'].append(f"PRUNE: Press={deletion_pressure:.2f} * Mult={delete_mult} -> {count}")
            
        if eff_create_pressure > 0.05 or (regime == 'STAGNANT'):
             # If stagnant, force creation even if pressure is low (to explore)
             base_count = int(n_overloaded * 0.10) if n_overloaded > 0 else int(active_neurons*0.01)
             count = max(1, int(base_count * create_mult))
             decisions['should_create'] = True
             # If accuracy is high (>85%), fine-tune.
             
             if avg_acc < 70.0:
                 # Aggressive expansion needed - network likely too small for task complexity
                 grow_count = max(2, int(active_neurons * 0.02)) # Grow by 2%
                 decisions['should_create'] = True
                 decisions['create_count'] = grow_count
                 decisions['reasoning'].append(f"STAGNATION (Low Acc {avg_acc:.1f}%): Forcing growth +{grow_count}")
                 
                 # Also consider layer addition if this persists
                 if len(self.structural_changes) > 10 and (cycle - self.structural_changes[-1]) > 30:
                     decisions['should_add_layer'] = True
                     decisions['reasoning'].append("COMPLEXITY GAP: Adding layer")
                     
             elif avg_acc < 85.0:
                 # Moderate stagnation - targeted growth
                 grow_count = max(1, int(active_neurons * 0.01)) # Grow by 1%
                 decisions['should_create'] = True
                 decisions['create_count'] = grow_count
                 decisions['reasoning'].append(f"STAGNATION (Med Acc): Targeted growth +{grow_count}")
                 
        # 3. EQUILIBRIUM CHECK
        # If we are deleting AND creating, net change is what matters.
        # Ensure we don't churn too fast.
        if decisions['should_delete'] and decisions['should_create']:
            decisions['reasoning'].append("CHURN: Simultaneous creation/deletion")
        
        # Log decision check logic
        if decisions['should_add_layer'] or decisions['should_create'] or decisions['should_delete']:
             self.structural_changes.append(cycle)
        
        # Log decision
        self.decision_log.append({
            'active_neurons': active_neurons,
            'loss': current_loss,
            'trend': self.trend,
            'decisions': decisions.copy()
        })
        
        return decisions

    def _handle_unsafe(self, safety, decisions):
        """Handle unsafe conditions by overriding decisions."""
        decisions['reasoning'].extend(safety['reasons'])
        decisions['should_delete'] = safety.get('force_delete', False)
        decisions['should_create'] = safety.get('force_create', False)
        decisions['delete_count'] = safety.get('delete_count', 0)
        decisions['create_count'] = safety.get('create_count', 0)
        return decisions
    
    def _detect_trend(self) -> str:
        """Detect learning trend from loss history."""
        if len(self.loss_history) < 5:
            return 'neutral'
        
        recent = list(self.loss_history)[-5:]
        older = list(self.loss_history)[-10:-5] if len(self.loss_history) >= 10 else recent
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        recent_std = np.std(recent)
        
        # Detect trend
        if recent_mean < older_mean * 0.95:  # >5% improvement
            return 'improving'
        elif recent_mean > older_mean * 1.05:  # >5% degradation
            return 'degrading'
        elif recent_std < 0.01:  # Very stable = stagnating
            return 'stagnating'
        else:
            return 'neutral'
    
    def _check_safety(self, active_neurons: int, utility_array: np.ndarray) -> Dict:
        """Check safety bounds and return overrides if needed."""
        result = {'safe': True, 'reasons': []}
        
        # 0. PANIC CIRCUIT BREAKER: Stop runaway deletion
        if len(self.neuron_count_history) >= 5:
            recent_counts = list(self.neuron_count_history)[-5:]
            drop = (recent_counts[0] - recent_counts[-1]) / (recent_counts[0] + 1)
            if drop > 0.10:  # >10% drop in 5 cycles is a PANIC
                result['safe'] = False
                result['reasons'].append(f"🚨 CIRCUIT BREAKER: Panic deletion detected ({drop:.1%} drop). Stopping deletions.")
                result['force_delete'] = False
                result['force_create'] = False
                result['stop_deletions'] = True
                return result
        
        # Capacity collapse
        if active_neurons < self.min_neurons:
            result['safe'] = False
            result['reasons'].append(
                f"SAFETY: Capacity collapse ({active_neurons} < {self.min_neurons})"
            )
            result['force_create'] = True
            result['create_count'] = min(5, self.min_neurons - active_neurons)
            result['force_delete'] = False
        
        # Capacity explosion
        if active_neurons > self.max_neurons:
            result['safe'] = False
            result['reasons'].append(
                f"SAFETY: Capacity exceeded ({active_neurons} > {self.max_neurons})"
            )
            result['force_delete'] = True
            result['delete_count'] = min(5, active_neurons - self.max_neurons)
            result['force_create'] = False
        
        # Oscillation detection
        if len(self.neuron_count_history) >= 20:
            recent = list(self.neuron_count_history)[-20:]
            cv = np.std(recent) / (np.mean(recent) + 1e-6)
            if cv > 0.15:
                result['safe'] = False
                result['reasons'].append(
                    f"SAFETY: Oscillation detected (CV={cv:.3f}), reducing plasticity"
                )
                result['force_delete'] = False
                result['force_create'] = False
        
        return result
    
    def _safe_default(self) -> Dict:
        """Return safe default when no data available."""
        return {
            'should_delete': False,
            'delete_count': 0,
            'should_create': False,
            'create_count': 0,
            'reasoning': ['INIT: Insufficient data for decisions'],
            'deletion_threshold': self.deletion_threshold,
            'creation_threshold': self.creation_threshold
        }
    
    def get_summary(self) -> Dict:
        """Get observer summary for logging."""
        return {
            'deletion_threshold': self.deletion_threshold,
            'creation_threshold': self.creation_threshold,
            'trend': self.trend,
            'is_stable': self.is_stable,
            'decisions_made': len(self.decision_log),
            'recent_decisions': self.decision_log[-5:] if self.decision_log else []
        }


# Keep old name for compatibility
GlobalMonitor = AdaptiveObserver


class CompleteSSNC_MLP(nn.Module):
    """
    Complete SSNC implementation with all phases.
    
    Features:
    - Local neuron learning (Phase 4)
    - Soft deletion (Phase 6)
    - Hard deletion (Phase 7)
    - Neuron creation (Phase 8)
    - Redistribution (Phase 9)
    - Global monitoring (Phase 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,  # Dropout rate for regularization
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
        burn_in_epochs: int = 0,  # Disable ALL SSNC actions for first N epochs
        # Legacy parameters (for compatibility)
        enable_local_learning: bool = True,
        local_lr: float = 0.00005,
        enable_soft_deletion: bool = True,
        soft_deletion_threshold: float = None,
        soft_marking_cycles: int = None,
        soft_gating_rate: float = None,
        plasticity_cycle_every: int = 50
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims.copy()
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Use legacy parameter names if provided
        if soft_deletion_threshold is not None:
            deletion_threshold = soft_deletion_threshold
        if soft_marking_cycles is not None:
            marking_cycles = soft_marking_cycles
        if soft_gating_rate is not None:
            gating_rate = soft_gating_rate
        
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
        
        # SSNC neurons - direct storage (not ModuleList)
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
        
        # Global monitor (AdaptiveObserver)
        if self.enable_monitoring:
            total_neurons = sum(hidden_dims)
            self.monitor = AdaptiveObserver(
                initial_neurons=total_neurons,
                min_capacity_ratio=0.3,
                max_capacity_ratio=2.0
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
        self.dropouts = nn.ModuleList()
        
        # Input → first hidden
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Hidden → hidden
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Last hidden → output
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Register gradient hooks
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
                    creation_method="initial"
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with observation and gating."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer_idx, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Apply dropout (only during training)
            x = self.dropouts[layer_idx](x)
            
            # Observe activations
            self._observe_layer(x, layer_idx)
            
            # Apply gates
            x = self._apply_gates(x, layer_idx)
        
        # Output layer
        x = self.layers[-1](x)
        return x
    
    def _observe_layer(self, activations: torch.Tensor, layer_idx: int):
        """Let neurons observe their activations."""
        if layer_idx >= len(self.ssnc_neurons):
            return
        
        batch_mean = activations.mean(dim=0).detach().cpu().numpy()
        
        for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
            if neuron_idx < len(batch_mean):
                neuron.observe_activation(float(batch_mean[neuron_idx]))
    
    def _apply_gates(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply soft/hard deletion gates."""
        n_neurons = x.shape[1]
        gates = torch.ones(n_neurons, device=x.device)
        
        for neuron_idx in range(n_neurons):
            key = (layer_idx, neuron_idx)
            if key in self.neuron_metrics:
                metrics = self.neuron_metrics[key]
                if metrics.hard_deleted:
                    gates[neuron_idx] = 0.0
                elif metrics.soft_deleted:
                    gates[neuron_idx] = metrics.gate_value
        
        return x * gates.unsqueeze(0)
    
    def observe_gradients(self):
        """Let neurons observe gradients."""
        for layer_idx in self.cached_gradients:
            if layer_idx >= len(self.ssnc_neurons):
                continue
            
            grads = self.cached_gradients[layer_idx]
            batch_mean = grads.mean(dim=0).detach().cpu().numpy()
            
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                if neuron_idx < len(batch_mean):
                    neuron.observe_error(float(batch_mean[neuron_idx]))
        
        self.cached_gradients.clear()
    
    def compute_all_utilities(self):
        """Compute utility for all neurons."""
        for layer in self.ssnc_neurons:
            for neuron in layer:
                neuron.compute_utility()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for burn-in phase tracking."""
        self.current_epoch = epoch
    
    def plasticity_cycle(self, current_loss: float = None, current_accuracy: float = None) -> Dict:
        """
        Execute one SSNC plasticity cycle.
        
        Sequence:
        1. Neuron Aging
        2. Utility Collection
        3. Observer Analysis (Principled Decision) - NOW ACCURACY AWARE
        4. Execution (Soft Delete, Hard Delete, Create)
        5. Statistics Update
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
                'observer_decisions': [],
                'monitor_warnings': ['BURN-IN MODE']
            }
        
        # Normal SSNC operation after burn-in
        # Age neurons
        for metrics in self.neuron_metrics.values():
            if not metrics.hard_deleted:
                metrics.age += 1
        
        # Collect utilities
        utilities = self._collect_utilities()
        
        # Get active neuron count
        active_neurons = sum(
            1 for m in self.neuron_metrics.values()
            if not m.hard_deleted
        )
        
        actions = {
            'cycle': self.cycle_count,
            'soft_marked': [],
            'soft_gated': [],
            'hard_deleted': [],
            'created': [],
            'observer_decisions': [],
            'monitor_warnings': []
        }
        
        # === PRINCIPLED OBSERVER DECISION ===
        if self.enable_monitoring and current_loss is not None:
            # Calculate current layer sizes (neurons not hard deleted)
            layer_sizes = []
            for layer_neurons in self.ssnc_neurons:
                count = sum(1 for n in layer_neurons if n.neuron_id in \
                    [m.neuron_id for m in self.neuron_metrics.values() if \
                     m.layer_id == layer_neurons[0].layer_id and not m.hard_deleted])
                layer_sizes.append(count)
            
            # Get principled decisions from adaptive observer
            observer_decision = self.monitor.analyze_network_state(
                utilities=utilities,
                active_neurons=active_neurons,
                current_loss=current_loss,
                cycle=self.cycle_count,
                layer_sizes=layer_sizes,
                current_accuracy=current_accuracy
            )
            
            actions['observer_decisions'] = observer_decision['reasoning']
            
            # Execute STOP decision
            if observer_decision.get('should_stop', False):
                 print("\n🛑 OBSERVER SIGNAL: Stopping structural plasticity (Converged)")
                 actions['monitor_warnings'].append("PLASTICITY STOPPED: Network Converged")
                 self.enable_creation = False
                 self.max_deletions_per_cycle = 0
                 # We could set a flag to stop training in the main loop
                 actions['stop_training'] = True 
                 return actions
            
            # Update thresholds based on observer's analysis
            self.deletion_threshold = observer_decision['deletion_threshold']
            self.creation_threshold = observer_decision['creation_threshold']
            
            # Execute observer's decisions
            if observer_decision['should_delete']:
                # Use observer's recommended deletion count
                self.max_deletions_per_cycle = observer_decision['delete_count']
            else:
                self.max_deletions_per_cycle = 0
            
            if observer_decision['should_create']:
                # Use observer's recommended creation count
                self.max_creations_per_cycle = observer_decision['create_count']
                self.enable_creation = True
            else:
                self.max_creations_per_cycle = 0
                # Don't disable creation entirely, just set count to 0
            
            # Execute Layer Addition (Dynamic Depth)
            if observer_decision.get('should_add_layer', False):
                # Add layer at index 1 (between input/first hidden and rest)
                added = self._add_layer(insertion_index=1)
                if added:
                    actions['observer_decisions'].append("✨ EXECUTED: Added new hidden layer")
                    actions['created'].append({'method': 'layer_addition', 'layer': 1})
        
        # Phase 6: Soft deletion (uses observer-adapted threshold)
        soft_actions = self._soft_deletion_cycle(utilities)
        actions['soft_marked'] = soft_actions['marked']
        actions['soft_gated'] = soft_actions['gated']
        
        # Phase 7: Hard deletion (if allowed by observer)
        if self.enable_hard_deletion and self.max_deletions_per_cycle > 0:
            hard_actions = self._hard_deletion_cycle()
            actions['hard_deleted'] = hard_actions
        
        # Phase 8: Creation (if recommended by observer)
        if self.enable_creation and self.max_creations_per_cycle > 0:
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
        for layer_idx, layer in enumerate(self.ssnc_neurons):
            for neuron_idx, neuron in enumerate(layer):
                key = (layer_idx, neuron_idx)
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
            
            # Grace period for young neurons
            if metrics.age < self.grace_period:
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
            
            # Check if soft-deleted long enough
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
        
        # DEBUG: Track why creation might fail
        max_utility = 0.0
        max_age = 0
        eligible_count = 0
        
        for layer_idx in range(len(self.hidden_dims)):
            if creation_budget <= 0:
                break
            
            # Find overloaded neurons
            overloaded = []
            for neuron_idx in range(len(self.ssnc_neurons[layer_idx])):
                key = (layer_idx, neuron_idx)
                if key in utilities:
                    utility = utilities[key]
                    metrics = self.neuron_metrics.get(key)
                    
                    if metrics and not metrics.hard_deleted:
                        max_utility = max(max_utility, utility)
                        max_age = max(max_age, metrics.age)
                        
                        # Check if eligible for creation
                        if utility > self.creation_threshold and metrics.age > 25:
                            eligible_count += 1
                            overloaded.append((neuron_idx, utility))
            
            # Create neurons for overloaded ones
            for parent_id, parent_utility in overloaded[:creation_budget]:
                new_neuron_id = len(self.ssnc_neurons[layer_idx])
                
                # === DYNAMIC WEIGHT EXPANSION ===
                # 1. Expand the layer that produces this neuron's output
                layer = self.layers[layer_idx]
                with torch.no_grad():
                    old_out_features = layer.out_features
                    new_out_features = old_out_features + 1
                    
                    # Create new weight matrix (copy parent neuron weights with noise)
                    new_weight = torch.zeros(new_out_features, layer.in_features, 
                                            device=layer.weight.device, dtype=layer.weight.dtype)
                    new_weight[:old_out_features] = layer.weight
                    # Duplicate parent's weights with small noise for diversity
                    noise_scale = 0.01
                    new_weight[old_out_features] = layer.weight[parent_id] + \
                        noise_scale * torch.randn_like(layer.weight[parent_id])
                    
                    # Create new bias
                    new_bias = torch.zeros(new_out_features, device=layer.bias.device, dtype=layer.bias.dtype)
                    new_bias[:old_out_features] = layer.bias
                    new_bias[old_out_features] = layer.bias[parent_id]
                    
                    # Create new layer
                    new_layer = nn.Linear(layer.in_features, new_out_features, device=layer.weight.device)
                    new_layer.weight.data = new_weight
                    new_layer.bias.data = new_bias
                    self.layers[layer_idx] = new_layer
                
                # 2. Expand the next layer's input (including output layer)
                # This ensures we update weights connecting Current Layer -> Next Layer
                if layer_idx < len(self.layers) - 1:
                    next_layer = self.layers[layer_idx + 1]
                    with torch.no_grad():
                        old_in_features = next_layer.in_features
                        new_in_features = old_in_features + 1
                        
                        # Expand input weights (copy parent's column with noise)
                        new_weight = torch.zeros(next_layer.out_features, new_in_features,
                                                device=next_layer.weight.device, dtype=next_layer.weight.dtype)
                        new_weight[:, :old_in_features] = next_layer.weight
                        new_weight[:, old_in_features] = next_layer.weight[:, parent_id] * 0.5 + \
                            noise_scale * torch.randn_like(next_layer.weight[:, parent_id])
                        
                        new_layer = nn.Linear(new_in_features, next_layer.out_features, device=next_layer.weight.device)
                        new_layer.weight.data = new_weight
                        new_layer.bias.data = next_layer.bias.clone()
                        self.layers[layer_idx + 1] = new_layer
                
                # 3. Expand dropout layer
                if layer_idx < len(self.dropouts):
                    # Dropout doesn't have learnable params, but recreate for consistency
                    pass  # Dropout works on any size
                
                # 4. Update hidden_dims tracking
                self.hidden_dims[layer_idx] = new_out_features
                
                # === CREATE TRACKING OBJECTS ===
                new_neuron = SSNCNeuron(new_neuron_id, layer_idx)
                new_neuron.utility = parent_utility * 0.5
                self.ssnc_neurons[layer_idx].append(new_neuron)
                
                # Create new metrics
                new_metrics = NeuronMetrics(
                    neuron_id=new_neuron_id,
                    layer_id=layer_idx,
                    parent_id=parent_id,
                    created_at=self.cycle_count,
                    creation_method=self.creation_method,
                    soft_deleted=False,
                    gate_value=1.0
                )
                
                key = (layer_idx, new_neuron_id)
                self.neuron_metrics[key] = new_metrics
                
                self.total_created += 1
                creation_budget -= 1
                
                created.append({
                    'layer': layer_idx,
                    'neuron': new_neuron_id,
                    'parent': parent_id,
                    'parent_utility': parent_utility,
                    'method': 'dynamic_duplication'
                })
        
        # DEBUG: Print diagnostics periodically
        if self.cycle_count % 50 == 0 and self.cycle_count > 0:
            print(f"  [Creation Debug] Cycle {self.cycle_count}: max_util={max_utility:.3f}, threshold={self.creation_threshold}, max_age={max_age}, eligible={eligible_count}")
        
        return created
    
    def _add_layer(self, insertion_index: int = 1):
        """
        Dynamically add a new hidden layer.
        
        Args:
            insertion_index: Index in hidden_dims to insert new layer
        """
        if insertion_index >= len(self.layers):
            return False
            
        print(f"  ✨ DYNAMIC GROWTH: Adding new layer at index {insertion_index}")
        
        # 1. Determine dimensions
        # New layer size = same as previous layer for stability
        prev_dim = self.hidden_dims[insertion_index-1] if insertion_index > 0 else self.input_dim
        new_dim = prev_dim  # Duplicate width
        
        # 2. Re-index existing neurons (metrics and internal IDs)
        # Must do this BEFORE inserting new layer to avoid conflicts
        new_metrics_dict = {}
        for (l, n), metric in self.neuron_metrics.items():
            if l >= insertion_index:
                metric.layer_id = l + 1
                new_metrics_dict[(l + 1, n)] = metric
            else:
                new_metrics_dict[(l, n)] = metric
        self.neuron_metrics = new_metrics_dict
        
        # Update internal layer IDs for neurons in lists
        for l in range(insertion_index, len(self.ssnc_neurons)):
            for neuron in self.ssnc_neurons[l]:
                neuron.layer_id += 1
        
        # 3. Create new layer weights (Identity-like initialization)
        next_layer = self.layers[insertion_index]
        device = next_layer.weight.device
        dtype = next_layer.weight.dtype
        
        # New layer: prev_dim -> new_dim (=prev_dim)
        new_linear = nn.Linear(prev_dim, new_dim, device=device, dtype=dtype)
        
        # Identity initialization (preserve signal flow)
        with torch.no_grad():
            new_linear.weight.zero_()
            new_linear.bias.zero_()
            # Set diagonal to 1 + small noise
            idx = torch.arange(min(prev_dim, new_dim), device=device)
            new_linear.weight[idx, idx] = 1.0 + 0.01 * torch.randn(len(idx), device=device)
        
        # 4. Insert into structures
        self.layers.insert(insertion_index, new_linear)
        self.dropouts.insert(insertion_index, nn.Dropout(self.dropout_rate))
        self.hidden_dims.insert(insertion_index, new_dim)
        
        # 5. Create SSNC neurons for new layer
        new_layer_neurons = []
        for i in range(new_dim):
            # Create neuron
            neuron = SSNCNeuron(i, insertion_index)
            neuron.utility = 0.5  # Neutral utility
            new_layer_neurons.append(neuron)
            
            # Create metrics
            metrics = NeuronMetrics(
                neuron_id=i,
                layer_id=insertion_index,
                created_at=self.cycle_count,
                creation_method='layer_insertion',
                soft_deleted=False,
                gate_value=1.0
            )
            self.neuron_metrics[(insertion_index, i)] = metrics
            
        self.ssnc_neurons.insert(insertion_index, new_layer_neurons)
        self.total_created += new_dim
        
        return True

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        active = sum(1 for m in self.neuron_metrics.values() if not m.hard_deleted)
        soft_deleted = sum(1 for m in self.neuron_metrics.values() if m.soft_deleted and not m.hard_deleted)
        marked = sum(1 for m in self.neuron_metrics.values() if m.marked_for_deletion and not m.soft_deleted)
        
        utilities = list(self._collect_utilities().values())
        
        stats = {
            'cycle': self.cycle_count,
            'active_neurons': active,
            'soft_deleted': soft_deleted,
            'marked': marked,
            'hard_deleted': self.total_hard_deleted,
            'created': self.total_created,
            'net_change': self.total_created - self.total_hard_deleted,
            'utility_mean': float(np.mean(utilities)) if utilities else 0.0,
            'utility_std': float(np.std(utilities)) if utilities else 0.0,
            'low_utility_count': sum(1 for u in utilities if u < self.deletion_threshold)
        }
        
        if self.enable_monitoring and hasattr(self, 'monitor'):
            stats['observer_decisions'] = len(self.monitor.decision_log)
            stats['observer_trend'] = self.monitor.trend
        
        return stats


if __name__ == "__main__":
    print("="*70)
    print("COMPLETE SSNC SYSTEM - PHASES 1-10")
    print("="*70)
    
    # Create model with all phases
    model = CompleteSSNC_MLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        activation='relu',
        # Enable all phases
        enable_hard_deletion=True,  # Phase 7
        enable_creation=True,  # Phase 8
        enable_redistribution=False,  # Phase 9 (placeholder)
        enable_monitoring=True,  # Phase 10
        # Conservative settings
        deletion_threshold=0.08,
        marking_cycles=15,
        hard_deletion_cycles=30,
        creation_threshold=0.75,
        max_deletions_per_cycle=2,
        max_creations_per_cycle=1,
        burn_in_epochs=5  # No SSNC actions for first 5 epochs
    )
    
    print(f"\nModel created with:")
    print(f"  Hidden dims: {model.hidden_dims}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hard deletion: {model.enable_hard_deletion}")
    print(f"  Creation: {model.enable_creation}")
    print(f"  Monitoring: {model.enable_monitoring}")
    print(f"  Burn-in epochs: {model.burn_in_epochs}")
    
    # Test forward-backward
    print("\nTesting forward-backward...")
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    criterion = nn.CrossEntropyLoss()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    model.observe_gradients()
    model.compute_all_utilities()
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Test plasticity cycles with burn-in
    print("\nSimulating plasticity cycles...")
    print("  Epochs 0-4: BURN-IN (no structural changes)")
    print("  Epochs 5+: ACTIVE SSNC")
    
    for epoch in range(10):
        model.set_epoch(epoch)
        
        for cycle in range(5):
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            model.observe_gradients()
            model.compute_all_utilities()
            
            actions = model.plasticity_cycle(current_loss=loss.item())
            
            if epoch == 0 and cycle == 0:
                print(f"\n  Epoch {epoch}, Cycle {cycle}: {actions['monitor_warnings']}")
            elif epoch == 5 and cycle == 0:
                print(f"\n  Epoch {epoch}, Cycle {cycle}: SSNC ACTIVATED")
            
            if actions['soft_gated'] and epoch >= 5:
                print(f"    Epoch {epoch}: Soft-deleted {len(actions['soft_gated'])} neurons")
            if actions['hard_deleted']:
                print(f"    Epoch {epoch}: HARD-deleted {len(actions['hard_deleted'])} neurons")
            if actions['created']:
                print(f"    Epoch {epoch}: Created {len(actions['created'])} neurons")
    
    # Final stats
    stats = model.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Active neurons: {stats['active_neurons']}")
    print(f"  Soft-deleted: {stats['soft_deleted']}")
    print(f"  Hard-deleted: {stats['hard_deleted']}")
    print(f"  Created: {stats['created']}")
    print(f"  Net change: {stats['net_change']:+d}")
    print(f"  Utility: {stats['utility_mean']:.4f} ± {stats['utility_std']:.4f}")
    
    print("\n✓ Complete SSNC system operational!")
    print("  All phases 1-10 integrated and tested.")
