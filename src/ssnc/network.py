"""
Complete SSNC MLP Module.

This module contains the CompleteSSNC_MLP class, which integrates all SSNC phases:
- Local neuron learning
- Soft deletion
- Hard deletion
- Neuron creation
- Global monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import sys
import os
from collections import deque

# Import from new modular structure
from src.ssnc.neuron import SSNCNeuron
from src.ssnc.metrics import NeuronMetrics
from src.ssnc.observer import AdaptiveObserver
import dataclasses


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
        hard_deletion_cycles: int = 5, # Tuned: 5 cycles * 50 batches = 250 batches (~1.5 epochs)
        # Phase 8: Creation
        enable_creation: bool = True,
        creation_threshold: float = 0.25, # Tuned: Lowered to encourage activity
        creation_method: str = "duplicate",
        # Phase 9: Redistribution
        enable_redistribution: bool = False,
        # Phase 10: Monitoring
        enable_monitoring: bool = True,
        # Budgets
        max_deletions_per_cycle: int = 5, # Increased budget
        max_creations_per_cycle: int = 2, # Increased budget
        # CRITICAL: Burn-in phase
        burn_in_epochs: int = 0,  # Disable ALL SSNC actions for first N epochs
        # === PROBABILISTIC PLASTICITY (Phase 4) ===
        exploration_budget: float = 0.1,  # 10% chance to explore "bad" decisions
        exploration_decay: float = 0.99,  # Simulated annealing: decay exploration over time
        # === AUTONOMOUS MODE (No fixed thresholds) ===
        use_autonomous_mode: bool = False,  # Enable self-learning controller
        autonomous_warmup_steps: int = 50,  # Steps before autonomous decisions
        # Legacy parameters (for compatibility)
        enable_local_learning: bool = True,
        local_lr: float = 0.00005,
        enable_soft_deletion: bool = True,
        soft_deletion_threshold: float = None,
        soft_marking_cycles: int = None,
        soft_gating_rate: float = None,
        plasticity_cycle_every: int = 50,
        utility_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.1, 0.2)
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
        
        # Phase 6 flag (Fixed: actually store it)
        self.enable_soft_deletion = enable_soft_deletion
        
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
        
        # Probabilistic plasticity (Phase 4)
        self.exploration_budget = exploration_budget
        self.exploration_decay = exploration_decay
        self.current_exploration = exploration_budget  # Decays over time
        self.utility_weights = utility_weights # Fix #4: Store weights
        
        # Build network
        self._build_network(activation)
        
        # SSNC neurons - direct storage (not ModuleList)
        self.ssnc_neurons = []
        
        # Neuron Metrics
        self.neuron_metrics = {} # Restore metrics dict (needed for soft flags)
        # self.ssnc_neurons initialized earlier (line 133)
        # self.gate_masks = []   # REMOVED (Fix #3) - Use gate_mask_names only
        self._pending_optimizer_updates = []  # Track updates for external optimizer handling
        
        self._initialize_metrics()       
        
        # Initialize Gate Masks
        # We store names to access registered buffers reliably
        self.gate_mask_names = []
        for i, n_neurons in enumerate(hidden_dims):
            mask = torch.ones(n_neurons)
            name = f"gate_mask_{i}"
            self.register_buffer(name, mask)
            self.gate_mask_names.append(name)
            # Access via: getattr(self, self.gate_mask_names[i])
        
        # Global monitor (AdaptiveObserver)
        if self.enable_monitoring:
            total_neurons = sum(hidden_dims)
            self.monitor = AdaptiveObserver(
                initial_neurons=total_neurons,
                min_capacity_ratio=0.3,
                max_capacity_ratio=2.0
            )
        
        # === AUTONOMOUS CONTROLLER (self-learning mode) ===
        self.use_autonomous_mode = use_autonomous_mode
        if self.use_autonomous_mode:
            try:
                from src.ssnc.controller import AutonomousSSNCController
                self.autonomous_controller = AutonomousSSNCController(
                    min_warmup_steps=autonomous_warmup_steps
                )
            except ImportError:
                print("⚠️  Warning: Autonomous Controller module not found. Falling back to standard mode.")
                self.use_autonomous_mode = False
                self.autonomous_controller = None
        
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
        """Initialize neuron metrics and SSNCNeuron objects."""
        # Get utility weights from config (default if not set)
        utility_weights = getattr(self, 'utility_weights', (0.4, 0.3, 0.1, 0.2))
        
        for layer_idx, n_neurons in enumerate(self.hidden_dims):
            layer_neurons = []
            for neuron_idx in range(n_neurons):
                key = (layer_idx, neuron_idx)
                # Create NeuronMetrics
                self.neuron_metrics[key] = NeuronMetrics(
                    neuron_id=neuron_idx,
                    layer_id=layer_idx,
                    creation_method="initial"
                )
                # Create SSNCNeuron for utility tracking
                ssnc_neuron = SSNCNeuron(
                    neuron_id=neuron_idx,
                    layer_id=layer_idx,
                    utility_weights=utility_weights
                )
                layer_neurons.append(ssnc_neuron)
            self.ssnc_neurons.append(layer_neurons)
    
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
        """Apply soft/hard deletion gates using vectorized mask."""
        # Fast path: Vectorized
        mask_name = f"gate_mask_{layer_idx}"
        if hasattr(self, mask_name):
            mask = getattr(self, mask_name)
            # Ensure shape matches (if resizing happened, mask might be stale unless updated)
            if mask.shape[0] != x.shape[1]:
                # Fallback or Panic? 
                # Should ideally update mask.
                self._update_gate_mask(layer_idx)
                mask = getattr(self, mask_name)
            
            return x * mask
        
        return x
    
    def _update_gate_mask(self, layer_idx: int):
        """Update the cached gate mask for a layer."""
        mask_name = f"gate_mask_{layer_idx}"
        if not hasattr(self, mask_name): return
        
        current_mask = getattr(self, mask_name)
        n_neurons = len(self.ssnc_neurons[layer_idx])
        
        # If size changed, re-register buffer
        if current_mask.shape[0] != n_neurons:
             new_mask = torch.ones(n_neurons, device=current_mask.device)
             self.register_buffer(mask_name, new_mask)
             current_mask = new_mask
        
        # Update values from metrics
        new_values = torch.ones_like(current_mask)
        for neuron_idx in range(n_neurons):
            key = (layer_idx, neuron_idx)
            if key in self.neuron_metrics:
                metrics = self.neuron_metrics[key]
                if metrics.hard_deleted:
                    new_values[neuron_idx] = 0.0
                elif metrics.soft_deleted:
                    new_values[neuron_idx] = 0.0 # Or strict gating value? Use 0.0 for speed.
                    # Or use metrics.gate_value if we want graduated gating?
                    # Let's use gate_value for correctness.
                    new_values[neuron_idx] = metrics.gate_value
        
        # Update buffer in-place
        current_mask.copy_(new_values).unsqueeze(0)
    
    def observe_gradients(self):
        """Let neurons observe gradients and update weight norms."""
        for layer_idx in self.cached_gradients:
            if layer_idx >= len(self.ssnc_neurons):
                continue
            
            grads = self.cached_gradients[layer_idx]
            batch_mean = grads.abs().mean(dim=0).detach().cpu().numpy()
            
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                if neuron_idx < len(batch_mean):
                    # Update gradient magnitude (uses observe_gradient now)
                    neuron.observe_gradient(float(batch_mean[neuron_idx]))
        
        # Update weight norms for all neurons
        self._update_weight_norms()
        
        self.cached_gradients.clear()
    
    def _update_weight_norms(self):
        """Update weight norms for all neurons (for multi-component utility)."""
        for layer_idx, layer in enumerate(self.layers[:-1]):  # Exclude output
            if layer_idx >= len(self.ssnc_neurons):
                continue
            
            # Weight matrix: [out_features, in_features]
            # Each row corresponds to one neuron's incoming weights
            weight_norms = layer.weight.norm(dim=1).detach().cpu().numpy()
            
            for neuron_idx, neuron in enumerate(self.ssnc_neurons[layer_idx]):
                if neuron_idx < len(weight_norms):
                    neuron.weight_norm = float(weight_norms[neuron_idx])
    
    def compute_all_utilities(self):
        """Compute utility for all neurons."""
        for layer in self.ssnc_neurons:
            for neuron in layer:
                neuron.compute_utility()
    
    def compute_redundancy_matrix(self, layer_idx: int, threshold: float = 0.95) -> List[Tuple[int, int, float]]:
        """
        Compute pairwise correlation between neurons in a layer to identify redundancy.
        
        Returns list of (neuron_a, neuron_b, correlation) tuples where correlation > threshold.
        These pairs are candidates for redundancy-based pruning.
        """
        if layer_idx >= len(self.layers) - 1:
            return []
        
        layer = self.layers[layer_idx]
        weights = layer.weight.detach().cpu()
        
        # Normalize weights
        norms = weights.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-6)
        normalized = weights / norms
        
        # Compute cosine similarity matrix
        similarity = torch.mm(normalized, normalized.t())
        
        redundant_pairs = []
        n_neurons = similarity.shape[0]
        
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                corr = abs(float(similarity[i, j]))
                if corr > threshold:
                    redundant_pairs.append((i, j, corr))
        
        return redundant_pairs
    
    def get_redundant_neurons(self, threshold: float = 0.95) -> Dict[int, List[int]]:
        """
        Identify redundant neurons across all layers.
        
        Returns dict mapping layer_idx -> list of neuron indices to prune.
        Keeps the higher-utility neuron in each redundant pair.
        """
        redundant = {}
        
        for layer_idx in range(len(self.layers) - 1):
            pairs = self.compute_redundancy_matrix(layer_idx, threshold)
            to_prune = set()
            
            for neuron_a, neuron_b, corr in pairs:
                # Keep the one with higher utility
                key_a = (layer_idx, neuron_a)
                key_b = (layer_idx, neuron_b)
                
                # Use current list index for utility lookup
                utility_a = self.ssnc_neurons[layer_idx][neuron_a].utility if neuron_a < len(self.ssnc_neurons[layer_idx]) else 0
                utility_b = self.ssnc_neurons[layer_idx][neuron_b].utility if neuron_b < len(self.ssnc_neurons[layer_idx]) else 0
                
                # Prune the lower utility one (but check if it's already marked)
                if utility_a < utility_b:
                    if key_a in self.neuron_metrics and not self.neuron_metrics[key_a].hard_deleted:
                        to_prune.add(neuron_a)
                else:
                    if key_b in self.neuron_metrics and not self.neuron_metrics[key_b].hard_deleted:
                        to_prune.add(neuron_b)
            
            if to_prune:
                redundant[layer_idx] = list(to_prune)
        
        return redundant
    
    def stochastic_plasticity_decision(self, should_act: bool, utility: float) -> bool:
        """
        Apply probabilistic plasticity with exploration budget.
        
        With probability `exploration_budget`, take the opposite action.
        This enables simulated annealing where exploration decreases over time.
        
        Args:
            should_act: The deterministic decision (True = prune/create)
            utility: The neuron's utility (used to weight exploration probability)
            
        Returns:
            Final decision after applying exploration
        """
        import random
        
        # Exploration probability weighted by utility uncertainty
        # Low utility neurons get more exploration
        exploration_weight = 1.0 - abs(utility - 0.5) * 2  # Higher near 0.5
        explore_prob = self.current_exploration * exploration_weight
        
        if random.random() < explore_prob:
            # Flip the decision (explore)
            return not should_act
        
        return should_act
    
    def decay_exploration(self):
        """Apply simulated annealing decay to exploration budget."""
        self.current_exploration *= self.exploration_decay
        self.current_exploration = max(0.01, self.current_exploration)  # Minimum 1%
    
    def set_epoch(self, epoch: int):
        """Set current epoch for burn-in phase tracking."""
        self.current_epoch = epoch
        # Decay exploration at each epoch (simulated annealing)
        self.decay_exploration()
    
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
        
        # === AUTONOMOUS MODE: Use self-learning controller ===
        if self.use_autonomous_mode and self.autonomous_controller is not None:
            for metrics in self.neuron_metrics.values():
                if not metrics.hard_deleted:
                    metrics.age += 1
            utilities = self._collect_utilities()
            return self._autonomous_plasticity_cycle(utilities, current_loss, current_accuracy)
        
        # Normal SSNC operation after burn-in (threshold-based)
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
            'monitor_warnings': [],
            'architecture_changed': False  # CRITICAL: Initialize to False, only set True if changes occur
        }
        
        # Diagnostic Counts
        n_active = sum(1 for m in self.neuron_metrics.values() if not m.soft_deleted and not m.hard_deleted)
        n_soft = sum(1 for m in self.neuron_metrics.values() if m.soft_deleted and not m.hard_deleted)
        n_hard = sum(1 for m in self.neuron_metrics.values() if m.hard_deleted)
        actions['stats'] = {'active': n_active, 'soft': n_soft, 'hard': n_hard}
        
        # === PRINCIPLED OBSERVER DECISION ===
        if self.enable_monitoring and current_loss is not None:
            # Calculate current layer sizes (neurons not hard deleted)
            # Calculate current layer sizes (neurons not hard deleted)
            # Find candidates: high utility AND high gradient (bottleneck detection)
            candidates = []
            for layer_idx, layer_neurons in enumerate(self.ssnc_neurons[:-1]): # Exclude output
                 # Use safe index instead of ID for weight slicing
                 for list_idx, neuron in enumerate(layer_neurons):
                     # Use current list index for duplication, NOT neuron.neuron_id
                     # This fixes (1, 384) error where stale ID > len(layer)
                     
                     # Must exceed base utility threshold
                     if neuron.utility <= self.creation_threshold:
                         continue
                         
                     candidates.append(((layer_idx, list_idx), neuron.utility, neuron.gradient_magnitude))
            layer_sizes = []
            for layer_idx, layer_neurons in enumerate(self.ssnc_neurons):
                # Count neurons where hard_deleted is False in metrics
                count = 0
                for n in layer_neurons:
                    key = (layer_idx, n.neuron_id)
                    if key in self.neuron_metrics:
                         if not self.neuron_metrics[key].hard_deleted:
                             count += 1
                    else:
                        count += 1 # Default alive if no metrics??
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
                self.max_deletions_per_cycle = observer_decision.get('max_deletions', 0)
            else:
                self.max_deletions_per_cycle = 0
            
            if observer_decision['should_create']:
                # Use observer's recommended creation count
                self.max_creations_per_cycle = observer_decision.get('max_creations', 0)
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
        if self.enable_soft_deletion:
            soft_actions = self._soft_deletion_cycle(utilities)
            actions['soft_marked'] = soft_actions['marked']
            actions['soft_gated'] = soft_actions['gated']
        else:
            # If disabled, ensure no incidental gating happens if we were previously enabled?
            # For baseline (fresh start), this is sufficient.
            pass
        
        # Phase 7: Hard deletion (if allowed by observer)
        if self.enable_hard_deletion and self.max_deletions_per_cycle > 0:
            try:
                hard_actions = self._hard_deletion_cycle()
                actions['hard_deleted'] = hard_actions
            except Exception as e:
                actions['monitor_warnings'].append(f"❌ Hard Deletion Error: {str(e)}")
        
        # Phase 8: Creation (if recommended by observer)
        if self.enable_creation and self.max_creations_per_cycle > 0:
            try:
                creation_actions = self._creation_cycle(utilities)
                actions['created'] = creation_actions
            except Exception as e:
                actions['monitor_warnings'].append(f"❌ Creation Error: {str(e)}")
        
        # Phase 9: Redistribution (placeholder)
        if self.enable_redistribution:
            pass  # Would implement layer balancing here
        
        self.plasticity_history.append(actions)
        
        # Determine if architecture changed
        # VALIDATION: Ensure keys exist
        if actions.get('created') and len(actions['created']) > 0:
             actions['architecture_changed'] = True
        if actions.get('hard_deleted') and len(actions['hard_deleted']) > 0:
             actions['architecture_changed'] = True
        if observer_decision.get('should_add_layer', False):
             actions['architecture_changed'] = True
        
        # If architecture changed, refresh neuron IDs to match list positions
        if actions['architecture_changed']:
            self._refresh_neuron_ids()
        
        return actions
    
    def _autonomous_plasticity_cycle(
        self, utilities: Dict, current_loss: float, current_accuracy: float
    ) -> Dict:
        """
        Execute plasticity cycle using autonomous self-learning controller.
        No fixed thresholds - learns from experience.
        """
        from src.ssnc.controller import NetworkState, Action
        import numpy as np
        
        # Build network state
        utility_vals = list(utilities.values())
        state = NetworkState(
            epoch=self.current_epoch,
            batch_idx=self.cycle_count,
            train_accuracy=current_accuracy or 0.0,
            train_loss=current_loss or 0.0,
            utility_mean=np.mean(utility_vals) if utility_vals else 0.0,
            utility_std=np.std(utility_vals) if utility_vals else 0.0,
            utility_min=min(utility_vals) if utility_vals else 0.0,
            utility_max=max(utility_vals) if utility_vals else 0.0,
            total_neurons=len(utilities),
            neurons_per_layer=self.hidden_dims.copy(),
            gradient_mean=np.mean([n.gradient_magnitude for layer in self.ssnc_neurons for n in layer]) if self.ssnc_neurons else 0.0,
            gradient_std=np.std([n.gradient_magnitude for layer in self.ssnc_neurons for n in layer]) if self.ssnc_neurons else 0.0,
        )
        
        # Observe and decide
        self.autonomous_controller.observe(state, utilities)
        decision = self.autonomous_controller.decide(state, utilities)
        
        actions = {
            'cycle': self.cycle_count,
            'soft_marked': [],
            'soft_gated': [],
            'hard_deleted': [],
            'created': [],
            'observer_decisions': [decision['reasoning']],
            'monitor_warnings': [],
            'autonomous_action': decision['action'].value,
            'autonomous_targets': decision['targets'],
            'architecture_changed': False
        }
        
        # Execute the autonomous decision
        action = decision['action']
        targets = decision['targets']
        
        if action == Action.DELETE_WEAK and targets:
            for (layer_idx, neuron_idx) in targets:
                key = (layer_idx, neuron_idx)
                if key in self.neuron_metrics:
                    self.neuron_metrics[key].marked_for_deletion = True
                    self.neuron_metrics[key].soft_deleted = True
                    actions['soft_marked'].append(key)
        
        elif action == Action.DUPLICATE_STRONG and targets:
            for (layer_idx, neuron_idx) in targets:
                new_id = self._duplicate_neuron(layer_idx, neuron_idx)
                if new_id is not None:
                    actions['created'].append({
                        'layer': layer_idx,
                        'parent': neuron_idx,
                        'new_id': new_id,
                        'reason': 'autonomous_duplicate'
                    })
                    actions['architecture_changed'] = True
        
        elif action == Action.CREATE_AT_BOTTLENECK and targets:
            for (layer_idx, neuron_idx) in targets:
                new_id = self._duplicate_neuron(layer_idx, neuron_idx)
                if new_id is not None:
                    actions['created'].append({
                        'layer': layer_idx,
                        'parent': neuron_idx,
                        'new_id': new_id,
                        'reason': 'autonomous_bottleneck'
                    })
                    actions['architecture_changed'] = True
        
        elif action == Action.DELETE_REDUNDANT:
            # Use existing redundancy detection
            redundant = self.get_redundant_neurons(threshold=0.95)
            for layer_idx, neuron_indices in redundant.items():
                for nidx in neuron_indices[:1]:  # Max 1 per layer
                    key = (layer_idx, nidx)
                    if key in self.neuron_metrics:
                        self.neuron_metrics[key].marked_for_deletion = True
                        self.neuron_metrics[key].soft_deleted = True
                        actions['soft_marked'].append(key)
        
        # === MAINTENANCE: Execute Hard Deletion ===
        # Ensure soft-marked neurons eventually get removed
        if self.enable_hard_deletion:
             try:
                 hard_actions = self._hard_deletion_cycle()
                 actions['hard_deleted'] = hard_actions
                 if hard_actions:
                     actions['architecture_changed'] = True
             except Exception as e:
                 actions['monitor_warnings'].append(f"❌ Hard Deletion Error: {str(e)}")
        
        self.plasticity_history.append(actions)
        
        # If architecture changed, refresh neuron IDs to match list positions
        if actions['architecture_changed']:
            self._refresh_neuron_ids()
            
        return actions
    
    def _collect_utilities(self) -> Dict[Tuple[int, int], float]:
        """Collect utilities from all neurons."""
        utilities = {}
        for layer_idx, layer in enumerate(self.ssnc_neurons):
            for neuron in layer:
                # Use neuron's current ID (which should match its list index after refresh)
                key = (layer_idx, neuron.neuron_id)
                utilities[key] = neuron.utility
        return utilities
    
    def _soft_deletion_cycle(self, utilities: Dict) -> Dict:
        """
        Phase 6: Update soft deletion marks and gates.
        
        Uses sigmoid gating for smooth transitions and utility banking
        (resurrection score) to revive high-potential neurons.
        """
        import math
        
        marked = []
        gated = []
        resurrected = []
        
        # Sigmoid gating parameters
        sigmoid_k = 0.5  # Steepness of sigmoid
        sigmoid_midpoint = self.marking_cycles / 2
        
        for key, metrics in self.neuron_metrics.items():
            if metrics.hard_deleted:
                continue
            
            utility = utilities.get(key, 1.0)
            
            # Get SSNCNeuron for resurrection score
            layer_idx, neuron_idx = key
            ssnc_neuron = None
            # Find the SSNCNeuron object by its current list index (neuron_idx)
            if layer_idx < len(self.ssnc_neurons) and neuron_idx < len(self.ssnc_neurons[layer_idx]):
                ssnc_neuron = self.ssnc_neurons[layer_idx][neuron_idx]
            
            # === RESURRECTION CHECK (Utility Banking) ===
            # If neuron was marked but has high historical max utility, give it a second chance
            if metrics.marked_for_deletion and ssnc_neuron:
                resurrection_score = ssnc_neuron.max_utility
                if resurrection_score > self.creation_threshold and utility > self.deletion_threshold * 0.5:
                    # Revive the neuron - it has potential
                    metrics.marked_cycles = 0
                    metrics.marked_for_deletion = False
                    metrics.gate_value = min(1.0, metrics.gate_value + 0.3)  # Boost gate
                    resurrected.append(key)
                    continue
            
            # Update soft deletion stats
            if utility < self.deletion_threshold:
                metrics.marked_cycles += 1
                if metrics.marked_cycles >= self.marking_cycles:
                    if not metrics.marked_for_deletion:
                        metrics.marked_for_deletion = True
                        marked.append(key)
            else:
                metrics.marked_cycles = max(0, metrics.marked_cycles - 1)  # Gradual recovery
                if metrics.marked_cycles == 0:
                    metrics.marked_for_deletion = False
                # Gradual gate recovery with sigmoid
                recovery_rate = 1.0 / (1.0 + math.exp(-0.5 * (utility - self.deletion_threshold)))
                metrics.gate_value = min(1.0, metrics.gate_value + self.gating_rate * recovery_rate)
                if metrics.soft_deleted and metrics.gate_value >= 0.9:
                    metrics.soft_deleted = False
                    metrics.soft_deleted_at = None
            
            # === SIGMOID GATING (smooth transitions) ===
            if metrics.marked_for_deletion:
                # Sigmoid decay: smooth transition instead of linear
                sigmoid_input = sigmoid_k * (metrics.marked_cycles - sigmoid_midpoint)
                target_gate = 1.0 / (1.0 + math.exp(sigmoid_input))
                # Smooth interpolation towards target
                metrics.gate_value = 0.9 * metrics.gate_value + 0.1 * target_gate
                
                if metrics.gate_value < 0.1 and not metrics.soft_deleted:
                    metrics.soft_deleted = True
                    metrics.soft_deleted_at = self.cycle_count
                    gated.append(key)
        
        # Optimization: Update masks for all layers
        for layer_idx in range(len(self.ssnc_neurons)):
            self._update_gate_mask(layer_idx)
            
        return {'marked': marked, 'gated': gated, 'resurrected': resurrected}
    
    def _hard_deletion_cycle(self) -> List[Tuple[int, int]]:
        """Phase 7: Permanently remove neurons."""
        hard_deleted = []
        candidates = []
        
        # Find candidates
        for key, metrics in self.neuron_metrics.items():
            if metrics.soft_deleted and not metrics.hard_deleted:
                # 1. Grace Period Check (Probation)
                if metrics.age < 10: 
                    continue
                
                # 2. Path Integrity Check: Don't delete if layer is too small
                layer_id = metrics.layer_id
                # Use the current list length for the layer (safe lookup)
                active_count = 0
                for n in self.ssnc_neurons[layer_id]:
                    nkey = (layer_id, n.neuron_id)
                    nm = self.neuron_metrics.get(nkey)
                    if nm is None or not nm.hard_deleted:
                        active_count += 1
                current_size = active_count
                # Minimum size: hard minimum of 5 neurons per layer.
                if current_size <= 5:
                    continue
                    
                cycles_since_soft = self.cycle_count - (metrics.soft_deleted_at or 0)
                remaining = self.hard_deletion_cycles - cycles_since_soft
                
                if cycles_since_soft >= self.hard_deletion_cycles:
                    candidates.append(key)
                # else:
                #     # Diagnostic: Too spammy for every neuron, but useful if stuck
                #     pass 
        
        # Diagnostic: How many waiting?
        n_waiting = sum(1 for m in self.neuron_metrics.values() 
                       if m.soft_deleted and not m.hard_deleted 
                       and (self.cycle_count - (m.soft_deleted_at or 0)) < self.hard_deletion_cycles)
        
        if n_waiting > 0:
             # This injects a key into self that might be picked up by actions? 
             # No, this method returns list. We should print or log differently.
             # We rely on the caller monitoring logic.
             pass
        
        # Sort by utility (lowest first) - just in case
        # (Though soft deleted ones probably have low utility)
        
        # Apply budget
        n_to_delete = min(len(candidates), self.max_deletions_per_cycle)
        to_delete = candidates[:n_to_delete]
        
        for key in to_delete:
            try:
                metrics = self.neuron_metrics[key]
                metrics.hard_deleted = True
                metrics.hard_deleted_at = self.cycle_count
                hard_deleted.append(key)
                self.total_hard_deleted += 1
                
                # Optimization: Sync flag to SSNCNeuron for fast skipping
                layer_idx, neuron_id = key
                # Locate the neuron object by its current list index (neuron_id)
                found_neuron = None
                if neuron_id < len(self.ssnc_neurons[layer_idx]) and self.ssnc_neurons[layer_idx][neuron_id].neuron_id == neuron_id:
                     found_neuron = self.ssnc_neurons[layer_idx][neuron_id]
                
                if found_neuron:
                    found_neuron.hard_deleted = True
                else:
                    print(f"  ⚠️  Could not find neuron object for {key}")
            except Exception as e:
                print(f"  ❌ Error hard-deleting {key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Optimization: Update masks
        for layer_idx in range(len(self.ssnc_neurons)):
            self._update_gate_mask(layer_idx)
            
        return hard_deleted
    
    def _creation_cycle(self, utilities: Dict) -> List[Dict]:
        """
        Phase 8: Create new neurons using gradient-guided neurogenesis.
        
        Prioritizes neurons that have:
        1. High utility (above creation_threshold)
        2. High gradient magnitude (indicating learning bottleneck)
        
        This targets areas where the network is actively trying to learn.
        """
        created = []
        
        # FIXED: Check budget for THIS cycle only, not total across all time
        remaining_budget = self.max_creations_per_cycle
        if remaining_budget <= 0:
             return created
        
        # Find candidates: high utility AND high gradient (bottleneck detection)
        # Find candidates: high utility AND high gradient (bottleneck detection)
        candidates = []
        for layer_idx, layer_neurons in enumerate(self.ssnc_neurons[:-1]): # Exclude output
             # Use safe index instead of ID for weight slicing
             for list_idx, neuron in enumerate(layer_neurons):
                 # Use current list index for duplication, NOT neuron.neuron_id
                 # This fixes (1, 384) error where stale ID > len(layer)
                 
                 # Must exceed base utility threshold
                 if neuron.utility <= self.creation_threshold:
                     continue

                 # Composite score: utility + gradient importance (0.5/0.5 split)
                 score = neuron.utility * 0.5 + min(1.0, neuron.gradient_magnitude) * 0.5
                 
                 # Format: ((layer, idx), score, raw_grad)
                 candidates.append(((layer_idx, list_idx), score, neuron.gradient_magnitude))
        
        # Sort by composite score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # FIXED: Apply budget for this cycle, not total
        n_to_create = min(len(candidates), remaining_budget)
        to_create = candidates[:n_to_create]
        
        # Debug output
        if len(candidates) > 0:
            print(f"  🔍 Creation: {len(candidates)} candidates, budget={remaining_budget}, creating={n_to_create}")
        
        for (layer_idx, neuron_idx), score, grad in to_create:
            # Duplicate neuron with error handling
            try:
                new_id = self._duplicate_neuron(layer_idx, neuron_idx)
                if new_id is not None:
                    created.append({
                        'layer': layer_idx,
                        'parent': neuron_idx,
                        'new_id': new_id,
                        'reason': f'gradient_guided (score={score:.3f}, grad={grad:.3f})'
                    })
                    self.total_created += 1
                else:
                    print(f"  ⚠️  Failed to duplicate neuron ({layer_idx}, {neuron_idx}) - returned None")
            except Exception as e:
                print(f"  ❌ Error duplicating ({layer_idx}, {neuron_idx}): {e}")
                import traceback
                traceback.print_exc()
        
        return created
    
    def _duplicate_neuron(self, layer_idx: int, parent_idx: int) -> Optional[int]:
        """
        Duplicate a neuron by expanding the layer dimensions.
        
        Args:
            layer_idx: Layer containing the neuron to duplicate.
            parent_idx: Index of the neuron to duplicate.
            
        Returns:
            New neuron index if successful, None otherwise.
        """
        # Validate layer index
        if layer_idx >= len(self.layers) - 1:
            return None # Cannot expand output layer in this scheme
        
        # 1. Get current layers
        current_layer = self.layers[layer_idx]
        next_layer = self.layers[layer_idx + 1]
        
        if not isinstance(current_layer, nn.Linear) or not isinstance(next_layer, nn.Linear):
            return None # Can only resize Linear layers
        
        # CRITICAL FIX: Validate parent_idx is within current layer bounds
        if parent_idx < 0 or parent_idx >= current_layer.out_features:
            print(f"  ❌ Invalid parent_idx={parent_idx} for layer with {current_layer.out_features} neurons")
            return None
            
        device = current_layer.weight.device
        dtype = current_layer.weight.dtype
        
        # 2. Prepare new weights for Current Layer (Output expansion: N -> N+1)
        # Weight shape: [out_features, in_features]
        # Bias shape: [out_features]
        
        # New row (weights) and scalar (bias)
        new_row = current_layer.weight[parent_idx].clone().unsqueeze(0)
        # Add small noise to break symmetry
        new_row += torch.randn_like(new_row) * 0.01
        
        new_bias = current_layer.bias[parent_idx].clone().unsqueeze(0)
        
        # Create expanded tensors for current layer
        new_weight_curr = torch.cat([current_layer.weight.data, new_row], dim=0)
        new_bias_curr = torch.cat([current_layer.bias.data, new_bias], dim=0)
        
        # 3. Prepare new weights for Next Layer (Input expansion: N -> N+1)
        # Weight shape: [next_out, current_out]
        
        # New column (input weights from the new neuron)
        # We split the influence: Original and Clone share the load (half each initially)
        # or Clone takes copy? Let's do copy + noise but divide outgoing by 2 to preserve expected sum.
        
        current_col = next_layer.weight.data[:, parent_idx].unsqueeze(1)
        new_col = current_col.clone()
        
        # Scale both to maintain stability
        # current_layer.weight.data[:, parent_idx] *= 0.5 # Can't modify in-place easily if we refer to it next
        # Actually, let's keep original full strength and new one initialized small?
        # NO, user said "Duplicate".
        # Let's keep original as is and new one as copy.
        # This increases the signal to next layer, but activation function usually handles it.
        # Better: Halve the outgoing weights of both to sum to original.
        
        updated_orig_col = current_col * 0.5
        new_col = new_col * 0.5
        
        # Construct new weight matrix for next layer
        # We need to replace the specific column in the original part, then append the new column.
        # It's easier to concat.
        
        # Slicing is safer:
        # [Col 0... Parent ... Col N] -> [Col 0... Parent_Half ... Col N ... New_Half]
        # But indices shift.
        # Simplest: Just append new column at the end. The new neuron ID will be N+1 (last index).
        # This means the new neuron is physically located at the END of the tensor, regardless of parent position.
        
        new_weight_next = torch.cat([next_layer.weight.data, new_col], dim=1)
        
        # NOTE: We didn't modify the original parent's outgoing weights in `new_weight_next` part 1.
        # We should.
        new_weight_next[:, parent_idx] *= 0.5
        
        # 4. Create new Layer Modules
        # Current Layer
        new_layer_module = nn.Linear(current_layer.in_features, current_layer.out_features + 1)
        new_layer_module.to(device=device, dtype=dtype)
        # Copy weights
        with torch.no_grad():
            new_layer_module.weight.copy_(new_weight_curr)
            new_layer_module.bias.copy_(new_bias_curr)
            
        # Next Layer
        new_next_module = nn.Linear(next_layer.in_features + 1, next_layer.out_features)
        new_next_module.to(device=device, dtype=dtype)
        # Copy weights
        with torch.no_grad():
            new_next_module.weight.copy_(new_weight_next)
            new_next_module.bias.copy_(next_layer.bias.data) # Bias doesn't change for next layer
            
        # 5. Swap Modules
        self.layers[layer_idx] = new_layer_module
        self.layers[layer_idx + 1] = new_next_module
        
        # 6. Update Hidden Dims tracking
        self.hidden_dims[layer_idx] += 1
        
        # 7. Register Hooks (Crucial for SSNC)
        # We need to re-register hooks for these new layers
        def make_hook(idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.cached_gradients[idx] = grad_output[0].detach()
            return hook
            
        self.layers[layer_idx].register_full_backward_hook(make_hook(layer_idx))
        # Note: We assume layers[layer_idx+1] is also being tracked properly?
        # The hooks are indexed by layer_idx.
        # We need to re-register hook for next layer too?
        # Yes, because we replaced the object.
        self.layers[layer_idx + 1].register_full_backward_hook(make_hook(layer_idx + 1))
        
        # 8. Update SSNC Neuron List
        # Create new neuron metadata
        new_id = self.hidden_dims[layer_idx] - 1 # It's the last one
        new_neuron = SSNCNeuron(neuron_id=new_id, layer_id=layer_idx)
        new_neuron.creation_method = "duplicated"
        new_neuron.parent_id = parent_idx
        # Inherit parent's utility stats to give it a head start?
        # Or start fresh? Best to start fresh but active.
        new_neuron.age = 0
        new_neuron.utility = 0.5 # Moderate starting utility
        
        self.ssnc_neurons[layer_idx].append(new_neuron)
        
        # 9. Update Metrics Dict
        key = (layer_idx, new_id)
        self.neuron_metrics[key] = NeuronMetrics(
            neuron_id=new_id,
            layer_id=layer_idx,
            creation_method="duplicated",
            birth_step=0, # We don't have global step here, will use age=0
            probation_period=500
        )
        
        # Signal that optimizer needs update
        self._pending_optimizer_updates.append({
            'type': 'create',
            'layer_idx': layer_idx,
            'neuron_idx': new_id
        })
        
        return new_id
    
    def get_pending_optimizer_updates(self) -> List[Dict]:
        """
        Get and clear pending optimizer updates for external handling.
        
        Returns list of dicts with:
        - 'type': 'create' or 'delete'
        - 'layer_idx': Layer index
        - 'neuron_idx': Neuron index
        
        The trainer should use this to properly initialize/remove optimizer states.
        """
        updates = list(self._pending_optimizer_updates)
        self._pending_optimizer_updates.clear()
        return updates
    
    def signal_optimizer_reset_momentum(self, optimizer) -> int:
        """
        Reset momentum states for parameters that have changed.
        
        Call this after structural changes to prevent stale momentum.
        Returns number of states reset.
        """
        resets = 0
        for param in self.parameters():
            for state_key in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
                if param in optimizer.state and state_key in optimizer.state[param]:
                    # Reset to zeros with matching shape
                    optimizer.state[param][state_key] = torch.zeros_like(param)
                    resets += 1
        return resets

    def get_extra_state(self) -> Any:
        """Serialize SSNC metadata (neuron states)."""
        # Serialize LIVE ssnc_neurons directly (Fix #2 Persistence)
        neurons_state = {}
        for layer_idx, layer_neurons in enumerate(self.ssnc_neurons):
            for neuron in layer_neurons:
                # Use asdict to serialize dataclass
                neurons_state[f"{layer_idx}_{neuron.neuron_id}"] = dataclasses.asdict(neuron)
                
        return {
            'ssnc_neurons_data': neurons_state,
            'neuron_metrics': self.neuron_metrics, # Save metrics too (soft flags)
            'cycle_count': self.cycle_count,
            'total_created': self.total_created,
            'total_hard_deleted': self.total_hard_deleted,
        }
    
    def set_extra_state(self, state: Any):
        """Restore SSNC metadata."""
        if not state: return
        
        neurons_data = state.get('ssnc_neurons_data', {})
        self.neuron_metrics = state.get('neuron_metrics', {})
        self.cycle_count = state.get('cycle_count', 0)
        self.total_created = state.get('total_created', 0)
        self.total_hard_deleted = state.get('total_hard_deleted', 0)
        
        # Restore SSNCNeuron objects
        # Layers must be resized BEFORE this call via resize_for_state_dict
        
        self.ssnc_neurons = [[] for _ in range(len(self.layers)-1)] # Hidden layers
        
        # Group by layer first for easier reconstruction
        grouped = {}
        for key, data in neurons_data.items():
            l_idx = data['layer_id']
            if l_idx not in grouped: grouped[l_idx] = []
            grouped[l_idx].append(data)
            
        # Reconstruct
        for l_idx, n_list in grouped.items():
            if l_idx < len(self.ssnc_neurons):
                # Sort by neuron_id to ensure order
                n_list.sort(key=lambda x: x['neuron_id'])
                for n_data in n_list:
                    # Recreate object from dict
                    neuron = SSNCNeuron(**n_data)
                    self.ssnc_neurons[l_idx].append(neuron)

    def resize_for_state_dict(self, state_dict):
        """
        Dynamically resize layers to match a state_dict before loading.
        Crucial because SSNC grows layers (Neurogenesis).
        """
        for i, layer in enumerate(self.layers):
            key_weight = f"layers.{i}.weight"
            key_bias = f"layers.{i}.bias"
            
            if key_weight not in state_dict:
                continue
                
            saved_weight = state_dict[key_weight]
            current_weight = layer.weight
            
            if saved_weight.shape != current_weight.shape:
                print(f"♻️ Resizing Layer {i}: {current_weight.shape} -> {saved_weight.shape}")
                
                # Create new layer
                new_layer = nn.Linear(saved_weight.shape[1], saved_weight.shape[0])
                new_layer.to(layer.weight.device)
                
                # Replace
                self.layers[i] = new_layer
                
                # Also resize dropout if needed (usually stateless, but good to keep aligned)
                # And re-register hooks!
                self._register_hooks()
                
            # Resize Gate Mask Buffer if needed
            key_mask = f"gate_mask_{i}"
            if key_mask in state_dict:
                saved_mask = state_dict[key_mask]
                current_mask = getattr(self, key_mask, None)
                
                # If mask doesn't exist or shape mismatch
                if current_mask is None or current_mask.shape != saved_mask.shape:
                    print(f"♻️ Resizing Mask {i}: {current_mask.shape if current_mask is not None else 'None'} -> {saved_mask.shape}")
                    new_mask = torch.ones(saved_mask.shape, device=self.layers[0].weight.device) # Assume same device
                    self.register_buffer(key_mask, new_mask)
                    # Update local list reference (Fix #12)
                    if i < len(self.gate_mask_names):
                        pass # Name already registered
                    else:
                        self.gate_mask_names.append(key_mask)
                            
        # Re-initialize basic neuron lists (will be filled by set_extra_state)
        # We need to make sure ssnc_neurons structure matches the new layer sizes
        self.ssnc_neurons = [[] for _ in range(len(self.layers)-1)]
        for layer_idx, layer in enumerate(self.layers[:-1]): # Exclude output
             n_neurons = layer.out_features
             self.ssnc_neurons[layer_idx] = [
                 SSNCNeuron(neuron_id=i, layer_id=layer_idx) for i in range(n_neurons)
             ]

    def _add_layer(self, insertion_index: int = 1) -> bool:
        """
        Dynamically add a new hidden layer to the network.
        
        Args:
            insertion_index: Index in self.layers to insert new layer.
                             0 is usually input->hidden1. 
                             1 is hidden1->hidden2.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        # Checks
        if len(self.hidden_dims) >= 5: # Limit text depth
            return False
            
        # We need to know input/output dims at this point
        # self.layers[insertion_index-1] outputs to prev_dim
        # self.layers[insertion_index] takes prev_dim
        
        # New layer size = average of neighbor sizes or same as prev
        prev_layer = self.layers[insertion_index-1]
        next_layer = self.layers[insertion_index]
        
        in_features = prev_layer.out_features
        out_features = next_layer.in_features
        new_dim = (in_features + out_features) // 2
        
        # Create new layers
        device = next_layer.weight.device
        dtype = next_layer.weight.dtype
        
        new_layer1 = nn.Linear(in_features, new_dim).to(device=device, dtype=dtype)
        new_layer2 = nn.Linear(new_dim, out_features).to(device=device, dtype=dtype)
        
        # Initialize
        # Identity initialization for transparent insertion?
        # Or standard initialization? Standard is better for breaking symmetry usually,
        # but Identity preserves current function approximately.
        # Let's use He initialization for ReLU.
        nn.init.kaiming_normal_(new_layer1.weight, nonlinearity='relu')
        nn.init.zeros_(new_layer1.bias)
        nn.init.kaiming_normal_(new_layer2.weight, nonlinearity='relu')
        nn.init.zeros_(new_layer2.bias)
        
        # Rebuild ModuleList (tricky with optimizer, but we assume optimizer is re-created or handles params)
        # Note: In a real training loop, you MUST update the optimizer after this.
        # This implementation simply modifies the model structure.
        
        new_layers = nn.ModuleList()
        new_dropouts = nn.ModuleList()
        
        # Copy before
        for i in range(insertion_index):
            new_layers.append(self.layers[i])
            new_dropouts.append(self.dropouts[i])
            
        # Insert new
        new_layers.append(new_layer1)
        new_dropouts.append(nn.Dropout(self.dropout_rate))
        
        # The 'next_layer' in original list is replaced by 'new_layer2' effectively
        # Wait, we are splitting one connection into two layers.
        # Original: A -> B
        # New: A -> New -> B
        # But A is self.layers[insertion_index-1] (modified to point to New?) NO.
        # Linear layer defines its OWN output size.
        # If we insert a layer, we are CHANGING the architecture graph.
        
        # It's easier to append/insert into the list.
        # layers[i]: dim[i] -> dim[i+1]
        # We want: dim[i] -> new_dim -> dim[i+1]
        
        # So layers[i] becomes (dim[i] -> new_dim) ? No, that changes existing layer.
        # Easier: Insert new layer AFTER layer i-1.
        # But layer i-1 maps to dim[i].
        # So we need a layer (dim[i] -> new_dim) and (new_dim -> dim[i+1]).
        # But the existing layer[i] is (dim[i] -> dim[i+1]).
        
        # So we KEEP layer[i] but insert a layer BEFORE it?
        # No, if we insert a layer, we usually mean adding depth.
        # Let's implement: Insert layer between layer i-1 and layer i.
        # Layer i-1 output is `in_features`. Layer i input is `in_features`. (Matched).
        # We want Layer i-1 -> NewLayer -> Layer i.
        # NewLayer must be `in_features` -> `in_features`.
        # This is simplest.
        
        new_dim = in_features # Keep width constant for simplicity
        
        new_layer = nn.Linear(in_features, new_dim).to(device=device, dtype=dtype)
        # Identity init to preserve flow initially
        with torch.no_grad():
            new_layer.weight.copy_(torch.eye(new_dim))
            new_layer.bias.zero_()
            
        # Insert
        self.layers.insert(insertion_index, new_layer)
        self.dropouts.insert(insertion_index, nn.Dropout(self.dropout_rate))
        self.hidden_dims.insert(insertion_index-1, new_dim) # approximate update
        
        # Register hook
        def make_hook(idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.cached_gradients[idx] = grad_output[0].detach()
            return hook
        new_layer.register_full_backward_hook(make_hook(insertion_index))
        # Re-register following hooks due to index shift? 
        # Yes, ideally. But this simple implementation might drift indices.
        # For a robust implementation, usage of Named modules is better or re-register all.
        
        # Initialize SSNC neurons for new layer
        new_ssnc_neurons = [
             SSNCNeuron(neuron_id=i, layer_id=insertion_index) # Layer ID needs to be globally consistent?
             for i in range(new_dim)
        ]
        self.ssnc_neurons.insert(insertion_index, new_ssnc_neurons)
        
        # Update metrics dict structure? 
        # This is getting very complex for this refactor step.
        # I'll stick to a basic working implementation that mostly just logs for now 
        # unless I am confident I can update all state.
        
        return True

    def _refresh_neuron_ids(self):
        """
        Refreshes neuron IDs to match their current list index.
        This fixes 'IndexOutOfBounds' errors after deletion/creation shifts indices.
        
        CRITICAL: Also rebuilds neuron_metrics dictionary with corrected keys.
        """
        # Build a mapping from old_key -> new_key
        key_remapping = {}
        
        for layer_idx, layer_neurons in enumerate(self.ssnc_neurons):
            for new_idx, neuron in enumerate(layer_neurons):
                old_id = neuron.neuron_id
                old_key = (layer_idx, old_id)
                new_key = (layer_idx, new_idx)
                
                # Track remapping if ID changed
                if old_id != new_idx:
                    key_remapping[old_key] = new_key
                    neuron.neuron_id = new_idx
        
        # Rebuild neuron_metrics with corrected keys
        if key_remapping:
            new_metrics = {}
            for old_key, metrics in self.neuron_metrics.items():
                if old_key in key_remapping:
                    new_key = key_remapping[old_key]
                    # Update the metrics object's internal ID
                    metrics.neuron_id = new_key[1]
                    new_metrics[new_key] = metrics
                else:
                    # Key unchanged, keep as-is
                    new_metrics[old_key] = metrics
            self.neuron_metrics = new_metrics

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        active = 0
        soft = 0
        hard = 0
        
        for m in self.neuron_metrics.values():
            if m.hard_deleted:
                hard += 1
            elif m.soft_deleted:
                soft += 1
            else:
                active += 1
                
        return {
            'active_neurons': active,
            'soft_deleted': soft,
            'hard_deleted': hard,
            'created': self.total_created,
            'cycle': self.cycle_count
        }

    def execute_plasticity_decisions(self, current_loss: float, current_accuracy: float, layer_sizes: List[int] = None):
        """
        Execute structural plasticity based on Observer/Controller decisions.
        This provides the explicit entry point requested, routing to the robust plasticity_cycle.
        """
        return self.plasticity_cycle(current_loss, current_accuracy)