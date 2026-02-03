"""
Phased SSNC Observer - Three-Phase Training Methodology

Phases:
1. GROWTH: Build capacity → Reach target accuracy (liberal creation)
2. OPTIMIZATION: Fine-tune → Find peak accuracy (balanced)
3. PRUNING: Remove redundancy → Minimize while maintaining accuracy (aggressive deletion)

Key Principle: First optimize accuracy, THEN optimize efficiency.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Phase(Enum):
    GROWTH = "GROWTH"
    OPTIMIZATION = "OPTIMIZATION"
    PRUNING = "PRUNING"


# Phase-specific configurations
PHASE_CONFIGS = {
    Phase.GROWTH: {
        'target_accuracy': 95.0,        # Higher target for Phase 2 transition
        'min_epochs': 10,               # Minimum time in phase
        'stability_epochs': 3,          # Epochs to confirm stability
        
        # Creation: LIBERAL
        'creation_threshold': 0.20,     # Top 20%
        'creation_budget': 15,          # Generous
        
        # Deletion: MINIMAL
        'deletion_threshold': 0.05,     # Bottom 5% only
        'deletion_budget': 5,           # Very conservative
        
        # Triggers
        'create_on_avg_utility_above': 0.50,
        'create_on_class_acc_below': 50.0,
    },
    Phase.OPTIMIZATION: {
        'stability_epochs': 5,          # Epochs without improvement
        'stability_threshold': 0.005,   # Accuracy std < 0.5%
        
        # Creation: MODERATE
        'creation_threshold': 0.30,     # Top 10%
        'creation_budget': 8,           # Moderate
        
        # Deletion: MODERATE
        'deletion_threshold': 0.15,     # Bottom 15%
        'deletion_budget': 8,           # Moderate
    },
    Phase.PRUNING: {
        'accuracy_tolerance': 0.02,     # 2% drop allowed
        'max_rollbacks': 3,             # Safety limit
        'min_neurons_ratio': 0.5,       # Don't prune below 50% of peak
        
        # Creation: EMERGENCY ONLY
        'creation_threshold': 0.45,     # Top 5%
        'creation_budget': 3,           # Minimal
        
        # Deletion: AGGRESSIVE
        'deletion_threshold': 0.25,     # Bottom 25%
        'deletion_budget': 20,          # Aggressive
        'deletion_batch_size': 5,       # Test after each batch
        
        # Adaptive budget
        'budget_increase_on_success': 2,
        'budget_decrease_on_failure': 0.5,
    }
}


class PhasedSSNCObserver:
    """
    Three-phase SSNC observer for accuracy-first, then efficiency training.
    
    Phase 1 (GROWTH): Liberal creation, minimal deletion → reach target accuracy
    Phase 2 (OPTIMIZATION): Balanced changes → find peak accuracy
    Phase 3 (PRUNING): Aggressive deletion with safety → minimize while maintaining
    """
    
    def __init__(
        self,
        initial_neurons: int,
        target_accuracy: float = 75.0,
        min_neurons_ratio: float = 0.5,
        max_neurons_ratio: float = 1.5
    ):
        self.initial_neurons = initial_neurons
        self.target_accuracy = target_accuracy
        self.min_neurons = int(initial_neurons * min_neurons_ratio)
        self.max_neurons = int(initial_neurons * max_neurons_ratio)
        
        # Phase state
        self.phase = Phase.GROWTH
        self.epochs_in_phase = 0
        self.phase_start_epoch = 0
        
        # Accuracy tracking
        self.best_accuracy = 0.0
        self.best_accuracy_epoch = 0
        self.epochs_since_improvement = 0
        self.accuracy_history = deque(maxlen=20)
        self.per_class_accuracy_history = deque(maxlen=10)
        
        # Structure tracking
        self.neuron_count_history = deque(maxlen=50)
        self.peak_neurons = initial_neurons
        
        # Phase 3 specific
        self.safe_accuracy = 0.0  # Set when entering Phase 3
        self.rollback_count = 0
        self.current_deletion_budget = PHASE_CONFIGS[Phase.PRUNING]['deletion_budget']
        self.current_deletion_threshold = PHASE_CONFIGS[Phase.PRUNING]['deletion_threshold']
        
        # Utility tracking
        self.utility_history = deque(maxlen=20)
        
        # Decision log
        self.decision_log = []
        self.phase_transitions = []
        
    def get_config(self) -> Dict:
        """Get current phase configuration."""
        return PHASE_CONFIGS[self.phase].copy()
    
    def analyze(
        self,
        current_epoch: int,
        current_accuracy: float,
        per_class_accuracy: List[float],
        current_loss: float,
        active_neurons: int,
        utilities: Dict[Tuple[int, int], float],
        cycle: int = 0
    ) -> Dict:
        """
        Analyze network state and return phase-aware decisions.
        
        Returns:
            Dict with:
            - should_create, max_creations
            - should_delete, max_deletions  
            - phase_transition (bool)
            - new_phase (if transitioning)
            - reasoning (list of strings)
        """
        # Update tracking
        self.accuracy_history.append(current_accuracy)
        self.per_class_accuracy_history.append(per_class_accuracy)
        self.neuron_count_history.append(active_neurons)
        self.epochs_in_phase += 1
        
        # Track best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.best_accuracy_epoch = current_epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Track peak neurons (for Phase 3 minimum)
        self.peak_neurons = max(self.peak_neurons, active_neurons)
        
        # Initialize decision
        config = self.get_config()
        decision = {
            'should_create': False,
            'max_creations': 0,
            'should_delete': False,
            'max_deletions': 0,
            'phase_transition': False,
            'new_phase': None,
            'reasoning': [f"Phase: {self.phase.value}, Epoch {current_epoch}"],
            'creation_threshold': config['creation_threshold'],
            'deletion_threshold': config['deletion_threshold'],
            'save_checkpoint': False,
            'rollback_requested': False,
        }
        
        # Check phase transition
        transition = self._check_phase_transition(current_accuracy, active_neurons)
        if transition:
            decision['phase_transition'] = True
            decision['new_phase'] = transition
            decision['save_checkpoint'] = True
            decision['reasoning'].append(f"🔄 PHASE TRANSITION: {self.phase.value} → {transition.value}")
            return decision
        
        # Phase-specific analysis
        if self.phase == Phase.GROWTH:
            self._analyze_growth_phase(decision, utilities, current_accuracy, per_class_accuracy, active_neurons)
        elif self.phase == Phase.OPTIMIZATION:
            self._analyze_optimization_phase(decision, utilities, current_accuracy, active_neurons)
        else:  # PRUNING
            self._analyze_pruning_phase(decision, utilities, current_accuracy, active_neurons)
        
        # Log decision
        self.decision_log.append({
            'epoch': current_epoch,
            'phase': self.phase.value,
            'accuracy': current_accuracy,
            'neurons': active_neurons,
            'decision': decision.copy()
        })
        
        return decision
    
    def _check_phase_transition(self, current_accuracy: float, active_neurons: int) -> Optional[Phase]:
        """Check if we should transition to a new phase."""
        config = self.get_config()
        
        if self.phase == Phase.GROWTH:
            # Transition to OPTIMIZATION when target accuracy reached and stable
            min_epochs = config.get('min_epochs', 10)
            stability_epochs = config.get('stability_epochs', 3)
            target = config.get('target_accuracy', self.target_accuracy)
            
            if self.epochs_in_phase >= min_epochs:
                if current_accuracy >= target:
                    # Check stability
                    if len(self.accuracy_history) >= stability_epochs:
                        recent = list(self.accuracy_history)[-stability_epochs:]
                        acc_std = np.std(recent)
                        if acc_std < 1.0:  # Stable within 1%
                            return Phase.OPTIMIZATION
        
        elif self.phase == Phase.OPTIMIZATION:
            # Transition to PRUNING when peak accuracy found and stable
            stability_epochs = config.get('stability_epochs', 5)
            stability_threshold = config.get('stability_threshold', 0.005)
            
            if self.epochs_since_improvement >= stability_epochs:
                if len(self.accuracy_history) >= stability_epochs:
                    recent = list(self.accuracy_history)[-stability_epochs:]
                    acc_std = np.std(recent)
                    if acc_std < stability_threshold * 100:  # Convert to percentage
                        if current_accuracy >= self.target_accuracy * 1.05:  # ~90%+ for transition
                            return Phase.PRUNING
        
        return None
    
    def transition_to(self, new_phase: Phase):
        """Execute phase transition."""
        old_phase = self.phase
        
        # Record transition
        self.phase_transitions.append({
            'from': old_phase.value,
            'to': new_phase.value,
            'epoch': self.epochs_in_phase,
            'accuracy': self.best_accuracy,
            'neurons': list(self.neuron_count_history)[-1] if self.neuron_count_history else 0
        })
        
        # Update phase
        self.phase = new_phase
        self.epochs_in_phase = 0
        
        # Phase-specific initialization
        if new_phase == Phase.PRUNING:
            # Set accuracy floor for rollback
            self.safe_accuracy = self.best_accuracy - (PHASE_CONFIGS[Phase.PRUNING]['accuracy_tolerance'] * 100)
            self.rollback_count = 0
            self.current_deletion_budget = PHASE_CONFIGS[Phase.PRUNING]['deletion_budget']
            self.current_deletion_threshold = PHASE_CONFIGS[Phase.PRUNING]['deletion_threshold']
        
        print(f"\n{'='*60}")
        print(f"🔄 PHASE TRANSITION: {old_phase.value} → {new_phase.value}")
        print(f"   Best accuracy: {self.best_accuracy:.2f}%")
        if new_phase == Phase.PRUNING:
            print(f"   Safe accuracy floor: {self.safe_accuracy:.2f}%")
        print(f"{'='*60}\n")
    
    def _analyze_growth_phase(
        self,
        decision: Dict,
        utilities: Dict,
        current_accuracy: float,
        per_class_accuracy: List[float],
        active_neurons: int
    ):
        """Phase 1: Liberal creation, minimal deletion."""
        config = self.get_config()
        reasoning = decision['reasoning']
        
        utility_values = list(utilities.values())
        if not utility_values:
            return
        
        utility_array = np.array(utility_values)
        mean_utility = np.mean(utility_array)
        
        # === LIBERAL CREATION ===
        should_create = False
        create_reason = ""
        
        # Trigger 1: High average utility (network overloaded)
        if mean_utility > config['create_on_avg_utility_above']:
            should_create = True
            create_reason = f"High avg utility ({mean_utility:.3f} > {config['create_on_avg_utility_above']})"
        
        # Trigger 2: Any class performing poorly
        min_class_acc = min(per_class_accuracy) if per_class_accuracy else 100
        if min_class_acc < config['create_on_class_acc_below']:
            should_create = True
            worst_class = per_class_accuracy.index(min_class_acc)
            create_reason = f"Class {worst_class} struggling ({min_class_acc:.1f}% < {config['create_on_class_acc_below']}%)"
        
        # Trigger 3: Accuracy plateau (loss not improving)
        if len(self.accuracy_history) >= 5:
            recent_acc = list(self.accuracy_history)[-5:]
            if np.std(recent_acc) < 0.5 and current_accuracy < self.target_accuracy:
                should_create = True
                create_reason = f"Accuracy plateau at {current_accuracy:.1f}%"
        
        if should_create and active_neurons < self.max_neurons:
            decision['should_create'] = True
            decision['max_creations'] = config['creation_budget']
            reasoning.append(f"✨ GROWTH CREATE: {create_reason} → +{config['creation_budget']}")
        
        # === MINIMAL DELETION ===
        # Only delete truly dead neurons (utility < 0.05 consistently)
        n_dead = np.sum(utility_array < config['deletion_threshold'])
        
        if n_dead > 0 and active_neurons > self.min_neurons:
            # Very conservative deletion
            n_to_delete = min(config['deletion_budget'], n_dead)
            decision['should_delete'] = True
            decision['max_deletions'] = n_to_delete
            reasoning.append(f"🗑️ GROWTH DELETE: {n_dead} dead neurons → -{n_to_delete}")
    
    def _analyze_optimization_phase(
        self,
        decision: Dict,
        utilities: Dict,
        current_accuracy: float,
        active_neurons: int
    ):
        """Phase 2: Balanced creation and deletion for peak accuracy."""
        config = self.get_config()
        reasoning = decision['reasoning']
        
        utility_values = list(utilities.values())
        if not utility_values:
            return
        
        utility_array = np.array(utility_values)
        
        # === BALANCED CREATION ===
        n_overloaded = np.sum(utility_array > config['creation_threshold'])
        
        if n_overloaded > len(utility_array) * 0.1:  # >10% overloaded
            if active_neurons < self.max_neurons:
                n_to_create = min(config['creation_budget'], int(n_overloaded * 0.1))
                if n_to_create > 0:
                    decision['should_create'] = True
                    decision['max_creations'] = n_to_create
                    reasoning.append(f"✨ OPT CREATE: {n_overloaded} overloaded → +{n_to_create}")
        
        # === BALANCED DELETION ===
        n_useless = np.sum(utility_array < config['deletion_threshold'])
        
        if n_useless > 0 and active_neurons > self.min_neurons:
            n_to_delete = min(config['deletion_budget'], n_useless)
            decision['should_delete'] = True
            decision['max_deletions'] = n_to_delete
            reasoning.append(f"🗑️ OPT DELETE: {n_useless} low utility → -{n_to_delete}")
        
        # === SAVE CHECKPOINT ON IMPROVEMENT ===
        if self.epochs_since_improvement == 0:
            decision['save_checkpoint'] = True
            reasoning.append(f"💾 New best accuracy: {current_accuracy:.2f}%")
    
    def _analyze_pruning_phase(
        self,
        decision: Dict,
        utilities: Dict,
        current_accuracy: float,
        active_neurons: int
    ):
        """Phase 3: Aggressive deletion with accuracy protection."""
        config = self.get_config()
        reasoning = decision['reasoning']
        
        # Check if we've hit rollback limit
        if self.rollback_count >= config['max_rollbacks']:
            reasoning.append(f"⚠️ Max rollbacks reached ({self.rollback_count}). Pausing pruning.")
            return
        
        utility_values = list(utilities.values())
        if not utility_values:
            return
        
        utility_array = np.array(utility_values)
        
        # === CHECK ACCURACY PROTECTION ===
        if current_accuracy < self.safe_accuracy:
            # Accuracy dropped too much - need recovery, not more deletion
            decision['should_create'] = True
            decision['max_creations'] = config['creation_budget']
            reasoning.append(f"🚨 EMERGENCY: Accuracy {current_accuracy:.1f}% < safe {self.safe_accuracy:.1f}%")
            reasoning.append(f"🚨 Creating {config['creation_budget']} emergency neurons")
            return
        
        # === MINIMUM NEURONS CHECK ===
        min_allowed = int(self.peak_neurons * config['min_neurons_ratio'])
        if active_neurons <= min_allowed:
            reasoning.append(f"⚠️ At minimum neurons ({active_neurons} <= {min_allowed}). Stopping pruning.")
            return
        
        # === AGGRESSIVE DELETION ===
        n_candidates = np.sum(utility_array < self.current_deletion_threshold)
        
        if n_candidates > 0:
            # Use adaptive budget
            n_to_delete = min(self.current_deletion_budget, n_candidates)
            
            # Limit to batch size for gradual testing
            batch_size = config['deletion_batch_size']
            n_to_delete = min(n_to_delete, batch_size)
            
            decision['should_delete'] = True
            decision['max_deletions'] = n_to_delete
            decision['gradual_delete'] = True  # Signal to use gradual deletion
            decision['deletion_batch_size'] = batch_size
            
            reasoning.append(f"🔥 PRUNE DELETE: {n_candidates} candidates → -{n_to_delete} (batch {batch_size})")
            reasoning.append(f"   Budget: {self.current_deletion_budget}, Threshold: {self.current_deletion_threshold:.3f}")
    
    def report_deletion_result(self, success: bool, accuracy_after: float):
        """Report result of deletion batch for adaptive budget adjustment."""
        if self.phase != Phase.PRUNING:
            return
        
        config = PHASE_CONFIGS[Phase.PRUNING]
        
        if success and accuracy_after >= self.safe_accuracy:
            # Safe deletion - can be more aggressive
            self.current_deletion_budget = min(
                25,  # Max budget
                self.current_deletion_budget + config['budget_increase_on_success']
            )
        else:
            # Problematic deletion - be more conservative
            self.current_deletion_budget = max(
                5,  # Min budget
                int(self.current_deletion_budget * config['budget_decrease_on_failure'])
            )
            self.current_deletion_threshold += 0.05  # Raise threshold
            self.rollback_count += 1
    
    def get_phase_summary(self) -> Dict:
        """Get summary of current phase state."""
        return {
            'phase': self.phase.value,
            'epochs_in_phase': self.epochs_in_phase,
            'best_accuracy': self.best_accuracy,
            'best_accuracy_epoch': self.best_accuracy_epoch,
            'epochs_since_improvement': self.epochs_since_improvement,
            'safe_accuracy': self.safe_accuracy if self.phase == Phase.PRUNING else None,
            'rollback_count': self.rollback_count if self.phase == Phase.PRUNING else None,
            'phase_transitions': self.phase_transitions
        }


# Export configurations
__all__ = ['PhasedSSNCObserver', 'Phase', 'PHASE_CONFIGS']
