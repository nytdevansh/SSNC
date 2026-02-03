"""
HIGH-ACCURACY Observer - Optimized for 95%+ Performance

Key Changes from Original:
1. Raises convergence threshold from 88% to 94% (allow reaching 95%+)
2. More lenient during early training (allow network to learn)
3. Only prunes aggressively after high accuracy is reached
4. Smarter regime detection for high-accuracy scenarios
"""

import numpy as np
from collections import deque
from typing import Dict, List

class HighAccuracyObserver:
    """
    Observer optimized for reaching 95%+ accuracy on TREC-6.
    """
    
    def __init__(
        self,
        initial_neurons: int,
        min_capacity_ratio: float = 0.5,  # Less aggressive minimum (was 0.3)
        max_capacity_ratio: float = 2.0,  # Allow more growth (was 1.5)
        adaptation_rate: float = 0.1,
        target_accuracy: float = 95.0  # NEW: Target accuracy goal
    ):
        self.initial_neurons = initial_neurons
        self.min_neurons = int(initial_neurons * min_capacity_ratio)
        self.max_neurons = int(initial_neurons * max_capacity_ratio)
        self.adaptation_rate = adaptation_rate
        self.target_accuracy = target_accuracy
        
        # Adaptive thresholds (More lenient initially)
        self.deletion_percentile = 15  # Less aggressive (was 20)
        self.creation_percentile = 85  # Less selective (was 90)
        self.deletion_threshold = 0.12   # Lower initial (was 0.15)
        self.creation_threshold = 0.28   # Lower initial (was 0.30)
        
        # History tracking
        self.loss_history = deque(maxlen=20)
        self.cycle_count = 0
        self.accuracy_history = deque(maxlen=20)
        self.utility_distribution_history = deque(maxlen=10)
        self.neuron_count_history = deque(maxlen=50)
        self.structural_changes = deque(maxlen=100)
        
        # Decision log
        self.decision_log = []
        
        # State
        self.is_stable = True
        self.trend = 'neutral'
        
        # Stability
        self.last_deletion_step: Dict[int, int] = {}
        self.deletion_cooldown: int = 200
        
        # NEW: High accuracy tracking
        self.reached_high_accuracy = False
        self.high_accuracy_threshold = 92.0  # Start being more conservative at 92%
        
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
        Analyze network state with focus on reaching high accuracy.
        """
        self.loss_history.append(current_loss)
        self.neuron_count_history.append(active_neurons)
        if current_accuracy is not None:
            self.accuracy_history.append(current_accuracy)
            
        self.cycle_count = cycle
        
        # Track if we've reached high accuracy
        if current_accuracy and current_accuracy >= self.high_accuracy_threshold:
            self.reached_high_accuracy = True

        # === SAFETY BRAKE (Only after reaching decent accuracy) ===
        allow_deletion = True
        reasoning_prefix = []
        
        # More lenient safety brake - only activate if we're doing well
        avg_acc = np.mean(list(self.accuracy_history)[-10:]) if len(self.accuracy_history) >= 10 else 0.0
        
        if avg_acc > 85.0 and len(self.loss_history) > 5:
            recent_losses = list(self.loss_history)[-5:]
            recent_min = min(recent_losses)
            if current_loss > 1.2 * recent_min:  # More lenient (was 1.15)
                allow_deletion = False
                reasoning_prefix.append("🛑 SAFETY BRAKE: Loss spike detected.")
        
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
        
        # === ADAPT THRESHOLDS ===
        current_del = float(np.percentile(utility_array, self.deletion_percentile))
        current_create = float(np.percentile(utility_array, self.creation_percentile))
        
        # Smooth updates
        self.deletion_threshold = 0.7 * self.deletion_threshold + 0.3 * current_del
        self.creation_threshold = 0.7 * self.creation_threshold + 0.3 * current_create
        
        # Calculate pressures
        n_overloaded = np.sum(utility_array > self.creation_threshold)
        n_useless = np.sum(utility_array < self.deletion_threshold)
        total_n = len(utility_array)
        creation_pressure = n_overloaded / total_n if total_n > 0 else 0
        deletion_pressure = n_useless / total_n if total_n > 0 else 0
        
        # Debug output
        if self.cycle_count % 10 == 0:
            print(f"🎯 Thresholds: Del<{self.deletion_threshold:.3f}, "
                  f"Create>{self.creation_threshold:.3f}, "
                  f"Pressures: D={deletion_pressure:.2%}, C={creation_pressure:.2%}")
        
        # Calculate performance metrics
        acc_std = 0.0
        loss_std = 0.0
        if len(self.accuracy_history) >= 10:
            rec_acc = list(self.accuracy_history)[-10:]
            avg_acc = np.mean(rec_acc)
            acc_std = np.std(rec_acc)
        if len(self.loss_history) >= 10:
            rec_loss = list(self.loss_history)[-10:]
            loss_std = np.std(rec_loss)

        # === DIAGNOSE REGIME (UPDATED FOR HIGH ACCURACY) ===
        regime = 'LEARNING'  # Default: still learning
        reasoning = []
        
        # CRITICAL FIX: Only converge at 94%+ (was 88%)
        if avg_acc >= 94.0 and acc_std < 0.3:
            regime = 'CONVERGING'
            reasoning.append(f"DIAGNOSIS: {regime} - Target reached! (Acc={avg_acc:.1f}%)")
        elif loss_std > 0.4 and len(self.loss_history) >= 10:
            regime = 'OSCILLATING'
        elif avg_acc >= self.high_accuracy_threshold and acc_std < 1.0:
            regime = 'HIGH_ACCURACY'  # NEW: Maintain high accuracy
            reasoning.append(f"DIAGNOSIS: {regime} (Acc={avg_acc:.1f}%) - Optimizing")
        elif loss_std > 0.6:
            regime = 'UNSTABLE'
        elif avg_acc < 40.0 and cycle > 50:
            regime = 'UNDERFITTING_SEVERE'
        elif avg_acc < 70.0 and cycle > 30:
            regime = 'UNDERFITTING'
        else:
            reasoning.append(f"DIAGNOSIS: {regime} (Acc={avg_acc:.1f}%, Loss={current_loss:.3f})")
        
        # === SET POLICY (OPTIMIZED FOR HIGH ACCURACY) ===
        create_mult = 1.0  # More generous default
        delete_mult = 1.0
        thresh_adjust = 0.0
        should_stop = False
        reduce_lr = False
        
        if regime == 'CONVERGING':
            # Only stop if we're at target
            if avg_acc >= self.target_accuracy:
                should_stop = True
                reasoning.append(f"ACTION: TARGET REACHED ({avg_acc:.1f}% >= {self.target_accuracy}%)")
            else:
                # Close but not there yet - fine-tune
                create_mult = 0.2
                delete_mult = 0.5
                reasoning.append("ACTION: NEAR TARGET - Fine-tuning")
                
        elif regime == 'HIGH_ACCURACY':
            # At high accuracy - be conservative with changes
            create_mult = 0.3
            delete_mult = 0.7
            reasoning.append("ACTION: HIGH_ACCURACY - Conservative maintenance")
            
        elif regime == 'OSCILLATING':
            create_mult = 0.0
            delete_mult = 0.0
            reduce_lr = True
            reasoning.append("ACTION: OSCILLATING - Freezing structure")
            
        elif regime == 'UNSTABLE':
            create_mult = 0.0
            delete_mult = 0.2  # Gentle pruning only
            reasoning.append("ACTION: UNSTABLE - Gentle pruning")
            
        elif regime == 'UNDERFITTING_SEVERE':
            # Need more capacity
            create_mult = 2.0
            delete_mult = 0.2
            reasoning.append("ACTION: SEVERE UNDERFITTING - Growing network")
            
        elif regime == 'UNDERFITTING':
            # Moderate growth needed
            create_mult = 1.5
            delete_mult = 0.5
            reasoning.append("ACTION: UNDERFITTING - Moderate growth")
            
        else:  # LEARNING
            # Normal learning - balanced approach
            create_mult = 1.0
            delete_mult = 0.8
            reasoning.append("ACTION: LEARNING - Balanced plasticity")
            
        # === EXECUTE DECISIONS ===
        decisions = {
            'should_delete': False, 'max_deletions': 0,
            'should_create': False, 'max_creations': 0,
            'should_add_layer': False, 'should_stop': should_stop,
            'prune_layer_idx': -1, 'reasoning': reasoning,
            'deletion_threshold': self.deletion_threshold,
            'creation_threshold': self.creation_threshold + thresh_adjust,
            'reduce_lr': reduce_lr,
            'plasticity_cooldown': 10 if (create_mult > 1.5 or delete_mult > 1.5) else 0
        }
        
        if should_stop:
            return decisions

        # Capacity checks
        if active_neurons >= self.max_neurons:
            reasoning.append(f"⚠️ CAPACITY: At max neurons ({self.max_neurons})")
            create_mult = 0.0
        
        if active_neurons <= self.min_neurons:
            reasoning.append(f"⚠️ CAPACITY: At min neurons ({self.min_neurons})")
            delete_mult = 0.0

        # === DELETION DECISION ===
        eff_del_pressure = deletion_pressure * delete_mult
        
        if eff_del_pressure > 0.02 and allow_deletion:  # Higher threshold
            # Gentler deletion
            count = max(1, int(n_useless * 0.10 * delete_mult))
            decisions['should_delete'] = True
            decisions['max_deletions'] = count
            decisions['reasoning'].append(f"PRUNE: {n_useless} useless * 10% * {delete_mult:.1f} = {count}")
        elif not allow_deletion and eff_del_pressure > 0.02:
            decisions['reasoning'].extend(reasoning_prefix)
            
        # === CREATION DECISION ===
        eff_create_pressure = creation_pressure * create_mult
        
        # More aggressive creation when needed
        if eff_create_pressure > 0.05:  # Lower threshold
            base_count = int(n_overloaded * 0.15) if n_overloaded > 0 else 0
            count = max(1, int(base_count * create_mult))
            
            # Less restrictive caps
            max_growth_cap = max(3, int(active_neurons * 0.01))  # 1% instead of 0.5%
            count = min(count, max_growth_cap)
            
            # Create if accuracy is below target
            if avg_acc < self.target_accuracy:
                decisions['should_create'] = True
                decisions['max_creations'] = count
                decisions['reasoning'].append(f"CREATE: Acc {avg_acc:.1f}% < {self.target_accuracy}%, adding {count}")
            else:
                decisions['reasoning'].append(f"SKIP CREATE: Acc {avg_acc:.1f}% >= {self.target_accuracy}%")
        
        # === BALANCE ENFORCEMENT (Less strict) ===
        if decisions['should_delete'] and decisions['should_create']:
            # Only need deletions >= 30% of creations (was 50%)
            min_deletions = int(0.3 * decisions['max_creations'])
            if decisions['max_deletions'] < min_deletions:
                max_allowed_creations = int(3.0 * decisions['max_deletions'])
                decisions['max_creations'] = min(decisions['max_creations'], max_allowed_creations)
                decisions['reasoning'].append(f"BALANCE: Adjusted to {decisions['max_creations']} creations")
        
        # === ABSOLUTE CAPS ===
        if decisions['max_creations'] > 15:  # Higher cap (was 8)
            decisions['max_creations'] = 15
            decisions['reasoning'].append("CAP: Limit 15 creations/cycle")
        
        if decisions['max_deletions'] > 20:  # Lower cap (was 25)
            decisions['max_deletions'] = 20
            decisions['reasoning'].append("CAP: Limit 20 deletions/cycle")
        
        # Log structural changes
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
    
    def _safe_default(self):
        """Return safe default decision."""
        return {
            'should_delete': False,
            'max_deletions': 0,
            'should_create': False,
            'max_creations': 0,
            'should_add_layer': False,
            'should_stop': False,
            'prune_layer_idx': -1,
            'reasoning': ['No utilities available'],
            'deletion_threshold': self.deletion_threshold,
            'creation_threshold': self.creation_threshold,
            'reduce_lr': False,
            'plasticity_cooldown': 0
        }


# Alias for backward compatibility with existing codebase
AdaptiveObserver = HighAccuracyObserver

