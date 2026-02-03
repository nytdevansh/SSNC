"""
Patch for observer.py - Add percentile-based adaptive thresholds

Apply these changes to your observer.py file
"""

# ============================================
# CHANGE 1: In __init__ method (line 46-48)
# ============================================

# REPLACE:
#     self.deletion_threshold = 0.05
#     self.creation_threshold = 0.25

# WITH:
        # Percentile-based adaptive thresholds
        self.deletion_percentile = 15  # Bottom 15%
        self.creation_percentile = 85  # Top 15%
        self.deletion_threshold = 0.05  # Initial (will adapt)
        self.creation_threshold = 0.25  # Initial (will adapt)


# ============================================
# CHANGE 2: In analyze_network_state (around line 126-137)
# ============================================

# FIND this section:
#     # === ADAPT THRESHOLDS BASED ON DISTRIBUTION ===
#     # Deletion: target bottom 5-10% IF they're truly low
#     p10 = np.percentile(utility_array, 10)
#     p90 = np.percentile(utility_array, 90)
#     mean_utility = np.mean(utility_array)
#     
#     # Calculate Pressure (Organic Demand)
#     n_overloaded = np.sum(utility_array > self.creation_threshold)
#     n_useless = np.sum(utility_array < self.deletion_threshold)

# REPLACE WITH:

        # === ADAPT THRESHOLDS BASED ON DISTRIBUTION ===
        # Use percentiles for truly adaptive thresholds
        self.deletion_threshold = float(np.percentile(utility_array, self.deletion_percentile))
        self.creation_threshold = float(np.percentile(utility_array, self.creation_percentile))
        
        p10 = float(np.percentile(utility_array, 10))
        p90 = float(np.percentile(utility_array, 90))
        mean_utility = float(np.mean(utility_array))
        
        # Calculate Pressure (Organic Demand) using ADAPTIVE thresholds
        n_overloaded = np.sum(utility_array > self.creation_threshold)
        n_useless = np.sum(utility_array < self.deletion_threshold)
        total_n = len(utility_array)
        creation_pressure = n_overloaded / total_n if total_n > 0 else 0
        deletion_pressure = n_useless / total_n if total_n > 0 else 0
        
        # Debug output
        if self.cycle_count % 10 == 0:  # Print every 10 cycles
            print(f"🎯 Adaptive Thresholds: Del<{self.deletion_threshold:.3f}, "
                  f"Create>{self.creation_threshold:.3f}, "
                  f"Pressures: D={deletion_pressure:.2%}, C={creation_pressure:.2%}")


# ============================================
# CHANGE 3: Add cycle counter tracking
# ============================================

# In __init__, ADD:
        self.cycle_count = 0

# In analyze_network_state, at the START, ADD:
        self.cycle_count += 1


# ============================================
# Complete __init__ should look like:
# ============================================

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
    
    # Percentile-based adaptive thresholds
    self.deletion_percentile = 15  # Bottom 15%
    self.creation_percentile = 85  # Top 15%
    self.deletion_threshold = 0.05  # Initial (will adapt)
    self.creation_threshold = 0.25  # Initial (will adapt)
    
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
    self.trend = 'neutral'
    self.cycle_count = 0  # NEW
    
    # Stability
    self.last_deletion_step: Dict[int, int] = {}
    self.deletion_cooldown: int = 200
