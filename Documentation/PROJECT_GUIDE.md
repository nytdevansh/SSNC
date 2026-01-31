# SSNC Project Guide - Complete Implementation

## Project Overview

**SSNC** (Self-Stabilizing Neuronal Control) is a decentralized learning framework where SSNC-Neurons are autonomous agents with bounded Neuron Local State and no structural authority, managed by a slow-cycle Structural Observer that conservatively regulates topology.

**Goal**: Build a network that can restructure itself during training through local utility computation, Observer-driven Structural Actions, and self-stabilizing mechanisms that prevent both Silent Failure and runaway growth.

---

## The Complete System Architecture

### Phase Progression

```
PHASE 0: Planning ✅
         └─ Define constraints, safety principles, architecture

PHASE 1: Baseline Instrumentation ✅
         └─ Observable MLP with per-neuron Neuron Local State tracking
         └─ Baseline for performance comparison

PHASE 2: SSNC-Neuron Implementation ✅
         └─ Neurons track bounded Neuron Local State (activation, variance, correlation, age, inactivity)
         └─ Compute Neuron Utility Signals via smoothed correlation
         └─ No Structural Actions yet - observation and monitoring only

PHASE 3: Utility Validation ✅
         └─ Test if Neuron Utility Signals actually predict neuron importance
         └─ Ablation testing, stability analysis, Noise Robustness
         └─ Validates foundation for Phases 4+

PHASE 4: Bounded Local Learning ✅
         └─ SSNC-Neurons adapt own weights via Bounded Local Updates
         └─ Uses Neuron Local State (activation mean/variance, error correlation)
         └─ Bounded magnitude (max_update_norm = 0.01) for stability
         └─ Utility-modulated learning rates

PHASE 5: Structural Observer ✅
         └─ Central Observer aggregates Neuron Local State on slow Observer Cycles
         └─ Computes Structural Health Metrics (utility distribution, inactivity, load balance)
         └─ Generates Structural Action decisions (log-only in PHASE 5)
         └─ Does not learn weights or directly optimize loss

PHASE 6: Soft Deletion ✅
         └─ Reversible gating of low-Utility-Signal neurons
         └─ Gradual magnitude reduction via Soft Deletion gating
         └─ Weights preserved; neuron continues monitoring Utility Signal
         └─ Allows Recovery Tests

PHASE 7: Hard Deletion ✅
         └─ Irreversible removal after prolonged Soft Deletion
         └─ Only after confirmation that Soft Deletion causes no performance loss
         └─ Reclaims computational resources

PHASE 8: Neurogenesis (Creation & Duplication) ✅
         └─ Dynamic capacity growth via Neuron Creation and Neuron Duplication
         └─ Duplicates high-Utility-Signal neurons when overloaded
         └─ New neurons start with low gate_value, gradually ungate if useful

PHASE 9: Structural Redistribution 🔧
         └─ Rebalance representational load across layers via Structural Redistribution
         └─ Framework ready, implementation pending

PHASE 10: Global Safety & Stagnation Detection ✅
         └─ Global Progress Monitor detects Silent Failure and stagnation
         └─ Prevent self-destruction via Stagnation Stop Rule
         └─ Prevent runaway growth
         └─ Stabilize oscillations via Timescale Separation enforcement
```

---

## System Components

### 1. Neuron-Level (PHASE 2, 4)

**What each SSNC-Neuron tracks (Neuron Local State)**:
```python
- activation_mean: Running average of neuron outputs
- activation_variance: Variance in outputs over observation window
- inactivity_counter: Batches since last significant firing (detects Inactive Neurons)
- error_correlation: Correlation between neuron activation and Downstream Error
- age: Time since creation (lifetime in Observer Cycles)
- neuron_utility_signal: Smoothed correlation computed over Utility Window
- gate_value: Soft Deletion mechanism (1.0 = active, 0.0 = deleted)
```

**What each SSNC-Neuron computes**:
- Observes own Neuron Local State automatically
- Computes Neuron Utility Signal every forward pass
- Performs Bounded Local Updates (PHASE 4+)
- Responds to gate modifications (Soft Deletion application)

**Safety mechanisms (Structural Invariance Constraints)**:
- Gradual gating (not instant deletion via Soft Deletion)
- Hysteresis (wait before reversing decisions via marking cycles)
- Age protection (don't delete young neurons via grace periods)
- Bounded magnitude updates (max update norm = 0.01)

### 2. Structural Observer Level (PHASE 5, 6, 7, 8)

**What the Observer does on Observer Cycles**:
```python
1. Aggregate all Neuron Local State
2. Compute Structural Health Metrics:
   - Neuron Utility Signal distribution (variance, mean)
   - Inactive Neuron count (detection via inactivity_counter)
   - Layer load balance (output magnitude across layers)
   - Overload detection (neurons at capacity limits)
3. Generate Structural Actions:
   - Identify Candidates for Deletion (low utility, sustained inactivity)
   - Identify neurons for Neuron Creation or Duplication
   - Rebalancing recommendations (Structural Redistribution)
4. Execute conservatively:
   - Soft Deletion before Hard Deletion (reversibility-first)
   - Limited actions per cycle (prevent cascading failures)
   - Logging for transparency and auditing
```

**Conservative by design via Timescale Separation**:
- Observer operates on slow cycles (50-200 batches apart)
- Multiple Observer cycles before Hard Deletion
- Soft Deletion → monitor → Hard Deletion pipeline
- Capacity budgets prevent runaway growth (Stagnation Stop Rule)

### 3. Structural Manager (PHASE 6-10)

**Responsibilities**:
- Execute Soft Deletion decisions: apply gate values, reduce output magnitude
- Execute Hard Deletion decisions: permanently remove neurons, reclaim resources
- Execute Neuron Creation/Duplication: allocate new capacity
- Maintain Structural Invariance Constraints
- Manage capacity within hard bounds (min/max neurons per layer)
- Detect and prevent Silent Failure (via Global Progress Monitor)
- Track Structural Health Metrics for analysis and recovery
- Log all Structural Actions for debugging

---

## How It All Works Together

### Training Cycle

```
Epoch:
  Batch Loop:
    1. Forward pass
       └─ Neurons compute activations
       └─ Observer collects statistics
       
    2. Backward pass (standard SGD)
       └─ Compute global gradients
       └─ Update all weights
       
    3. Local learning (PHASE 4+)
       └─ Each neuron updates own weights
       └─ Utility-modulated learning rates
       └─ Clipped updates for safety
       
    4. Observer cycle (every N batches)
       └─ Compute structural health
       └─ Identify problem neurons
       └─ Generate decisions
       
    5. Execution (PHASE 6+)
       └─ Apply soft deletions (gate neurons)
       └─ Apply hard deletions (remove permanently)
       └─ Create new neurons if needed
       
    6. Logging
       └─ Record all statistics
       └─ Track deletions/creations
       └─ Update health scores
```

### Key Flows

**Soft Deletion Flow**:
```
Low-utility neuron detected
    ↓
Mark for deletion (counter++)
    ↓
Wait marking_cycles cycles
    ↓
Start gating: gate_value *= (1 - gating_rate)
    ↓
Neuron output reduced to ~5% of original
    ↓
[Can recover if utility improves]
    ↓
Wait soft_deletion_cycles more
    ↓
Hard delete: Permanently remove neuron
```

**Creation Flow**:
```
High-utility neuron detected
    ↓
[Wait for maturity: age > 200]
    ↓
Check if layer overloaded
    ↓
Duplicate neuron: New = Old + small noise
    ↓
New neuron starts gated: gate_value = 0.1
    ↓
Gradually ungate if utility remains high
    ↓
[Can re-gate if utility drops]
```

---

## Configuration Deep Dive

### Critical Parameters

**1. Deletion Threshold** (Controls how aggressive deletion is)
```python
deletion_utility_threshold: float = 0.08

# How to think about it:
# - 0.02: Very conservative (low = hard to delete)
# - 0.08: Moderate (default)
# - 0.15: Aggressive (high = easy to delete)

# Utilities typically range 0.0 to 0.5
# Setting this above mean utility → most neurons safe
# Setting this at mean → normal churn
# Setting this below mean → aggressive pruning
```

**2. Marking Cycles** (How long before deletion actually happens)
```python
marking_cycles_required: int = 15

# How to think about it:
# - 5: Neurons deleted immediately (risky)
# - 15: Moderate waiting period
# - 30: Very conservative (safe, slow)

# Each cycle = plasticity_cycle_every batches
# So: 15 cycles * 50 batches = 750 total batches of observation
```

**3. Gating Rate** (How fast neurons disappear)
```python
gating_rate: float = 0.05

# How to think about it:
# - 0.01: Very slow gating (safer, more gradual)
# - 0.05: Moderate (default)
# - 0.20: Fast gating (more abrupt)

# Each cycle: gate_value *= (1.0 - gating_rate)
# So: gate_value = gate_value * 0.95 with gating_rate=0.05
# After 20 cycles: 0.95^20 = 0.36 (neuron at 36% activity)
# After 60 cycles: 0.95^60 = 0.05 (neuron at 5% activity)
```

**4. Max Deletions Per Cycle** (Prevent mass deletion)
```python
max_deletions_per_cycle: int = 2

# How to think about it:
# - 1: Very conservative (delete one at a time)
# - 2-3: Moderate (balanced)
# - 5+: Aggressive (fast adaptation)

# Safety mechanism: prevents losing too many neurons at once
```

**5. Creation Threshold** (When to create new neurons)
```python
creation_threshold: float = 0.75

# How to think about it:
# - 0.60: Create frequently (aggressive growth)
# - 0.75: Moderate (default)
# - 0.90: Rarely create (conservative)

# Only create for neurons with utility > threshold
# AND that are already mature (age > 200)
```

### Preset Configurations

**Conservative** (Safe for testing)
```python
CONFIG = {
    'deletion_threshold': 0.05,
    'marking_cycles': 20,
    'gating_rate': 0.02,
    'max_deletions_per_cycle': 1,
    'max_creations_per_cycle': 1,
    'creation_threshold': 0.85,
}
# Result: Minimal plasticity, very stable, slow
```

**Moderate** (Balanced, recommended)
```python
CONFIG = {
    'deletion_threshold': 0.08,
    'marking_cycles': 15,
    'gating_rate': 0.05,
    'max_deletions_per_cycle': 2,
    'max_creations_per_cycle': 1,
    'creation_threshold': 0.75,
}
# Result: Reasonable plasticity, stable, normal speed
```

**Aggressive** (Fast, for quick experiments)
```python
CONFIG = {
    'deletion_threshold': 0.15,
    'marking_cycles': 8,
    'gating_rate': 0.10,
    'max_deletions_per_cycle': 5,
    'max_creations_per_cycle': 2,
    'creation_threshold': 0.60,
}
# Result: High plasticity, may be unstable, fast changes
```

---

## Understanding Utility

### What Is Utility?

A composite metric predicting neuron importance, combining:

```python
utility = (
    0.2 * activation_magnitude    # How much does neuron fire?
    + 0.2 * activation_variance   # Varied or repetitive?
    + 0.4 * error_correlation     # Correlates with downstream errors?
    + 0.2 * interaction_term      # Redundant with neighbors?
) * inactivity_penalty * age_bonus
```

**In plain English**:
- High activation magnitude: Neuron is used
- High variance: Neuron responds to different inputs
- High error correlation: Changes matter for outputs
- Low redundancy: Doesn't just repeat neighbors
- No inactivity: Keeps firing (not dead)
- Mature age: Has had time to develop

### Typical Ranges

```
Distribution of utilities in a healthy network:
Mean: ~0.15-0.25
Std Dev: ~0.05-0.15
Min: ~0.00
Max: ~0.50

Red flags:
- All utilities same → No differentiation
- All very low → Network dormant
- All very high → Metric not working
- Oscillating rapidly → Instability
```

### PHASE 3: Validating Utility

Before trusting utilities for decisions, test:

**1. Ablation Test**
- Delete each neuron (zero its output)
- Measure loss increase
- Correlate with utility
- Want: Correlation > 0.3 (utility predicts importance)

**2. Stability Test**
- Run same batch 10 times
- Track utility variance
- Want: CV < 0.1 (low variance)

**3. Component Analysis**
- Break utility into parts
- Which contribute most?
- Should be: error_correlation > others

**4. Noise Robustness**
- Add input noise
- Recompute utilities
- Want: Rank correlation > 0.7

---

## Common Issues & Solutions

### Issue 1: 0 Neurons Deleted (Plasticity Inactive)

**Diagnosis**:
```python
stats = model.get_statistics()
print(f"Total deleted: {stats['total_soft_deleted']}")  # = 0 ❌
```

**Root Causes**:
1. Neuron storage broken (neurons not accessible)
2. Utilities never computed
3. Threshold too low (no neurons qualify)
4. Marking cycles too long (never reaches deletion)

**Solutions**:
1. Verify neurons stored in simple lists (not nn.ModuleList)
2. Add logging: `print(f"Utilities: {get_utilities(model)}")`
3. Raise threshold: `deletion_threshold=0.20`
4. Lower marking cycles: `marking_cycles=5`

**Prevention**: Use `actually_working_ssnc.py` (simple storage)

### Issue 2: Instability (NaN Loss, Exploding Accuracy)

**Diagnosis**:
```python
# Loss explodes to NaN within 5 epochs
# Accuracy crashes from 80% to 20%
```

**Root Causes**:
1. No gradient clipping
2. Gating rate too high
3. Fashion-MNIST parameters (needs different normalization)
4. Deletion threshold too low (too many deletions)

**Solutions**:
1. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
2. Lower gating rate: `gating_rate=0.02`
3. Use Fashion-MNIST normalization: `Normalize((0.2860,), (0.3530,))`
4. Raise threshold: `deletion_threshold=0.05`

**Prevention**: Use `train_fashion_mnist_stable.py` (pre-tuned)

### Issue 3: Too Many Deletions

**Diagnosis**:
```python
# Stats show 100+ deletions by epoch 5
# Accuracy drops significantly
```

**Root Causes**:
1. Threshold too high (too many neurons qualify)
2. Gating rate too high (neurons disappear too fast)
3. Marking cycles too short (deletions happen too quickly)

**Solutions**:
1. Lower threshold: `deletion_threshold=0.02`
2. Lower gating rate: `gating_rate=0.01`
3. Raise marking cycles: `marking_cycles=30`
4. Limit deletions: `max_deletions_per_cycle=1`

### Issue 4: No Utility Differentiation

**Diagnosis**:
```python
utilities = get_all_utilities(model)
print(np.std(utilities))  # = 0.002 (should be > 0.05) ❌
```

**Root Causes**:
1. All neurons activated equally (bad architecture)
2. Utility formula broken
3. Network too small

**Solutions**:
1. Increase layer sizes
2. Debug utility computation (check PHASE 3 validation)
3. Verify error_correlation is computed correctly

---

## Verification & Validation

### Sanity Checks (Per Epoch)

```python
# Check 1: Utilities are bounded and reasonable
utilities = get_all_utilities(model)
assert 0.0 <= np.min(utilities), "Negative utilities!"
assert np.max(utilities) <= 1.0, "Utilities > 1!"
assert np.std(utilities) > 0.02, "No differentiation!"

# Check 2: No pathological activations
activations = get_all_activations(model)
assert not np.any(np.isnan(activations)), "NaN activations!"
assert np.max(np.abs(activations)) < 100, "Exploding activations!"

# Check 3: Loss is reasonable
assert not np.isnan(loss), "NaN loss!"
assert loss < 10.0, "Very high loss!"

# Check 4: Accuracy not crashing
assert accuracy > previous_accuracy * 0.95, "Accuracy dropped >5%!"
```

### Full Validation (End of Training)

```python
# Should satisfy ALL of these:
✓ Final accuracy >= 85% (on Fashion-MNIST)
✓ Total deletions > 0 (some plasticity happened)
✓ Utility std > 0.05 (neurons differentiate)
✓ No NaN in loss curve (training stable)
✓ Created neurons >= 0 (capacity restored if deleted)
✓ Final accuracy >= initial (didn't get worse)
```

---

## Debugging Workflow

### Step 1: Verify Basics (PHASE 1-2)

```bash
# Run baseline (no plasticity)
python train_baseline.py
# Expected: ~97% accuracy, smooth convergence

# Run PHASE 2 (observation only)
python train_ssnc_phase2.py
# Expected: Same accuracy, utilities differentiate
```

### Step 2: Validate Utilities (PHASE 3)

```bash
# Run utility validation
python train_ssnc_phase3.py
# Expected: 4 tests pass, utility metric is valid
```

### Step 3: Enable Plasticity (PHASE 6+)

```bash
# Start with conservative settings
python train_complete_ssnc.py --config conservative
# Expected: Some deletions, stable training

# Try working implementation
python train_simple_working.py
# Expected: Clear "Deleted X neurons" messages
```

### Step 4: Debug Specific Issues

**If plasticity inactive**:
- Check neuron storage: `print(type(model.ssnc_neurons))`
- Should be `list`, not `nn.ModuleList`
- Verify utilities computed: Add logging in utility calculation

**If unstable**:
- Add gradient clipping immediately
- Run with conservative config
- Check Fashion-MNIST normalization

**If too many deletions**:
- Lower threshold
- Raise marking_cycles
- Limit max_deletions_per_cycle

---

## Advanced Tuning

### For Maximum Plasticity Activity

```python
# Want to see lots of deletions/creations?
'deletion_threshold': 0.20,      # High = easy to delete
'marking_cycles': 5,             # Short = fast action
'gating_rate': 0.15,            # Fast = visible change
'max_deletions_per_cycle': 5,   # Many = lots of action
'creation_threshold': 0.60,     # Low = easy to create
'plasticity_cycle_every': 30,   # Frequent = more events
```

### For Maximum Stability

```python
# Want maximum stability and minimum plasticity activity?
'deletion_threshold': 0.02,      # Low = hard to delete
'marking_cycles': 50,            # Long = very careful
'gating_rate': 0.01,            # Slow = gradual change
'max_deletions_per_cycle': 1,   # One = very limited
'creation_threshold': 0.90,     # High = rarely create
'plasticity_cycle_every': 200,  # Infrequent = fewer events
```

### For Fashion-MNIST Specifically

```python
# Fashion-MNIST needs more care than MNIST
'learning_rate': 0.01,           # Lower than MNIST
'activation': 'relu',            # More stable than tanh
'weight_decay': 1e-4,           # Regularization
'normalize': (0.2860, 0.3530),  # Fashion values
'deletion_threshold': 0.08,      # Moderate
'marking_cycles': 15,            # Patient
'gating_rate': 0.05,            # Gradual
```

---

## Next Steps

1. **Understand Current State**: Run `python train_simple_working.py`
2. **Read Details**: Study `COMPLETE_SSNC_DOCUMENTATION.md`
3. **Experiment**: Try different configs (conservative → aggressive)
4. **Verify**: Check PHASE 3 validation on your data
5. **Implement PHASE 9**: Redistribution of capacity across layers
6. **Research**: Explore advanced topics (multi-layer coordination, etc.)

---

## Key Takeaways

- **Simplicity Wins**: Direct Python lists beat complex Module wrapping
- **Gradual Changes**: Soft deletion → hard deletion is safer than instant removal
- **Conservative Default**: Better stable + inactive than unstable + active
- **Validation Matters**: PHASE 3 testing catches issues early
- **Parameters Are Sensitive**: Small changes can disable entire system
- **Logging Is Critical**: Without logs, debugging is impossible
- **Hybrid Learning Works**: Global SGD + local updates = better results
