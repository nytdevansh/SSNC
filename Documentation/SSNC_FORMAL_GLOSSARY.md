# SSNC: Formal Terminology & Definitions

**This glossary freezes vocabulary for precision. No motivation, no hype — just exact meanings.**

---

## Core Framework

### Self-Stabilizing Neuronal Control (SSNC)

A decentralized learning framework in which SSNC-Neurons adapt locally through bounded updates, while a slow Structural Observer regulates Structural Actions (creation, duplication, deletion) to maintain long-term stability without global gradient control.

---

## Neuron-Level Concepts

### SSNC-Neuron

An adaptive computational unit with:
- Local Neuron Local State (fixed-size statistics)
- Bounded learning capability (magnitude-clipped updates)
- No direct access to global objectives or loss function
- No authority over network topology or Structural Actions

An SSNC-Neuron cannot:
- Add, delete, or duplicate neurons
- Rewire distant layers
- Directly optimize global loss
- Make Structural Actions

### Neuron Local State

A fixed-size set of statistics maintained per SSNC-Neuron, typically including:
- Activation moving average
- Activation variance
- Correlation estimate with Downstream Error
- Inactivity counter (batches without significant firing)
- Age (lifetime in Observer Cycles)
- Gate value (for Soft Deletion)

This state is local, persistent, updated continuously, and accessible only by the SSNC-Neuron and the Structural Observer.

### Bounded Local Updates

Small, magnitude-clipped weight modifications performed by an SSNC-Neuron.

Properties:
- Magnitude clipped (e.g., max_update_norm = 0.01)
- Utility-modulated (high Neuron Utility Signal → larger updates)
- Uses only local Neuron Local State and Downstream Error signals
- No global gradient information
- Cumulative effect = fine-tuning via local adaptation

### Downstream Error

A task-dependent signal representing error or deviation observed in layers after an SSNC-Neuron.

It is:
- Indirect (not a full backpropagation gradient)
- Noisy (signal-to-noise ratio varies by layer)
- Delayed (not immediate feedback)
- Used only for correlation computation with activation

Never used as a direct learning signal (only for Neuron Utility Signal).

### Neuron Utility Signal

A scalar value computed by an SSNC-Neuron that estimates its usefulness.

Defined as:
A smoothed correlation between the neuron's activation and Downstream Error, computed over a Utility Window.

Properties:
- Local computation (computed by neuron itself)
- Bounded (typically 0.0 to 0.5 in healthy networks)
- Slow-changing (averaged over observation window)
- Sign-sensitive (positive = helpful, negative = harmful)
- Used by Structural Observer for Structural Actions

### Utility Window

A fixed-length temporal buffer over which Neuron Utility Signal is averaged.

Purpose:
- Reduce noise from single-batch observations
- Prevent reactive (jittery) behavior
- Stabilize Observer decisions
- Typical range: 50-200 batches

---

## Activity & State Detection

### Inactive Neuron

A neuron whose activation magnitude remains below a predefined threshold for a full Utility Window.

Inactivity is behavioral (observable), not structural (a decision).

Detection: `inactivity_counter >= utility_window_length`

### Candidate for Deletion

A neuron that has:
- Low Neuron Utility Signal (below deletion threshold)
- Sustained inactivity over multiple Observer Cycles
- No recovery of Neuron Utility Signal despite multiple measurement cycles

Being a candidate does not trigger automatic deletion—only marks the neuron for Observer consideration.

---

## Structural Actions & Mechanisms

### Structural Action

An Observer-authorized change to network topology, such as:
- Neuron Creation (Neurogenesis)
- Neuron Duplication
- Soft Deletion initiation
- Hard Deletion execution
- Structural Redistribution of capacity

Properties:
- Conservative (small number per Observer Cycle)
- Sparse (infrequent, not every batch)
- Reversible when possible (Soft Deletion before Hard Deletion)
- Logged for transparency and debugging

### Soft Deletion

A reversible state in which:
- Neuron output is gradually gated toward zero
- Weights remain intact
- Neuron continues monitoring Neuron Utility Signal and updating Neuron Local State
- Recovery is possible if Neuron Utility Signal improves

Implementation:
```
gate_value = gate_value * (1.0 - gating_rate)  # e.g., 0.95 per cycle
output_actual = output_computed * gate_value
```

After ~60 cycles: neuron output reduced to ~5% of original.

Duration: Continues until Hard Deletion decision or recovery.

### Hard Deletion

The irreversible removal of an SSNC-Neuron and its connections.

Only executed after:
- Prolonged Soft Deletion (confirmed low impact)
- No measurable negative effect on performance
- Structural Observer explicit approval
- Occurs at very slow timescales (100+ Observer Cycles after Soft Deletion)

Once Hard Deleted, the neuron cannot recover.

### Neuron Creation (Neurogenesis)

The addition of a new SSNC-Neuron in a weakly activated state.

Properties:
- Local connections only (no long-range rewiring)
- Minimal initial influence (gate_value = 0.1)
- No immediate task impact
- Weights initialized from overloaded neuron (Neuron Duplication) or randomly
- Activation and Neuron Utility Signal develop gradually

Triggers:
- High Neuron Utility Signal in existing neurons (indicating overload)
- Layer capacity constraints

### Neuron Duplication

A Structural Action where:
- An overloaded high-Utility-Signal neuron is copied
- Weights and connections are redistributed
- Duplicates are weakly gated initially (gate_value = 0.1)
- Each copy learns independently

Used instead of trying to "overload" a single neuron beyond capacity.

### Structural Redistribution

The process by which the Structural Observer:
- Shifts representational load across adjacent layers
- Rebalances neuron allocation
- Maintains locality of connections

Triggered only after sustained overload detection.

### Structural Invariance Constraint

A hard rule defining topology elements that cannot be violated.

Examples:
- Input/output layers are immutable
- Locality of connections is preserved
- Core pathway structure is protected

Ensures recoverability and prevents destructive self-modification.

### Structural Health Metrics

Observer-level measurements used for diagnosis and logging:
- Count of active vs. total SSNC-Neurons
- Distribution of Neuron Utility Signal (mean, std, min, max)
- Count of Inactive Neurons per layer
- Neuron age distribution
- Add/delete frequency per cycle
- Network depth and width drift
- Capacity utilization

Used only for diagnosis and reporting, never directly for learning.

---

## Observer & Timescale Concepts

### Structural Observer (SSNC-Observer)

A supervisory process that:
- Does not learn weights
- Does not directly optimize performance
- Operates on slow Observer Cycles
- Manages topology only via Structural Actions
- Aggregates Neuron Local State signals (does not generate them)
- Makes conservative binary decisions (delete? create? redistribute?)

Key property: Observer never touches weight learning—only topology.

### Observer Cycle

A discrete, slow iteration during which the Structural Observer:
1. Collects Neuron Local State from all SSNC-Neurons
2. Evaluates Structural Health Metrics
3. Optionally performs limited Structural Actions
4. Logs all changes

Observer cycles are orders of magnitude slower than neuron updates:
- Neuron weight updates: every batch (fast)
- Observer cycles: every 50-200 batches (slow)
- Hard Deletion: every 100+ Observer Cycles (very slow)

### Timescale Separation

The enforced ordering of update speeds:
1. SSNC-Neuron weight updates (fast, per-batch, small magnitude)
2. Neuron Utility Signal updates (medium, per-batch, smoothed over window)
3. Structural Observer evaluation (slow, per 50-200 batches)
4. Hard Deletion execution (very slow, per 100+ Observer Cycles)

This separation is mandatory for stability and prevents:
- Reactive oscillations
- Cascading structural failures
- Loss of learned representations

---

## Failure Detection & Prevention

### Silent Failure

A state where:
- Loss is bounded (no explosion)
- No numeric instability occurs
- But learning progress has stalled or reversed

Indicators:
- Accuracy unchanged for 10+ epochs
- Validation loss increasing slowly but steadily
- Neuron Utility Signal distribution becoming uniform (no differentiation)

Detected only via long-term monitoring (Global Progress Monitor).

### Global Progress Monitor

A read-only diagnostic mechanism that:
- Tracks smoothed global performance (loss, accuracy)
- Detects Silent Failure and stagnation
- Emits a stagnation level signal (0.0 = learning, 1.0 = stagnated)

Never directly changes weights or topology.

### Stagnation Stop Rule

A safety mechanism that halts Structural Actions when:
- Learning progress < ε for T consecutive Observer Cycles
- Structural size continues to grow without performance gain

Prevents infinite slow drift and resource exhaustion.

### Non-Explosion Guarantee (Empirical)

An observed property that:
- Loss remains bounded (does not diverge to ±∞)
- Activations remain finite (no NaN or Inf)
- Training does not collapse mid-epoch

Not a formal proof, but a measurable practical condition.

---

## Recovery & Validation

### Recovery Test

An evaluation protocol where:
- A fraction of SSNC-Neurons is removed or damaged
- SSNC is allowed to adapt and recover
- Recovery speed and stability are measured
- Final performance is compared to undamaged baseline

Validates self-stabilization claims.

### Exploration Mode (Observer)

A temporary Observer state activated during sustained stagnation.

Allows:
- Slightly relaxed Structural Action thresholds
- Cautious structural experimentation
- Increased Neuron Creation rate

Automatically exits after:
- Improvement in Global Progress Monitor
- Timeout threshold reached

---

## Architecture & Implementation

### Warm Start Assumption

An explicit assumption that SSNC begins from:
- A pre-trained model (initialized with task-specific knowledge)
- Or a minimally viable structured network

Cold start (training from random initialization) is out of scope.

### Bounded Learning Regime

The operational mode where:
- All SSNC-Neurons perform Bounded Local Updates
- All weights have gradients clipped below threshold
- No weight can change by more than max_update_norm per update
- Cumulative learning is smooth and stable

---

## Formal Constraints & Guarantees

### Specification Properties

SSNC must satisfy:

1. **Decentralization**: No SSNC-Neuron has authority over topology
2. **Bounded Updates**: All weight changes are clipped by magnitude
3. **Timescale Separation**: Structural Actions are order(s) of magnitude slower than learning
4. **Reversibility-First**: Soft Deletion precedes Hard Deletion
5. **Self-Stabilization**: Creation balances Deletion to prevent Silent Failure
6. **Locality**: Structural Actions preserve connection locality
7. **Logging**: All Structural Actions are logged for auditing
8. **Conservative Defaults**: Parameters favor stability over plasticity activity

---

## Domain-Specific Context

### Utility Window (Fashion-MNIST)

Typical setting: 50-100 batches

- Too short (< 20): Noisy, reactive deletion
- Too long (> 200): Sluggish response to changes
- Optimal: 50-100 (balance of stability and responsiveness)

### Gating Rate (Soft Deletion)

Typical setting: 0.05 per Observer Cycle

- gate_value *= (1.0 - gating_rate)
- After 20 cycles: 0.95^20 ≈ 0.36 (36% of original)
- After 60 cycles: 0.95^60 ≈ 0.05 (5% of original)

### Deletion Threshold (Neuron Utility Signal)

Typical range: 0.05-0.15

- Below 0.05: Very conservative (few deletions)
- 0.08-0.10: Moderate (balanced activity)
- Above 0.15: Aggressive (high deletion rate)

---

## Usage Example

Here's how terminology aligns in execution:

```python
# Initialize SSNC with bounded learning
model = SSNCNetwork(hidden_dims=[384, 192])

# Each SSNC-Neuron maintains Neuron Local State
for neuron in model.ssnc_neurons:
    neuron.activation_mean = ...
    neuron.inactivity_counter = ...
    neuron.compute_utility_signal()  # Neuron Utility Signal

# Every batch: Bounded Local Updates
for neuron in model.ssnc_neurons:
    update = neuron.compute_bounded_update(downstream_error)
    neuron.weights += update  # clipped magnitude

# Every N batches: Observer Cycle
observer.evaluate()
candidates = observer.find_candidates_for_deletion()  # Candidates for Deletion
observer.execute_structural_actions()  # Soft Deletion, Neurogenesis

# Verify self-stabilization
global_monitor.check_for_silent_failure()
if stagnated:
    enable_exploration_mode()
```

---

## Cross-References

All SSNC documentation must use these terms consistently:
- Not "neuron activation" but "neuron output" or "activation magnitude"
- Not "importance" but "Neuron Utility Signal"
- Not "neuroplasticity" but "Structural Actions"
- Not "global observer" but "Structural Observer"
- Not "pruning" but "Hard Deletion" (after Soft Deletion)
- Not "adding neurons" but "Neuron Creation" or "Neurogenesis"
- Not "fast learning" but "weight updates"; "fast updates" but "Bounded Local Updates"
- Not "death" but "Hard Deletion" (irreversible) or "Soft Deletion" (reversible)

---

**This glossary is the source of truth for SSNC terminology. All code, docs, and discussions must reference these exact definitions.**
