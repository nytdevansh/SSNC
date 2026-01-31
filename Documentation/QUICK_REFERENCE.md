# SSNC Terminology Quick Reference Card

**Frozen Vocabulary**: 31 January 2026

---

## The Framework

**SSNC** = Self-Stabilizing Neuronal Control
- Decentralized learning with bounded local updates
- Slow topology regulation via Observer
- No global gradient control
- Self-stabilizing (prevents Silent Failure)

---

## Three Levels of Adaptation

### Level 1: SSNC-Neuron (Fast)
- Performs **Bounded Local Updates** each batch
- Uses **Neuron Local State**: activation mean/variance, error correlation, age, inactivity counter, gate value
- Computes **Neuron Utility Signal** = smoothed correlation with Downstream Error

### Level 2: Structural Observer (Slow)
- Operates on **Observer Cycles** (every 50-200 batches)
- Aggregates **Neuron Local State** from all neurons
- Computes **Structural Health Metrics** (utility distribution, inactivity, load balance)
- Makes **Structural Actions**: Soft Deletion, Hard Deletion, Neurogenesis, Redistribution

### Level 3: Global Monitoring (Very Slow)
- **Global Progress Monitor** detects **Silent Failure** (learning stalled)
- **Stagnation Stop Rule** prevents runaway growth
- Enforces **Timescale Separation** (fast updates < slow decisions)

---

## Core Actions

| Action | Reversible | Duration | Purpose |
|--------|-----------|----------|---------|
| **Soft Deletion** | ✓ Yes | gate_value *= 0.95 for ~60 cycles | Test impact before permanent removal |
| **Hard Deletion** | ✗ No | After 100+ cycles of Soft Deletion | Permanently remove low-utility neurons |
| **Neurogenesis** | ✓ Yes (re-gate) | Starts gated, ungates if useful | Expand capacity when overloaded |
| **Neuron Duplication** | ✓ Yes | Copy + weak gating | Neurogenesis method: copy high-utility neurons |

---

## Key Metrics

### Neuron Utility Signal
- **Definition**: Smoothed correlation between neuron output and Downstream Error
- **Range**: 0.0 - 0.5 (typical healthy network)
- **Computed over**: Utility Window (50-200 batches)
- **Used for**: Soft Deletion and Neurogenesis decisions

### Structural Health Metrics
- Utility Signal distribution (mean, std)
- Count of Inactive Neurons
- Layer load balance
- Neuron age distribution
- Structural Action frequency

### Non-Explosion Guarantee
- Loss remains finite (no explosion to ±∞)
- Activations remain bounded (no NaN/Inf)
- Training doesn't collapse

---

## Decision Rules

```
Per Observer Cycle:

1. Aggregate Neuron Local State
2. Compute Neuron Utility Signals
3. For each neuron:
   IF utility_signal < soft_deletion_threshold AND marked_for_N_cycles:
      Apply Soft Deletion (gate_value *= 0.95)
   IF soft_deleted_for_M_cycles AND no_performance_impact:
      Apply Hard Deletion (remove permanently)
   IF utility_signal > creation_threshold AND neuron.age > 200:
      Create Duplicate (Neurogenesis)

4. Global Progress Monitor:
   IF learning_stalled AND loss_bounded:
      Enable Exploration Mode
   IF too_many_structural_actions:
      Trigger Stagnation Stop Rule
```

---

## Parameter Meanings

| Parameter | Range | Effect |
|-----------|-------|--------|
| soft_deletion_threshold | 0.02-0.15 | Lower = conservative, fewer deletions |
| soft_marking_cycles | 8-30 | Higher = patient observation before action |
| hard_deletion_cycles | 20-100 | Higher = longer verification before hard delete |
| gating_rate | 0.01-0.20 | Lower = slower gating, safer |
| creation_threshold | 0.60-0.90 | Lower = more neurogenesis, aggressive growth |
| max_deletions_per_cycle | 1-5 | Higher = faster adaptation, higher risk |
| utility_window | 50-200 batches | Longer = smoother decisions, more lag |
| observer_cycle_every | 50-200 batches | Higher = more frequent decisions |

---

## State Lifecycle

```
INITIAL (age < grace_period)
   ↓
ACTIVE (or → Candidate for Deletion if low utility)
   ↓
SOFT_DELETED (gate_value reduced gradually)
   ├→ Can recover if utility improves
   └→ Hard Deletion if impact verified
   ↓
HARD_DELETED (permanent removal, no recovery)

Parallel: CREATED (from Neurogenesis)
   → Starts gated (gate_value = 0.1)
   → Ungates if useful
   → Can become Active or re-gated
```

---

## Timescale Separation (Critical!)

```
Speed hierarchy (slowest to fastest):
1. Hard Deletion execution:       every 100+ Observer Cycles
2. Hard Deletion decision:         every 25 Observer Cycles
3. Soft Deletion application:      every Observer Cycle
4. Neuron Utility Signal updates:  every batch
5. Neuron weight updates:          every batch
6. Activation computation:         every forward pass
```

**Why it matters**: Prevents jittery, reactive decisions that destabilize learning

---

## Safety Mechanisms

### Soft Deletion → Hard Deletion Pipeline
1. Low Utility Signal detected
2. Soft Deletion initiated (gate_value reduced)
3. Monitor impact for 25-100 cycles
4. If no performance loss: Hard Deletion approved
5. Permanent removal executed

### Stagnation Stop Rule
- Halts Structural Actions if learning stalled
- Prevents resource exhaustion from endless growth
- Automatically activates Exploration Mode

### Global Progress Monitor
- Tracks smoothed loss/accuracy
- Detects Silent Failure (loss bounded, progress stopped)
- Prevents network from silently breaking

### Structural Invariance Constraints
- I/O layers immutable
- Connection locality preserved
- Core pathways protected
- Ensures recoverability

---

## Common Configurations

### Conservative (Safe, Slow)
```
deletion_threshold: 0.05
marking_cycles: 20
hard_deletion_cycles: 40
creation_threshold: 0.80
max_deletions_per_cycle: 1
max_creations_per_cycle: 1
```

### Moderate (Balanced, Recommended)
```
deletion_threshold: 0.08
marking_cycles: 15
hard_deletion_cycles: 30
creation_threshold: 0.75
max_deletions_per_cycle: 2
max_creations_per_cycle: 1
```

### Aggressive (Fast, High Risk)
```
deletion_threshold: 0.12
marking_cycles: 8
hard_deletion_cycles: 20
creation_threshold: 0.70
max_deletions_per_cycle: 5
max_creations_per_cycle: 2
```

---

## Verification Checklist

During training, verify:

- ✅ Neuron Utility Signal distribution has variance (std > 0.05)
- ✅ Soft Deletions occur (marked count > 0)
- ✅ Soft Deletions precede Hard Deletions (reversibility)
- ✅ Some Neurogenesis when high-utility neurons exist
- ✅ Loss remains bounded (no explosion)
- ✅ Accuracy maintained or improved
- ✅ Structural Actions limited per Observer Cycle
- ✅ Timescale Separation enforced (no too-frequent changes)
- ✅ No Silent Failure detected

---

## Definitions to Remember

**SSNC-Neuron**: Computational unit with bounded local learning, no structural authority

**Neuron Utility Signal**: Scalar estimating importance via smoothed correlation

**Soft Deletion**: Reversible gating (gates weights toward 0, preserves structure)

**Hard Deletion**: Irreversible removal (permanent, after Soft Deletion verification)

**Neurogenesis**: Creating new neurons (or duplicating existing high-utility neurons)

**Structural Observer**: Supervisory process managing topology on slow cycles

**Observer Cycle**: Single slow iteration (aggregates state, makes decisions)

**Timescale Separation**: Enforcement of ordering: fast updates < slow topology decisions

**Silent Failure**: Learning stalled while loss bounded (detected by Global Progress Monitor)

**Bounded Local Updates**: Weight changes magnitude-clipped per neuron

**Downstream Error**: Indirect task signal (not full gradient) for correlation computation

---

## Reference Documents

- **SSNC_FORMAL_GLOSSARY.md** — Complete definitions (40+ terms)
- **TERMINOLOGY_BRIDGE.md** — Informal ↔ Formal mapping
- **README.md** — Overview and quick start
- **PROJECT_GUIDE.md** — Implementation phases
- **COMPLETE_SSNC_DOCUMENTATION.md** — Full system details

---

**Quick Principle**: SSNC = Decentralized + Bounded + Slow Observation = Self-Stabilizing

**When in doubt**: Check SSNC_FORMAL_GLOSSARY.md for exact definition
