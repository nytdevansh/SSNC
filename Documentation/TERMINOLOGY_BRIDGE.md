# SSNC Terminology Bridge: Informal → Formal

Quick reference for translating between casual and formal SSNC terminology.

---

## Core Concepts

| Informal | Formal | Definition |
|----------|--------|-----------|
| "neuron" | SSNC-Neuron | Adaptive unit with Neuron Local State and Bounded Local Updates |
| "neuron importance" | Neuron Utility Signal | Smoothed correlation between activation and Downstream Error |
| "self-awareness" | Neuron Local State | Fixed-size statistics tracked per neuron |
| "global observer" | Structural Observer | Supervisory process managing topology via slow Observer Cycles |
| "deleting neurons" | Hard Deletion or Soft Deletion | Irreversible removal or reversible gating |
| "gating" | Soft Deletion | Gradual reduction of output via gate_value multiplication |
| "gradually delete" | Soft Deletion | Reversible gating-based removal process |
| "kill" | Hard Deletion | Irreversible permanent removal |
| "create neurons" | Neurogenesis, Neuron Creation | Adding new SSNC-Neurons to network |
| "duplicate neurons" | Neuron Duplication | Copying high-Utility-Signal neuron weights |
| "dead neurons" | Inactive Neurons | Neurons below activation threshold for Utility Window |
| "candidate" | Candidate for Deletion | Low-Utility-Signal neurons being monitored |
| "plasticity" | Structural Actions | All topology-modifying operations (deletion, creation, etc.) |
| "inactive" | Silent Failure | Learning progress halts while loss remains bounded |
| "hysteresis" | Timescale Separation | Enforced ordering of update speeds |
| "marking cycles" | Observation period | Duration before Structural Actions trigger |
| "utility threshold" | Neuron Utility Signal threshold | Boundary determining deletion/creation candidates |

---

## Phases & Components

| Informal | Formal |
|----------|--------|
| PHASE 2 neurons | SSNC-Neurons with Neuron Local State |
| PHASE 3 validation | Neuron Utility Signal validation via Recovery Tests |
| PHASE 4 learning | Bounded Local Updates with Utility Window |
| PHASE 5 monitoring | Structural Observer with Structural Health Metrics |
| PHASE 6 soft deletion | Soft Deletion via gradual gating |
| PHASE 7 hard deletion | Hard Deletion after Soft Deletion verification |
| PHASE 8 creation | Neurogenesis via Neuron Creation and Neuron Duplication |
| PHASE 10 safety | Global Progress Monitor for Silent Failure detection |

---

## Parameter Terminology

| Informal | Formal |
|----------|--------|
| "deletion threshold" | Neuron Utility Signal threshold for Soft Deletion |
| "marking cycles" | Observation cycles before Structural Actions |
| "gating rate" | Rate of gate_value reduction per Observer Cycle |
| "utility window" | Temporal buffer for Neuron Utility Signal averaging |
| "observer cycle" | Slow iteration period for Structural Observer |
| "max deletions per cycle" | Maximum Hard Deletions per Observer Cycle |
| "max creations per cycle" | Maximum Neurogenesis events per Observer Cycle |

---

## Common Phrases

### Informal Statements → Formal Equivalents

**"The network deletes neurons"**
→ "The Structural Observer executes Hard Deletion decisions based on low Neuron Utility Signals"

**"Neurons adapt locally"**
→ "Each SSNC-Neuron performs Bounded Local Updates using its Neuron Local State"

**"The system is plasticity-heavy"**
→ "The Observer Cycle frequency and Structural Action limits allow high topology evolution"

**"Utilities predict importance"**
→ "Neuron Utility Signal shows high correlation with importance (validated in PHASE 3)"

**"Gradual deletion protects performance"**
→ "Soft Deletion allows testing of impact before Hard Deletion, maintaining Non-Explosion Guarantee"

**"The network regulates itself"**
→ "The Structural Observer aggregates Neuron Local State and makes conservative Structural Actions via Timescale Separation"

**"We prevent cascading failures"**
→ "Stagnation Stop Rule and Global Progress Monitor prevent Silent Failure and runaway growth"

**"Neurons are independent agents"**
→ "Each SSNC-Neuron maintains autonomous Bounded Local Updates with no authority over topology"

---

## State Machine Terminology

### Informal State Names → Formal Equivalents

| Informal | Formal | Transition Trigger |
|----------|--------|-------------------|
| "young neuron" | INITIAL state | age < grace_period |
| "useful neuron" | ACTIVE state | Neuron Utility Signal above threshold |
| "low-utility neuron" | Candidate for Deletion | Neuron Utility Signal low & sustained |
| "gating neuron" | SOFT_DELETED state | Marked for soft_marking_cycles |
| "dead neuron" | HARD_DELETED state | Soft deleted for hard_deletion_cycles |
| "new neuron" | CREATED state | Created via Neurogenesis, starts gated |

---

## Decision-Making Terminology

| Informal | Formal |
|----------|--------|
| "Should we delete?" | Is neuron a Candidate for Deletion? |
| "Should we create?" | Is any neuron overloaded (high Utility Signal & mature)? |
| "Should we hard delete?" | Has neuron been Soft Deleted long enough without impact? |
| "Are we stagnant?" | Global Progress Monitor detects Silent Failure? |
| "Should we slow down?" | Is Timescale Separation being violated? |

---

## Attribute Terminology

| Informal | Formal |
|----------|--------|
| "activation strength" | Activation magnitude in Neuron Local State |
| "neuronal variance" | Activation variance in Neuron Local State |
| "inactivity score" | Inactivity counter in Neuron Local State |
| "error sensitivity" | Error correlation in Neuron Local State |
| "age" | Lifetime age in Neuron Local State |
| "gate" | Gate value in Soft Deletion mechanism |
| "plasticity budget" | Maximum Structural Actions per Observer Cycle |

---

## Safety Mechanism Terminology

| Informal | Formal |
|----------|--------|
| "waiting before deletion" | Observation cycles and Utility Window |
| "not deleting immediately" | Soft Deletion before Hard Deletion |
| "preventing cascade" | Stagnation Stop Rule and Global Progress Monitor |
| "preserving weights" | Weights intact during Soft Deletion |
| "allowing recovery" | Utilities can improve during Soft Deletion observation |
| "protecting young neurons" | Grace period before SSNC-Neurons eligible for Soft Deletion |

---

## Measurement Terminology

| Informal | Formal |
|----------|--------|
| "utility distribution" | Neuron Utility Signal variance and statistics |
| "network health" | Structural Health Metrics from Observer |
| "plasticity activity" | Frequency of Structural Actions per epoch |
| "structural changes" | Count of Hard Deletions and Neurogenesis events |
| "stability" | Non-Explosion Guarantee maintained, no Silent Failure |

---

## Philosophy Mapping

**Informal Philosophy**: "Neurons learn locally and the system adapts its structure"

**Formal Philosophy**: "SSNC-Neurons perform Bounded Local Updates via Neuron Local State while the Structural Observer, operating on slow Observer Cycles via Timescale Separation, manages conservative Structural Actions (Soft Deletion → Hard Deletion, Neurogenesis) to maintain self-stabilization and prevent Silent Failure."

---

## Usage Guide

When reading older documentation or discussing SSNC:

1. If you see "neuron utility" → interpret as "Neuron Utility Signal"
2. If you see "observer" → interpret as "Structural Observer" 
3. If you see "delete/deletion" → clarify: is it Soft Deletion or Hard Deletion?
4. If you see "create/creation" → interpret as "Neurogenesis" (Neuron Creation or Duplication)
5. If you see "inactive" → check context: Inactive Neuron or Silent Failure?
6. If you see "local learning" → interpret as "Bounded Local Updates"
7. If you see "self-aware" → interpret as "Neuron Local State"
8. If you see "cycles" → clarify: Observer Cycles or training iterations?

---

**This bridge document helps maintain consistency while onboarding to formal terminology.**
