# Semantic Chaos Experiment Guide

## Overview

This experiment tests whether SSNC can **maintain structural memory** when task semantics change. Unlike the previous stability test, this directly measures SSNC's ability to preserve and reactivate dormant knowledge.

---

## Why This Test Matters

### The Problem with Previous Tests
- Fashion-MNIST was **too stable** (no semantic pressure)
- Base model never needed to forget or remember
- SSNC proved **resilience** but not **necessity**

### What This Test Does
- Forces **catastrophic forgetting** in Phase 2
- Tests **structural memory** in Phase 3
- Measures **recovery speed** (key metric)

---

## Three-Phase Design

### Phase 1: Normal Training (Epochs 1-10)
**Goal:** Both models learn original task

**Labels:**
```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

**Expected:**
- Both models reach ~88-89% accuracy
- SSNC creates initial structure
- Base model optimizes weights

### Phase 2: Rule Switch (Epochs 11-20)
**Goal:** Force both models to relearn

**Label Swaps:**
```
0 ↔ 1  (T-shirt ↔ Trouser)
2 ↔ 6  (Pullover ↔ Shirt)
4 ↔ 8  (Coat ↔ Bag)
```

**Expected:**
- Accuracy drops to ~30% (random guessing on swapped classes)
- Both models relearn to ~87%

**What Happens:**
- **Base model:** Overwrites old weights with new mappings
- **SSNC:** Gates old neurons, creates new ones for new mappings

**Critical SSNC Behavior:**
```
Old neurons:  "T-shirt → 0" (gated, utility drops)
New neurons:  "T-shirt → 1" (created, active)
```

### Phase 3: Restoration (Epochs 21-30)
**Goal:** Test structural memory

**Labels:** Original (same as Phase 1)

**Expected:**

**Base Model:**
```
Epoch 21: 30-35% (catastrophic forgetting)
Epoch 25: 70-75% (slow relearn)
Epoch 30: 85-87% (recovered but slower)
```

**SSNC Model:**
```
Epoch 21: 50-60% (partial memory!)
Epoch 23: 85%+ (fast recovery)
Epoch 30: 88-89% (full recovery)
```

**Why SSNC Should Win:**
- Gated neurons from Phase 1 **still exist**
- When labels restore, their utility **spikes**
- SSNC **ungates** them quickly
- Result: **Structural memory** recovers faster than weight retraining

---

## Success Criteria

### Primary Metric: Phase 3 Recovery

| Metric | Base Model | SSNC | Winner |
|--------|-----------|------|---------|
| **Initial accuracy (epoch 21)** | 30-35% | 50-60% | **SSNC** ✓ |
| **Epochs to 85%** | 8-10 | 2-4 | **SSNC** ✓ |
| **Final accuracy (epoch 30)** | 85-87% | 88-89% | **SSNC** ✓ |
| **Recovery speed** | Slow | **2x faster** | **SSNC** ✓ |

### Win Condition
**SSNC must:**
1. Start Phase 3 with >10% advantage (structural memory proof)
2. Recover to 85% in <5 epochs (vs >7 for base)
3. End with ≥1% higher final accuracy

If all three: **CLEAR WIN** ✓  
If two: **STRONG EVIDENCE** ✓  
If one: **WEAK EVIDENCE** ⚠️  
If none: **FAILED** ❌

---

## What You'll See During Training

### Phase 1 Output (Normal)
```
PHASE1: Epochs 1-10
Original labels
==================================================
Epoch  1: Train Loss: 0.5123 Train Acc: 81.23% Val Acc: 83.45%
Epoch  5: Train Loss: 0.2456 Train Acc: 90.12% Val Acc: 88.67%
Epoch 10: Train Loss: 0.1890 Train Acc: 93.45% Val Acc: 89.23%
```

### Phase 2 Output (Chaos)
```
PHASE2: Epochs 11-20
Label swaps: [(0, 1), (2, 6), (4, 8)]
==================================================
Epoch 11: Train Loss: 1.2345 Train Acc: 35.67% Val Acc: 36.12%  ← DROP!
Epoch 15: Train Loss: 0.4567 Train Acc: 82.34% Val Acc: 81.90%  ← RELEARN
Epoch 20: Train Loss: 0.2234 Train Acc: 91.23% Val Acc: 87.45%  ← RECOVERED
```

### Phase 3 Output (Memory Test)
```
PHASE3: Epochs 21-30
Original labels
==================================================

BASELINE:
Epoch 21: Train Loss: 1.1234 Train Acc: 40.12% Val Acc: 39.45%  ← FORGOT!
Epoch 25: Train Loss: 0.5678 Train Acc: 79.23% Val Acc: 77.12%  ← SLOW
Epoch 30: Train Loss: 0.2890 Train Acc: 89.45% Val Acc: 86.78%  ← FINAL

SSNC:
Epoch 21: Train Loss: 0.7890 Train Acc: 65.34% Val Acc: 62.12%  ← MEMORY!
Epoch 23: Train Loss: 0.3456 Train Acc: 88.67% Val Acc: 85.23%  ← FAST!
Epoch 30: Train Loss: 0.1789 Train Acc: 93.12% Val Acc: 89.01%  ← SUPERIOR
```

**Key difference:** SSNC's epoch 21 accuracy (62%) vs baseline (39%)  
**That 23% gap = structural memory**

---

## Interpreting Results

### Scenario 1: Clear Win ✓
```
SSNC Phase 3 Initial: 60%
Base Phase 3 Initial: 35%
Gap: 25%

Conclusion: SSNC preserved dormant structure from Phase 1
Claim: "SSNC demonstrates structural memory via gated neuron reactivation"
```

### Scenario 2: Partial Win ⚠️
```
SSNC Phase 3 Initial: 45%
Base Phase 3 Initial: 38%
Gap: 7%

Conclusion: Weak evidence of memory, needs tuning
Action: Reduce gating rate, increase hard deletion threshold
```

### Scenario 3: No Difference ❌
```
SSNC Phase 3 Initial: 36%
Base Phase 3 Initial: 35%
Gap: 1%

Conclusion: SSNC gates too aggressively, neurons hard-deleted
Action: Check if Phase 1 neurons survived to Phase 3
```

---

## Debugging Tips

### Check SSNC Neuron Counts
After Phase 1 (epoch 10):
```python
# Should see ~580-600 neurons total
# Soft-deleted: 20-40
# Hard-deleted: 0-5
```

After Phase 2 (epoch 20):
```python
# Should see ~600-650 neurons total
# Soft-deleted: 100-150 (old Phase 1 neurons gated!)
# Hard-deleted: 30-50
# Created: 50-100 (new Phase 2 neurons)
```

After Phase 3 (epoch 30):
```python
# Should see ~580-620 neurons total
# Soft-deleted: 50-80 (Phase 2 neurons now gated)
# Hard-deleted: 50-80
# Active: Phase 1 neurons reactivated!
```

### If SSNC Doesn't Win

**Problem:** Phase 3 initial accuracy same as baseline

**Possible Causes:**
1. **Gating too fast** → Phase 1 neurons hard-deleted before Phase 3
   - Solution: Increase `hard_deletion_cycles` to 100
2. **Creation too aggressive** → Too much churn, no stable memory
   - Solution: Already fixed with sustained utility requirement
3. **Grace period too short** → Young neurons deleted before stabilizing
   - Solution: Increase `grace_period` to 100

---

## Expected Terminal Output

```
======================================================================
ANALYSIS: Semantic Memory Test
======================================================================

Phase 3 Recovery Analysis:
----------------------------------------------------------------------

BASELINE:
  Initial accuracy (epoch 21): 35.23%
  Final accuracy (epoch 30):   86.45%
  Recovery gain:                51.22%
  Epochs to 85%:                8

SSNC:
  Initial accuracy (epoch 21): 58.67%
  Final accuracy (epoch 30):   88.92%
  Recovery gain:                30.25%
  Epochs to 85%:                3

======================================================================
✓ WIN: SSNC shows structural memory (>10% initial advantage)
======================================================================
```

---

## Running the Experiment

```bash
# Make sure you have the fixed SSNC file
python semantic_chaos_experiment.py
```

**Time estimate:** 45-60 minutes on CPU

**Output:**
- `semantic_chaos_results.json` - Detailed results
- Terminal analysis with winner determination

---

## What This Proves (If Successful)

### Scientific Claims You Can Make:

1. **SSNC maintains structural memory under semantic drift**
   - Evidence: >10% accuracy advantage when rules restore

2. **Gated neurons preserve learned representations**
   - Evidence: Faster recovery than weight retraining

3. **SSNC handles non-stationary tasks better than static networks**
   - Evidence: 2x faster recovery in Phase 3

4. **Structural plasticity enables knowledge reuse**
   - Evidence: Same neurons active in Phase 1 and Phase 3

### What You Can't Claim (Yet):

- Superior accuracy on stable tasks (already tested, neutral)
- Computational efficiency (not tested)
- Scalability to larger networks (not tested)

---

## Next Steps After This Experiment

### If SSNC Wins:
1. Try capacity-ceiling test (Option B from earlier)
2. Test on CIFAR-10 with domain shift
3. Write paper on structural memory

### If SSNC Loses:
1. Debug neuron lifecycle (check counts)
2. Tune gating/deletion parameters
3. Try Option C (capacity ceiling + drift)

---

## Key Insight

**This experiment doesn't test "is SSNC better at learning"**  
**It tests "can SSNC remember while forgetting"**

That's a fundamentally different question, and one where SSNC should have a clear advantage.

---

Good luck! 🚀
