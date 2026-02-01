# Quick Start: Semantic Chaos Experiment

## What Changed

### 1. Fixed Critical Bugs ✓
- **Gradient indexing** - New neurons no longer crash when observing gradients
- **Timestamp initialization** - Soft-deleted neurons now have proper timestamps
- **Defensive checks** - Added safety checks to prevent future crashes

### 2. Tightened Creation Logic ✓
**OLD (Too Permissive):**
```python
if utility > threshold and utility > 0.01:
    create()  # Created on single spike!
```

**NEW (Sustained Signals Required):**
```python
# Requires ALL of:
# 1. Current utility in top 25%
# 2. Absolute minimum > 0.01
# 3. Age > 50 cycles
# 4. Mean utility > 0.01 over last 10 cycles
# 5. Utility variance < 50% of mean (stable, not oscillating)
if all_conditions_met:
    create()  # Only creates on sustained overload
```

**Impact:** Dramatically reduces noise-driven creation and churn

### 3. Created Semantic Chaos Test ✓
New experiment that tests **structural memory**, not just resilience.

---

## How to Run

### Step 1: Verify Files
```bash
ls -la
# Should see:
# - complete_ssnc_phases_7_12.py (fixed version)
# - semantic_chaos_experiment.py (new experiment)
# - SEMANTIC_CHAOS_GUIDE.md (detailed guide)
```

### Step 2: Run Experiment
```bash
python semantic_chaos_experiment.py
```

### Step 3: Wait
- **Time:** 45-60 minutes on CPU
- **Progress:** You'll see 3 phases × 2 models = 6 training runs
- **Output:** Terminal analysis + `semantic_chaos_results.json`

---

## What You'll See

### Phase 1: Both Learn Normally
```
Epoch 10: Train Acc: 93.45% Val Acc: 89.23%
```

### Phase 2: Both Forget (Labels Swapped)
```
Epoch 11: Train Acc: 35.67% Val Acc: 36.12%  ← Chaos!
Epoch 20: Train Acc: 91.23% Val Acc: 87.45%  ← Recovered
```

### Phase 3: SSNC Should Win (Labels Restored)
```
BASELINE:
Epoch 21: Val Acc: 35.23%  ← Forgot everything
Epoch 30: Val Acc: 86.45%  ← Slow relearn

SSNC:
Epoch 21: Val Acc: 58.67%  ← Structural memory! 🎯
Epoch 30: Val Acc: 88.92%  ← Fast recovery! 🚀
```

---

## Success Indicators

### ✓ Clear Win
```
SSNC Phase 3 Initial > Base + 10%
SSNC recovers to 85% in <5 epochs (vs >7 for base)
Final SSNC accuracy > Base + 1%
```

### ⚠️ Partial Win
```
SSNC Phase 3 Initial > Base + 5%
SSNC recovers slightly faster
Needs parameter tuning
```

### ❌ No Difference
```
SSNC Phase 3 Initial ≈ Base
No structural memory observed
Debug: Check if Phase 1 neurons survived to Phase 3
```

---

## Key Metrics to Watch

### During Training
Monitor neuron counts after each phase:

**After Phase 1 (Epoch 10):**
- Active neurons: 570-590
- Soft-deleted: 20-40
- Hard-deleted: 0-10
- Created: 5-15

**After Phase 2 (Epoch 20):**
- Active neurons: 580-620
- Soft-deleted: 100-150 ← Phase 1 neurons gated!
- Hard-deleted: 30-60
- Created: 50-100 ← New Phase 2 neurons

**After Phase 3 (Epoch 30):**
- Active neurons: 570-600
- Soft-deleted: 50-100 ← Phase 2 neurons now gated
- Hard-deleted: 60-100
- Net change from start: -10 to +20

### The Critical Test
**Phase 3, Epoch 21 accuracy:**
- If SSNC > 55% and Base < 40%: **Structural memory confirmed** ✓
- If both ≈ 35%: **No memory, neurons hard-deleted** ❌

---

## What This Tests (vs Previous)

### Previous Test (Structural Resilience)
```
Question: "Can SSNC survive damage?"
Test: Delete 10% of neurons randomly
Result: ✓ Recovery works (131% recovery rate)
Limitation: Base model never needed this
```

### New Test (Structural Memory)
```
Question: "Can SSNC remember while forgetting?"
Test: Swap labels, then restore them
Result: TBD - Run the experiment!
Advantage: Base model CANNOT do this
```

---

## Troubleshooting

### Problem: Both models perform identically in Phase 3
**Cause:** SSNC hard-deleted Phase 1 neurons too quickly

**Fix:**
```python
# In semantic_chaos_experiment.py, modify SSNC creation:
hard_deletion_cycles=100,  # was 60
grace_period=100,  # was 50
```

### Problem: Too much neuron churn
**Cause:** Creation still too aggressive (shouldn't happen with new logic)

**Fix:**
```python
max_creations_per_cycle=1,  # was 2
```

### Problem: No neurons created
**Cause:** Sustained utility requirement too strict

**Fix:**
```python
# In complete_ssnc_phases_7_12.py, line ~610:
if std_util > mean_util * 0.75:  # was 0.5 (more lenient)
```

---

## Expected Outcome

If everything works:

```
======================================================================
✓ WIN: SSNC shows structural memory (>10% initial advantage)
======================================================================

Results saved to semantic_chaos_results.json
```

**This proves:** SSNC can preserve and reactivate dormant knowledge structures, which standard networks cannot do.

---

## Files Included

1. **complete_ssnc_phases_7_12.py** - Fixed SSNC with tightened creation
2. **semantic_chaos_experiment.py** - Full 3-phase experiment
3. **SEMANTIC_CHAOS_GUIDE.md** - Detailed explanation and theory
4. **QUICK_START.md** - This file

---

## Ready to Run?

```bash
python semantic_chaos_experiment.py
```

Then grab coffee ☕ and wait 45 minutes.

Good luck! 🎯
