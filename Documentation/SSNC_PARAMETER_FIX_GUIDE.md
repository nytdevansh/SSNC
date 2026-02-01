# Common SSNC Parameter Issues - Quick Fix Guide

## Issue 1: TypeError with 'dropout' parameter

**Error:**
```
TypeError: CompleteSSNC_MLP.__init__() got an unexpected keyword argument 'dropout'
```

**Cause:** CompleteSSNC_MLP doesn't accept a `dropout` parameter. SSNC has its own internal regularization through pruning.

**Fix:** Remove `dropout` from the initialization:

```python
# WRONG:
self.ssnc_mlp = CompleteSSNC_MLP(
    input_dim=384,
    hidden_dims=[768, 512, 384, 256],
    output_dim=4,
    activation='relu',
    dropout=0.15,  # <-- Remove this line
    ...
)

# CORRECT:
self.ssnc_mlp = CompleteSSNC_MLP(
    input_dim=384,
    hidden_dims=[768, 512, 384, 256],
    output_dim=4,
    activation='relu',
    # No dropout parameter needed
    ...
)
```

## Issue 2: TypeError with 'activation' parameter

**Error:**
```
TypeError: CompleteSSNC_MLP.__init__() got an unexpected keyword argument 'activation'
```

**Cause:** Some versions of CompleteSSNC_MLP may not accept 'activation' as a string.

**Fix:** Check your CompleteSSNC_MLP implementation. If it doesn't accept activation, remove it:

```python
# Try without activation parameter
self.ssnc_mlp = CompleteSSNC_MLP(
    input_dim=384,
    hidden_dims=[768, 512, 384, 256],
    output_dim=4,
    # activation='relu',  # <-- Remove if not supported
    ...
)
```

## Standard CompleteSSNC_MLP Parameters

These parameters are typically supported:

**Required:**
- `input_dim`: Input dimension (e.g., 384 for embeddings)
- `hidden_dims`: List of hidden layer sizes (e.g., [768, 512, 384, 256])
- `output_dim`: Output dimension (e.g., 4 for 4-class classification)

**SSNC Control:**
- `enable_local_learning`: Enable local learning rules (True/False)
- `enable_soft_deletion`: Enable soft deletion (True/False)
- `enable_hard_deletion`: Enable hard deletion (True/False)
- `enable_creation`: Enable neuron creation (True/False)
- `enable_redistribution`: Enable weight redistribution (True/False)
- `enable_monitoring`: Enable monitoring (True/False)

**SSNC Thresholds:**
- `soft_deletion_threshold`: Utility threshold for soft deletion (e.g., 0.05)
- `soft_marking_cycles`: Cycles to observe before soft deletion (e.g., 25)
- `hard_deletion_cycles`: Additional cycles before hard deletion (e.g., 50)
- `creation_threshold`: Utility threshold for creation (e.g., 0.73)
- `creation_method`: Method for creation ('duplicate', 'random', etc.)
- `max_deletions_per_cycle`: Max deletions per cycle (e.g., 2)
- `max_creations_per_cycle`: Max creations per cycle (e.g., 1)

**NOT Typically Supported:**
- ❌ `dropout`: Use SSNC's pruning instead
- ❌ `activation`: May not be parameterizable (check your implementation)
- ❌ `batch_norm`: Usually not included
- ❌ `bias`: Usually default behavior

## How to Find Supported Parameters

If you're unsure what parameters your CompleteSSNC_MLP accepts:

```python
import inspect
from complete_ssnc_system import CompleteSSNC_MLP

# Get the signature
sig = inspect.signature(CompleteSSNC_MLP.__init__)
print("Supported parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  - {param_name}")
```

## Quick Test

Try this minimal initialization to test:

```python
from complete_ssnc_system import CompleteSSNC_MLP

# Minimal test
model = CompleteSSNC_MLP(
    input_dim=10,
    hidden_dims=[20, 15],
    output_dim=4,
    enable_soft_deletion=True,
    enable_hard_deletion=True,
    enable_creation=True,
    soft_deletion_threshold=0.05,
    soft_marking_cycles=25,
    hard_deletion_cycles=50,
    creation_threshold=0.73,
    creation_method='duplicate',
    max_deletions_per_cycle=2,
    max_creations_per_cycle=1
)

print("✓ Model created successfully!")
print(f"  Initial neurons: {model.get_statistics()['total_active']}")
```

If this works, gradually add other parameters one at a time to find which ones are supported.

## Current Fix Applied

The FIXED version of train_real_gpqa_FIXED.py has been updated to:
1. ✅ Remove `dropout` parameter from CompleteSSNC_MLP initialization
2. ✅ Keep `activation='relu'` (usually supported, but remove if you get errors)
3. ✅ Keep all standard SSNC parameters

If you still get errors, try removing the `activation` parameter as well.
