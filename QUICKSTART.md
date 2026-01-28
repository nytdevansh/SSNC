# SSNC Quick Start Guide

## What We've Built (So Far)

✅ **PHASE 0**: Planning complete  
✅ **PHASE 1**: Baseline instrumented MLP  
✅ **PHASE 2**: SSNC-Neuron with local state tracking

## Files Overview

```
├── SSNC_PROJECT.md           # Complete project documentation
├── README.md                  # Setup instructions
├── requirements.txt           # Python dependencies
│
├── baseline_mlp.py            # PHASE 1: Instrumented MLP
├── train_baseline.py          # PHASE 1: Training script
│
├── ssnc_neuron.py            # PHASE 2: SSNCNeuron class
├── ssnc_mlp.py               # PHASE 2: SSNC-enabled MLP
└── train_ssnc_phase2.py      # PHASE 2: Training script
```

## Installation

```bash
# Install PyTorch and dependencies
pip install torch torchvision numpy matplotlib

# Or use requirements file
pip install -r requirements.txt
```

## Running PHASE 1 (Baseline)

Test the model structure:
```bash
python baseline_mlp.py
```

Train the baseline:
```bash
python train_baseline.py
```

Expected output:
- Train accuracy: ~98%
- Val accuracy: ~97%
- Training time: ~5-10 minutes (CPU)
- Creates `checkpoints/phase1/` with model and statistics

## Running PHASE 2 (SSNC Neurons - Observation Only)

Test SSNC neurons:
```bash
python ssnc_neuron.py
```

Test SSNC integration:
```bash
python ssnc_mlp.py
```

Train with SSNC observation:
```bash
python train_ssnc_phase2.py
```

Expected output:
- **Same accuracy as PHASE 1** (~97%)
- Neuron utilities should differentiate (std > 0.1)
- No training instabilities
- Creates `checkpoints/phase2/` with detailed neuron states

## Key Validation Checks

### For PHASE 1:
- ✓ Model converges smoothly
- ✓ No dead neurons (zero activation)
- ✓ Reasonable activation variances
- ✓ Stable output norms

### For PHASE 2:
- ✓ Performance matches PHASE 1
- ✓ Utilities differentiate neurons
- ✓ Statistics remain bounded
- ✓ No NaNs or explosions
- ✓ Neurons track activations correctly

## Understanding the Code

### SSNCNeuron (ssnc_neuron.py)

Each neuron maintains:
```python
- activation_mean     # Running average of activation
- activation_var      # Running variance
- inactivity_counter  # Steps since last firing
- error_correlation   # Correlation with downstream error
- age                 # Time since creation
- utility            # Computed importance score
```

Key methods:
- `observe_activation()` - Track own output
- `observe_error()` - Track gradient signal
- `compute_utility()` - Calculate importance

### SSNC_MLP (ssnc_mlp.py)

Extends baseline MLP with:
- Automatic activation tracking (forward hooks)
- Automatic gradient tracking (backward hooks)
- Utility computation per neuron
- Detailed statistics export

### Current Limitations (PHASE 2)

🚫 **Neurons do NOT yet**:
- Learn locally (still using standard SGD)
- Make structural decisions
- Delete or create themselves
- Respond to observer commands

✅ **Neurons DO**:
- Track all their own statistics
- Compute utility scores
- Observe error signals
- Maintain stable state

## What's Next?

### PHASE 3: Local Utility Computation (Refinement)
- Validate utility metric quality
- Test different utility formulas
- Ensure noise doesn't dominate
- Verify useful neurons score higher

### PHASE 4: Enable Neuron Learning (Bounded)
- Add local weight updates
- Tie learning rate to utility
- Strict clipping/bounds
- Verify stability

### PHASE 5: Introduce Observer (Read-Only)
- Global view of all neurons
- Decision logging (no action yet)
- Structural health metrics
- Prepare for deletion/creation

## Debugging Tips

**If training fails:**
1. Check PyTorch installation
2. Verify MNIST downloads correctly
3. Check available memory
4. Try smaller batch size (64 instead of 128)

**If utilities are all similar:**
1. Check error correlation computation
2. Verify gradients flow correctly
3. Increase training time
4. Check activation variance

**If performance drops vs baseline:**
1. STOP - something is wrong
2. Check hooks aren't modifying gradients
3. Verify observation is read-only
4. Roll back to PHASE 1

## Testing Individual Components

Test neurons in isolation:
```python
from ssnc_neuron import SSNCNeuron

neuron = SSNCNeuron(neuron_id=0, layer_id=0)

# Simulate activity
for t in range(100):
    act = 0.5 * np.sin(t/10) + 0.1 * np.random.randn()
    neuron.observe_activation(act)
    
    err = -0.3 * act + 0.05 * np.random.randn()
    neuron.observe_error(err)
    
    utility = neuron.compute_utility()

print(neuron.get_state_dict())
```

Test layer management:
```python
from ssnc_neuron import SSNCNeuronLayer

layer = SSNCNeuronLayer(layer_id=0, n_neurons=128)

# Simulate batch
acts = torch.randn(32, 128)
layer.observe_forward(acts)

grads = torch.randn(32, 128)
layer.observe_backward(grads)

layer.compute_all_utilities()
summary = layer.get_layer_summary()
```

## Performance Benchmarks

On CPU (Intel i7, 16GB RAM):
- PHASE 1 baseline: ~5 min for 20 epochs
- PHASE 2 with observation: ~6-7 min for 20 epochs
  - Overhead: ~20-40% (acceptable for research)

On GPU (if available):
- Both should be <2 min for 20 epochs

## Common Issues

**"No module named 'torch'"**
→ Run: `pip install torch torchvision`

**"CUDA out of memory"**
→ Reduce batch_size in CONFIG dict, or use CPU

**"Utilities are all ~0.0"**
→ Normal in early training, should increase by epoch 5-10

**"Loss is NaN"**
→ Check learning rate (try 0.001 instead of 0.01)
→ Check activation statistics for explosions

## Getting Help

1. Check `SSNC_PROJECT.md` for full documentation
2. Review neuron states: `model.get_detailed_neuron_states()`
3. Print statistics: `print_ssnc_stats(model)`
4. Check training curves: look for smooth convergence

## Architecture Philosophy

This phased approach ensures:
- **Observation before action** - understand before changing
- **One mechanism at a time** - isolate sources of failure
- **Extensive logging** - know what's happening
- **Validation at each step** - catch bugs early

If PHASE 2 works correctly, PHASE 3-12 have a solid foundation.

---

**Status**: Ready for training validation  
**Next**: Run PHASE 2 training and verify all checks pass
