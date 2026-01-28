# SSNC Implementation - Getting Started

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision numpy matplotlib
```

## PHASE 1: Baseline Model

**Status**: Code complete, ready to train

**Files**:
- `baseline_mlp.py` - Instrumented MLP with statistics tracking
- `train_baseline.py` - Training script for MNIST
- `requirements.txt` - Python dependencies

### What PHASE 1 Does

Creates a **normal MLP with full observability**:
- Standard architecture (no SSNC logic yet)
- Per-neuron activation tracking (mean, variance)
- Per-layer output norm tracking
- Smoothed loss tracking
- Automatic statistics via forward hooks
- Easy weight freezing for later phases

### Running PHASE 1

```bash
# Test the model structure
python baseline_mlp.py

# Train the baseline model
python train_baseline.py
```

This will:
1. Download MNIST (~10MB)
2. Train a 784 → 256 → 128 → 10 MLP
3. Save checkpoints to `checkpoints/phase1/`
4. Generate training curves
5. Save detailed statistics

### Expected Results

- Train accuracy: ~98%
- Val accuracy: ~97%
- Training time: ~5-10 minutes (CPU)

### What to Check

After training, verify:
- ✓ Model converges smoothly
- ✓ No layer has dead neurons (zero mean activation)
- ✓ Activation variances are reasonable (not too small/large)
- ✓ Output norms are stable across epochs

### Files Generated

```
checkpoints/phase1/
├── best_model.pt          # Best validation checkpoint
├── final_model.pt         # Final model + full history
├── training_history.json  # Detailed statistics
└── training_curves.png    # Loss/accuracy plots
```

### Next Steps

Once PHASE 1 is validated:
1. Review neuron statistics for anomalies
2. Confirm no dead neurons or unstable layers
3. **Move to PHASE 2**: Add SSNC-Neuron local state (observation only)

---

## Quick Test (No Training)

```python
from baseline_mlp import InstrumentedMLP, print_model_stats
import torch

# Create model
model = InstrumentedMLP(
    input_dim=784,
    hidden_dims=[128, 64],
    output_dim=10,
    activation='tanh'
)

# Test forward pass
x = torch.randn(32, 784)
y = model(x)
print(f"Input: {x.shape} → Output: {y.shape}")

# Check statistics
print_model_stats(model)
```

---

## Architecture Choices (from PHASE 0)

✅ **Framework**: PyTorch (easy introspection)  
✅ **Model**: Plain MLP (no BatchNorm, no Attention)  
✅ **Activation**: Tanh (smooth, SSNC-friendly)  
✅ **Task**: MNIST (simple supervised learning)  
✅ **Start**: Warm start (pretrained baseline)

These choices minimize complexity and make SSNC behavior clear.
# SSNC
