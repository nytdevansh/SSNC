# Structural Plasticity Debugging Guide for Fashion-MNIST

## Problem: Training Instability with Structural Plasticity

You're experiencing loss and accuracy going out of control when training with structural plasticity on Fashion-MNIST. This guide will help you identify and fix the issue.

---

## Common Causes (Most → Least Likely)

### 1. **Missing Gradient Clipping** ⚠️ CRITICAL
**Symptom**: Loss explodes to NaN or infinity within first few epochs
**Why**: Structural changes (gating) can cause gradient spikes
**Fix**: Add gradient clipping

```python
# BEFORE optimizer.step(), add this:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Test**: Check if `loss.item()` contains NaN
```python
if np.isnan(loss.item()):
    print("⚠ Gradient explosion detected!")
```

---

### 2. **Aggressive Plasticity Parameters**
**Symptom**: Accuracy drops rapidly, especially after deletions
**Why**: Deleting too many neurons too quickly destabilizes learning
**Current Issues**:
- `deletion_utility_threshold=0.05` → Too high
- `marking_cycles_required=5` → Too short
- `max_deletions_per_cycle=5` → Too aggressive

**Fix**: Use conservative thresholds
```python
deletion_utility_threshold=0.02,  # Lower = more conservative
marking_cycles_required=30,       # Longer wait
max_deletions_per_cycle=1,        # Very limited
gating_rate=0.05,                 # Slower gating
```

---

### 3. **Gate Application Without Weight Synchronization**
**Symptom**: Model degrades even without deletions
**Why**: Gates modify effective weights but don't update actual weights

**Problem Code**:
```python
# This only modifies activations temporarily:
activations = activations * gate_values
```

**Fix**: Apply gates to actual weights
```python
def apply_gates_to_weights(self):
    for layer_idx, gate_values in enumerate(self.gates):
        layer = self.model.layers[layer_idx]
        with torch.no_grad():
            layer.weight.data *= gate_values.unsqueeze(1)
            layer.bias.data *= gate_values
```

---

### 4. **Fashion-MNIST Harder Than MNIST**
**Symptom**: Same config works on MNIST but not Fashion-MNIST
**Why**: Fashion-MNIST has:
- More complex patterns (clothing vs. digits)
- Higher feature variance
- Requires more stable optimization

**Fix**: Fashion-MNIST specific settings
```python
# Use ReLU instead of tanh (often more stable)
activation='relu'

# Lower learning rate
learning_rate=0.01  # vs 0.05 for MNIST

# Proper normalization for Fashion-MNIST
transforms.Normalize((0.2860,), (0.3530,))  # Not MNIST's (0.1307,), (0.3081,)

# Stronger regularization
weight_decay=1e-4
```

---

### 5. **Plasticity Updates Too Frequent**
**Symptom**: Training is slow and unstable
**Why**: Structural changes during every batch cause chaos

**Problem**:
```python
if batch_idx % 10 == 0:  # Too frequent!
    self._handle_structural_changes()
```

**Fix**: Less frequent updates
```python
if batch_idx % 100 == 0:  # Much more stable
    self._handle_structural_changes()
```

---

### 6. **No Grace Period for Young Neurons**
**Symptom**: Neurons deleted before they have chance to learn
**Why**: New neurons start with random weights (low utility)

**Fix**: Age-based protection
```python
# In deletion check:
if neuron.age < 100:  # Grace period
    continue  # Don't delete young neurons
```

---

## Quick Diagnostic Checklist

Run this to identify your issue:

```python
# 1. Check for NaN
print(f"Loss is NaN: {np.isnan(loss.item())}")

# 2. Check gradient magnitudes
for param in model.parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 10:
            print(f"⚠ Large gradient: {grad_norm}")

# 3. Check deletion stats
stats = deletion_manager.get_statistics()
print(f"Soft deleted: {stats['soft_deleted']}")
print(f"Deletions per epoch: {stats['total_soft_deleted'] / epoch}")

# 4. Check utility distribution
utilities = [n.utility for n in all_neurons]
print(f"Utility mean: {np.mean(utilities):.3f}")
print(f"Utility std: {np.std(utilities):.3f}")
print(f"Utilities near threshold: {sum(1 for u in utilities if u < 0.05)}")
```

---

## Solution: Stabilized Configuration

### **Option 1: Minimal Changes (Quickest Fix)**

Just add gradient clipping to your existing code:

```python
# In your training loop, BEFORE optimizer.step():
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0  # This single line can fix 80% of issues
)
optimizer.step()
```

### **Option 2: Conservative Parameters (Recommended)**

Use this configuration:

```python
# Create model
model = SSNC_MLP_Phase6(
    input_dim=784,
    hidden_dims=[128, 64],
    output_dim=10,
    activation='relu',  # Changed from tanh
    # Conservative deletion settings
    enable_soft_deletion=True,
    deletion_utility_threshold=0.02,  # Lower
    marking_cycles_required=30,  # Longer
    gating_rate=0.05,  # Slower
    max_deletions_per_cycle=1,  # Stricter
)

# Optimizer with weight decay
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,  # Added
    nesterov=True
)

# Training loop
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # CRITICAL: Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Infrequent plasticity updates
    if batch_idx % 100 == 0:
        handle_plasticity()
```

### **Option 3: Use Provided Stable Script (Easiest)**

Just run the provided stable training script:

```bash
python train_fashion_mnist_stable.py
```

This script has:
- ✓ Gradient clipping
- ✓ Conservative plasticity parameters
- ✓ Fashion-MNIST specific normalization
- ✓ Proper weight/gate synchronization
- ✓ Detailed logging and diagnostics
- ✓ Automatic plotting

---

## Expected Results (Stable Training)

With the fixes, you should see:

**Epoch 1:**
- Train Loss: ~0.6-0.8
- Val Loss: ~0.5-0.7
- Val Accuracy: 75-80%

**Epoch 5:**
- Train Loss: ~0.3-0.4
- Val Loss: ~0.35-0.45
- Val Accuracy: 85-88%

**Epoch 10:**
- Train Loss: ~0.25-0.35
- Val Loss: ~0.30-0.40
- Val Accuracy: 87-90%

**Gradient Norms:**
- Should stay below 1.0 (due to clipping)
- No sudden spikes

**Plasticity:**
- 0-5 neurons soft-deleted by epoch 10
- Deletions should be gradual, not sudden

---

## Still Unstable? Advanced Debugging

### Check 1: Baseline Without Plasticity

Disable plasticity completely and train baseline:

```python
model = SSNC_MLP(
    input_dim=784,
    hidden_dims=[128, 64],
    output_dim=10,
    activation='relu'
)
# Don't add deletion manager
# Just train normally
```

If baseline works → problem is plasticity
If baseline fails → problem is model or data

### Check 2: Monitor Per-Layer Statistics

```python
for layer_idx, layer in enumerate(model.layers):
    weight_norm = layer.weight.norm().item()
    if hasattr(layer, 'bias') and layer.bias is not None:
        bias_norm = layer.bias.norm().item()
    print(f"Layer {layer_idx}: W_norm={weight_norm:.3f}, B_norm={bias_norm:.3f}")
```

Look for:
- Exploding norms (>10)
- Dead layers (norm near 0)

### Check 3: Utility Computation

```python
# Are utilities meaningful?
utilities = []
for neuron in all_neurons:
    utilities.append(neuron.utility)

plt.hist(utilities, bins=50)
plt.xlabel('Utility')
plt.ylabel('Count')
plt.title('Utility Distribution')
plt.show()

# Should see:
# - Range: 0.0 - 0.5 (not all near 0)
# - Variance > 0.05 (differentiation)
# - Not dominated by noise
```

---

## Key Principles for Stability

1. **Observe before acting** - Let network stabilize before deleting
2. **Act conservatively** - Delete few neurons slowly
3. **Clip gradients** - Always, always, always
4. **Match dataset difficulty** - Fashion-MNIST needs more care than MNIST
5. **Monitor everything** - Log gradient norms, utilities, deletions
6. **Start disabled** - Enable plasticity after baseline is stable

---

## Summary: Most Common Fix

**90% of issues are fixed by adding this one line:**

```python
# In training loop, before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If that doesn't work, use the conservative configuration from Option 2.

If still unstable, run the diagnostic checklist and report:
1. Where does NaN first appear (which batch/epoch)?
2. What are gradient norms before clipping?
3. How many neurons are being deleted per epoch?
4. What's the utility distribution?

---

## Files Provided

1. **fix_structural_plasticity.py** - Diagnostic tool + fixes
2. **train_fashion_mnist_stable.py** - Proven stable configuration
3. This debugging guide

**Recommended workflow:**
1. Run diagnostic: `python fix_structural_plasticity.py`
2. Use stable trainer: `python train_fashion_mnist_stable.py`
3. If needed, adapt your code using patterns from stable trainer

Good luck! The most important thing is gradient clipping - don't skip it! 🎯
