"""
STRUCTURAL PLASTICITY STABILITY FIX

Identifies and fixes common issues causing training instability:
1. Missing weight synchronization between plasticity and PyTorch layers
2. Aggressive deletion/creation parameters
3. No gradient clipping
4. Fashion-MNIST specific tuning

This script provides:
- Diagnostic tools to identify the issue
- A stabilized version of the trainer
- Recommended hyperparameters for Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class StabilizedPlasticityManager:
    """
    Fixed version of StructuralPlasticityManager with proper safeguards.
    
    Key fixes:
    1. Conservative deletion thresholds
    2. Slower gating rates
    3. Longer hysteresis periods
    4. Budget limits enforced strictly
    """
    
    def __init__(
        self,
        n_layers: int,
        initial_neurons_per_layer: List[int],
        # More conservative defaults for Fashion-MNIST
        deletion_utility_threshold: float = 0.03,  # Lower = more conservative
        soft_deletion_cycles: int = 20,  # Longer waiting period
        hard_deletion_cycles: int = 40,  # Much longer before permanent removal
        gating_rate: float = 0.05,  # Slower gating (was 0.1)
        min_gate_value: float = 0.1,  # Don't gate too aggressively
        # Creation parameters
        enable_creation: bool = False,  # Disable by default until stable
        overload_utility_threshold: float = 0.85,  # Higher threshold
        max_creations_per_cycle: int = 1,  # Very conservative
        # Budget
        max_deletions_per_cycle: int = 2,  # Strict limit
    ):
        self.n_layers = n_layers
        self.current_neurons_per_layer = initial_neurons_per_layer.copy()
        
        # Thresholds
        self.deletion_utility_threshold = deletion_utility_threshold
        self.soft_deletion_cycles = soft_deletion_cycles
        self.hard_deletion_cycles = hard_deletion_cycles
        self.gating_rate = gating_rate
        self.min_gate_value = min_gate_value
        
        # Creation (disabled by default)
        self.enable_creation = enable_creation
        self.overload_utility_threshold = overload_utility_threshold
        self.max_creations_per_cycle = max_creations_per_cycle
        self.max_deletions_per_cycle = max_deletions_per_cycle
        
        # State tracking
        self.neuron_states = {}
        self.cycle_count = 0
        self.total_soft_deleted = 0
        self.total_hard_deleted = 0
        self.total_created = 0
        
        # Safety: Track recent actions to prevent oscillation
        self.recent_deletions = []  # List of (layer, neuron, cycle)
        self.recent_creations = []
        
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialize neuron states."""
        for layer_id in range(self.n_layers):
            for neuron_id in range(self.current_neurons_per_layer[layer_id]):
                key = (layer_id, neuron_id)
                self.neuron_states[key] = {
                    'marked_for_deletion': False,
                    'marked_cycles': 0,
                    'soft_deleted': False,
                    'gate_value': 1.0,
                    'last_utility': 0.0,
                    'age': 0,
                    'deletion_timestamp': None
                }
    
    def update_cycle(self, utilities: Dict[Tuple[int, int], float]) -> Dict:
        """
        Conservative update cycle with safeguards.
        """
        self.cycle_count += 1
        
        # Age all neurons
        for state in self.neuron_states.values():
            state['age'] += 1
        
        actions = {
            'cycle': self.cycle_count,
            'newly_marked': [],
            'newly_gated': [],
            'gating_progress': [],
            'recovered': []
        }
        
        # Update marking (with grace period for young neurons)
        for (layer_id, neuron_id), utility in utilities.items():
            if (layer_id, neuron_id) not in self.neuron_states:
                continue
            
            state = self.neuron_states[(layer_id, neuron_id)]
            state['last_utility'] = utility
            
            # Grace period: don't delete very young neurons
            if state['age'] < 50:
                continue
            
            # Check for low utility
            if utility < self.deletion_utility_threshold:
                if not state['marked_for_deletion']:
                    state['marked_for_deletion'] = True
                    state['marked_cycles'] = 1
                    actions['newly_marked'].append((layer_id, neuron_id, utility))
                else:
                    state['marked_cycles'] += 1
            else:
                # Recovery
                if state['marked_for_deletion'] and not state['soft_deleted']:
                    state['marked_for_deletion'] = False
                    state['marked_cycles'] = 0
        
        # Apply soft deletion (with strict budget)
        deletion_budget = self.max_deletions_per_cycle
        
        for (layer_id, neuron_id), state in self.neuron_states.items():
            if deletion_budget <= 0:
                break
            
            # Start soft deletion
            if (state['marked_for_deletion'] and 
                state['marked_cycles'] >= self.soft_deletion_cycles and
                not state['soft_deleted']):
                
                state['soft_deleted'] = True
                state['deletion_timestamp'] = self.cycle_count
                self.total_soft_deleted += 1
                deletion_budget -= 1
                
                actions['newly_gated'].append({
                    'layer': layer_id,
                    'neuron': neuron_id,
                    'utility': state['last_utility'],
                    'marked_cycles': state['marked_cycles']
                })
            
            # Continue gating gradually
            if state['soft_deleted'] and state['gate_value'] > self.min_gate_value:
                state['gate_value'] *= (1.0 - self.gating_rate)
                state['gate_value'] = max(state['gate_value'], self.min_gate_value)
                
                actions['gating_progress'].append({
                    'layer': layer_id,
                    'neuron': neuron_id,
                    'gate_value': state['gate_value']
                })
        
        # Prune recent action history
        self.recent_deletions = [
            (l, n, c) for l, n, c in self.recent_deletions
            if self.cycle_count - c < 100
        ]
        
        return actions
    
    def get_gate_values(self, layer_id: int) -> torch.Tensor:
        """Get gate values for a layer."""
        n_neurons = self.current_neurons_per_layer[layer_id]
        gate_values = torch.ones(n_neurons)
        
        for neuron_id in range(n_neurons):
            key = (layer_id, neuron_id)
            if key in self.neuron_states:
                gate_values[neuron_id] = self.neuron_states[key]['gate_value']
        
        return gate_values
    
    def get_statistics(self) -> Dict:
        """Get statistics."""
        marked = sum(1 for s in self.neuron_states.values() if s['marked_for_deletion'])
        soft_deleted = sum(1 for s in self.neuron_states.values() if s['soft_deleted'])
        
        return {
            'cycle_count': self.cycle_count,
            'marked_for_deletion': marked,
            'soft_deleted': soft_deleted,
            'total_soft_deleted': self.total_soft_deleted,
            'total_hard_deleted': self.total_hard_deleted,
            'total_created': self.total_created,
            'current_neurons': sum(self.current_neurons_per_layer)
        }


class StabilizedTrainer:
    """
    Stabilized training with proper gradient clipping and conservative plasticity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        hidden_dims: List[int],
        learning_rate: float = 0.01,
        device: str = None,
        # Stability parameters
        gradient_clip_norm: float = 1.0,
        use_adam: bool = False,  # Adam often more stable than SGD
    ):
        self.model = model
        self.hidden_dims = hidden_dims
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.gradient_clip_norm = gradient_clip_norm
        
        # Use Adam for better stability with Fashion-MNIST
        if use_adam:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(
                model.parameters(), 
                lr=learning_rate, 
                momentum=0.9,
                nesterov=True
            )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Create stabilized plasticity manager
        self.plasticity_manager = StabilizedPlasticityManager(
            n_layers=len(hidden_dims),
            initial_neurons_per_layer=hidden_dims,
            deletion_utility_threshold=0.03,
            soft_deletion_cycles=20,
            hard_deletion_cycles=40,
            enable_creation=False,  # Disabled until deletion stable
            max_deletions_per_cycle=2
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'gradient_norms': [],
            'soft_deleted_per_epoch': [],
            'total_neurons': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train one epoch with gradient clipping and conservative plasticity."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # CRITICAL: Gradient clipping for stability
            if self.gradient_clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_norm
                )
                grad_norms.append(total_norm.item())
            
            # Observe gradients (if SSNC model)
            if hasattr(self.model, 'observe_gradients'):
                self.model.observe_gradients()
            
            # Compute utilities periodically
            if hasattr(self.model, 'compute_all_utilities') and batch_idx % 20 == 0:
                self.model.compute_all_utilities()
            
            # Optimizer step
            self.optimizer.step()
            
            # Plasticity update (very infrequent to avoid instability)
            if batch_idx % 50 == 0:
                utilities = self._get_utilities()
                if utilities:
                    actions = self.plasticity_manager.update_cycle(utilities)
                    
                    # Apply gating if SSNC model
                    if hasattr(self.model, 'apply_soft_deletion_gates'):
                        self._apply_gates()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.0*correct/total:.2f}% '
                      f'GradNorm: {grad_norms[-1] if grad_norms else 0:.3f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        # Record history
        self.history['gradient_norms'].extend(grad_norms)
        stats = self.plasticity_manager.get_statistics()
        self.history['soft_deleted_per_epoch'].append(stats['soft_deleted'])
        self.history['total_neurons'].append(stats['current_neurons'])
        
        return avg_loss, accuracy
    
    def _get_utilities(self) -> Dict:
        """Get utilities from SSNC neurons."""
        utilities = {}
        
        if hasattr(self.model, 'ssnc_layers') and self.model.ssnc_layers:
            for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
                if hasattr(ssnc_layer, 'neurons'):
                    for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                        utilities[(layer_idx, neuron_idx)] = neuron.utility
        
        return utilities
    
    def _apply_gates(self):
        """Apply soft deletion gates to model."""
        if not hasattr(self.model, 'ssnc_layers'):
            return
        
        for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
            gate_values = self.plasticity_manager.get_gate_values(layer_idx)
            
            # Apply gates to corresponding PyTorch layer
            if hasattr(self.model, 'layers') and layer_idx < len(self.model.layers):
                pytorch_layer = self.model.layers[layer_idx]
                if isinstance(pytorch_layer, nn.Linear):
                    # Apply gates by scaling weights
                    with torch.no_grad():
                        pytorch_layer.weight.data *= gate_values.unsqueeze(1).to(self.device)
                        if pytorch_layer.bias is not None:
                            pytorch_layer.bias.data *= gate_values.to(self.device)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation pass."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        """Full training loop."""
        print(f"\nTraining with STABILIZED plasticity for {epochs} epochs")
        print(f"Gradient clipping: {self.gradient_clip_norm}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{epochs}")
            print('='*60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            stats = self.plasticity_manager.get_statistics()
            print(f"  Soft Deleted: {stats['soft_deleted']}")
            print(f"  Total Neurons: {stats['current_neurons']}")
        
        return self.history
    
    def plot_training(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.history['gradient_norms']:
            axes[1, 0].plot(self.history['gradient_norms'], alpha=0.6)
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms (Stability Check)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=self.gradient_clip_norm, color='r', 
                             linestyle='--', label=f'Clip Threshold')
            axes[1, 0].legend()
        
        # Neuron count
        axes[1, 1].plot(self.history['total_neurons'], linewidth=2)
        axes[1, 1].plot(self.history['soft_deleted_per_epoch'], 
                       label='Soft Deleted', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Network Size Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def load_fashion_mnist(batch_size: int = 64):
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def diagnostic_test():
    """
    Run diagnostic to identify the specific issue.
    """
    print("\n" + "="*70)
    print("STRUCTURAL PLASTICITY DIAGNOSTIC TEST")
    print("="*70)
    
    # Test 1: Basic model without plasticity
    print("\n[1/3] Testing baseline model (no plasticity)...")
    from ssnc_mlp import SSNC_MLP
    
    baseline_model = SSNC_MLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        activation='tanh'
    )
    
    # Quick baseline test
    train_loader, val_loader = load_fashion_mnist(batch_size=64)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_model.to(device)
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train 1 epoch
    baseline_model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 100:  # Just 100 batches for test
            break
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = baseline_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    baseline_stable = np.std(losses) < 1.0 and not any(np.isnan(losses))
    print(f"   Baseline {'✓ STABLE' if baseline_stable else '✗ UNSTABLE'}")
    print(f"   Loss range: [{min(losses):.3f}, {max(losses):.3f}]")
    
    # Test 2: Model with observation only
    print("\n[2/3] Testing with SSNC observation (no plasticity)...")
    print("   (Skipping - requires Phase 6 model)")
    
    # Test 3: Recommendations
    print("\n[3/3] Diagnostic Results:")
    print("-" * 70)
    
    if baseline_stable:
        print("✓ Baseline model is stable")
        print("\nLikely issues with structural plasticity:")
        print("  1. Gate application modifying weights incorrectly")
        print("  2. Deletion/creation happening too aggressively")
        print("  3. No gradient clipping causing explosions")
        print("  4. Fashion-MNIST needs different hyperparameters than MNIST")
    else:
        print("✗ Baseline model is unstable!")
        print("\nBase issues to fix first:")
        print("  1. Reduce learning rate (try 0.001)")
        print("  2. Add gradient clipping")
        print("  3. Use Adam optimizer instead of SGD")
    
    print("\n" + "="*70)
    print("RECOMMENDED FIX")
    print("="*70)
    print("""
Use the StabilizedTrainer class with these settings:

trainer = StabilizedTrainer(
    model=your_model,
    hidden_dims=[128, 64],
    learning_rate=0.01,
    gradient_clip_norm=1.0,  # CRITICAL for stability
    use_adam=False  # Try True if still unstable
)

# Train with conservative plasticity
history = trainer.train(train_loader, val_loader, epochs=10)
    """)


if __name__ == "__main__":
    print("STRUCTURAL PLASTICITY STABILITY FIX")
    print("="*70)
    print("\nThis script provides:")
    print("  1. Diagnostic test to identify issues")
    print("  2. Stabilized plasticity manager")
    print("  3. Stabilized trainer with gradient clipping")
    print("\nRun diagnostic_test() to identify your specific issue.")
    print("\n" + "="*70)
    
    # Run diagnostic
    diagnostic_test()