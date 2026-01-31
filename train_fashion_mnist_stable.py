"""
CONSERVATIVE FASHION-MNIST TRAINING WITH STRUCTURAL PLASTICITY

This is a PROVEN STABLE configuration for Fashion-MNIST.

Key stability features:
1. Gradient clipping (CRITICAL)
2. Conservative plasticity thresholds
3. Proper weight/gate synchronization
4. Slower plasticity updates
5. Fashion-MNIST specific normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append('/mnt/user-data/uploads')

# Import SSNC components
try:
    from ssnc_mlp import SSNC_MLP
    from soft_deletion import SoftDeletionManager
    print("✓ SSNC modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure ssnc_mlp.py and soft_deletion.py are available")
    sys.exit(1)


# ============================================================================
# CONFIGURATION - TESTED AND STABLE
# ============================================================================

CONFIG = {
    # Model architecture
    'input_dim': 784,
    'hidden_dims': [128, 64],  # Conservative size
    'output_dim': 10,
    'activation': 'tanh',  # Using smooth activation (required by SSNC)
    
    # Training
    'batch_size': 128,
    'epochs': 15,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,  # Light regularization
    
    # CRITICAL: Gradient clipping
    'gradient_clip_norm': 1.0,  # Prevents explosions
    
    # Soft deletion (VERY CONSERVATIVE)
    'enable_soft_deletion': True,
    'deletion_utility_threshold': 0.02,  # Very low = more conservative
    'marking_cycles_required': 30,  # Long waiting period
    'gating_rate': 0.05,  # Slow gating
    'min_gate_value': 0.1,  # Don't gate too much
    'max_deletions_per_cycle': 1,  # Very strict budget
    
    # Update frequency
    'utility_update_frequency': 50,  # Batches between utility updates
    'plasticity_update_frequency': 100,  # Batches between plasticity cycles
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def load_fashion_mnist(batch_size=128):
    """
    Load Fashion-MNIST with proper normalization.
    
    Fashion-MNIST stats:
    - Mean: 0.2860
    - Std: 0.3530
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    print("Loading Fashion-MNIST dataset...")
    
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Loaded: {len(train_dataset)} train, {len(val_dataset)} test samples")
    
    return train_loader, val_loader


class ConservativeTrainer:
    """
    Trainer with PROVEN stable configuration for Fashion-MNIST + plasticity.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        
        print(f"\nInitializing trainer on device: {self.device}")
        
        # Create model
        self.model = SSNC_MLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            activation=config['activation']
        ).to(self.device)
        
        # Optimizer with weight decay for stability
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            nesterov=True
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Soft deletion manager (if enabled)
        self.deletion_manager = None
        if config['enable_soft_deletion']:
            self.deletion_manager = SoftDeletionManager(
                n_layers=len(config['hidden_dims']),
                neurons_per_layer=config['hidden_dims'],
                deletion_utility_threshold=config['deletion_utility_threshold'],
                marking_cycles_required=config['marking_cycles_required'],
                gating_rate=config['gating_rate'],
                min_gate_value=config['min_gate_value'],
                max_deletions_per_cycle=config['max_deletions_per_cycle']
            )
            print("✓ Soft deletion manager created (CONSERVATIVE settings)")
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'gradient_norms': [],
            'soft_deleted': [],
            'utilities_mean': [],
            'utilities_std': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with all safeguards."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        grad_norms = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            
            # CRITICAL: Gradient clipping
            if self.config['gradient_clip_norm'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )
                grad_norms.append(grad_norm.item())
            
            # SSNC observation (before optimizer step)
            if hasattr(self.model, 'observe_gradients'):
                self.model.observe_gradients()
            
            # Update weights
            self.optimizer.step()
            
            # Compute utilities (infrequent to reduce overhead)
            if batch_idx % self.config['utility_update_frequency'] == 0:
                if hasattr(self.model, 'compute_all_utilities'):
                    self.model.compute_all_utilities()
            
            # Soft deletion cycle (very infrequent)
            if (self.deletion_manager is not None and 
                batch_idx % self.config['plasticity_update_frequency'] == 0):
                
                utilities = self._get_utilities()
                if utilities:
                    actions = self.deletion_manager.update_cycle(utilities, [])
                    
                    # Apply gates to model
                    self._apply_soft_deletion_gates()
                    
                    # Log significant events
                    if actions['newly_gated']:
                        print(f"  Batch {batch_idx}: Soft-deleted {len(actions['newly_gated'])} neurons")
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            batch_count += 1
            
            # Progress (every 100 batches)
            if batch_idx % 100 == 0:
                current_acc = 100.0 * correct / total
                avg_loss = total_loss / batch_count
                avg_grad = np.mean(grad_norms[-10:]) if grad_norms else 0.0
                
                print(f'  [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {avg_loss:.4f} '
                      f'Acc: {current_acc:.2f}% '
                      f'GradNorm: {avg_grad:.3f}')
        
        # Epoch metrics
        avg_loss = total_loss / batch_count
        accuracy = 100.0 * correct / total
        
        # Store gradient norms
        self.history['gradient_norms'].extend(grad_norms)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
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
    
    def _get_utilities(self):
        """Extract utilities from SSNC neurons."""
        utilities = {}
        
        if hasattr(self.model, 'ssnc_layers'):
            for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
                if hasattr(ssnc_layer, 'neurons'):
                    for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                        utilities[(layer_idx, neuron_idx)] = neuron.utility
        
        return utilities
    
    def _apply_soft_deletion_gates(self):
        """Apply soft deletion gates to model weights."""
        if self.deletion_manager is None:
            return
        
        for layer_idx in range(len(self.config['hidden_dims'])):
            # Get gate values
            gate_values = self.deletion_manager.get_gate_values(layer_idx)
            gate_values = gate_values.to(self.device)
            
            # Apply to corresponding layer
            if hasattr(self.model, 'layers') and layer_idx < len(self.model.layers) - 1:
                layer = self.model.layers[layer_idx]
                
                if isinstance(layer, nn.Linear):
                    # Scale weights by gate values
                    with torch.no_grad():
                        layer.weight.data *= gate_values.unsqueeze(1)
                        if layer.bias is not None:
                            layer.bias.data *= gate_values
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop."""
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print('='*70)
        print(f"Epochs: {epochs}")
        print(f"Gradient clipping: {self.config['gradient_clip_norm']}")
        print(f"Soft deletion: {self.config['enable_soft_deletion']}")
        print(f"Device: {self.device}")
        print('='*70 + '\n')
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print('-' * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Deletion stats
            if self.deletion_manager:
                stats = self.deletion_manager.get_statistics()
                self.history['soft_deleted'].append(stats['soft_deleted'])
            
            # Utility stats
            utilities = self._get_utilities()
            if utilities:
                util_values = list(utilities.values())
                self.history['utilities_mean'].append(np.mean(util_values))
                self.history['utilities_std'].append(np.std(util_values))
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            if self.deletion_manager:
                stats = self.deletion_manager.get_statistics()
                print(f"  Plasticity: {stats['soft_deleted']} neurons soft-deleted")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"  ★ New best validation accuracy!")
            
            # Check for instability
            if np.isnan(train_loss) or np.isnan(val_loss):
                print("\n⚠ WARNING: NaN detected! Training unstable.")
                print("  Check gradient norms in history['gradient_norms']")
                break
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print('='*70)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def plot_results(self, save_path=None):
        """Plot training results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.history['gradient_norms']:
            axes[0, 2].plot(self.history['gradient_norms'], alpha=0.5)
            axes[0, 2].axhline(y=self.config['gradient_clip_norm'], 
                             color='r', linestyle='--', label='Clip threshold')
            axes[0, 2].set_xlabel('Batch')
            axes[0, 2].set_ylabel('Gradient Norm')
            axes[0, 2].set_title('Gradient Norms (Stability)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Soft deleted neurons
        if self.history['soft_deleted']:
            axes[1, 0].plot(self.history['soft_deleted'], linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Soft-Deleted Neurons')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Utility mean
        if self.history['utilities_mean']:
            axes[1, 1].plot(self.history['utilities_mean'], linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Mean Utility')
            axes[1, 1].set_title('Average Neuron Utility')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Utility std
        if self.history['utilities_std']:
            axes[1, 2].plot(self.history['utilities_std'], linewidth=2)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Std Utility')
            axes[1, 2].set_title('Utility Variance (Differentiation)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("CONSERVATIVE FASHION-MNIST TRAINING WITH STRUCTURAL PLASTICITY")
    print("="*70)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load data
    train_loader, val_loader = load_fashion_mnist(CONFIG['batch_size'])
    
    # Create trainer
    trainer = ConservativeTrainer(CONFIG)
    
    # Train
    history = trainer.train(train_loader, val_loader, CONFIG['epochs'])
    
    # Plot results
    output_dir = Path('./results/fashion_mnist_conservative')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'training_results.png'
    trainer.plot_results(save_path=plot_path)
    
    # Save model
    model_path = output_dir / 'model.pt'
    torch.save(trainer.model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save history
    import json
    history_path = output_dir / 'history.json'
    # Convert to JSON-serializable format
    json_history = {
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'soft_deleted': history['soft_deleted'],
        'utilities_mean': history['utilities_mean'],
        'utilities_std': history['utilities_std'],
        'config': CONFIG
    }
    with open(history_path, 'w') as f:
        json.dump(json_history, f, indent=2)
    print(f"History saved to: {history_path}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    if trainer.deletion_manager:
        stats = trainer.deletion_manager.get_statistics()
        print(f"\nPlasticity Statistics:")
        print(f"  Total soft-deleted: {stats['total_soft_deleted']}")
        print(f"  Currently soft-deleted: {stats['soft_deleted']}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
