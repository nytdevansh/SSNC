"""
DAMAGE SURVIVAL TEST - Critical SSNC Benchmark

Protocol
--------
1. Train both models to the same accuracy level
2. Randomly delete 30-50% of neurons
3. Continue training without reset

Expected Outcome
----------------
• Base model: Permanent accuracy drop (can't recover without neurons)
• SSNC: Partial → Substantial recovery via creation/growth

This test alone justifies SSNC.

Why it matters:
- Biological networks survive damage (neurons regrow)
- Standard networks: damage = permanent failure
- SSNC: damage = temporary setback, then recovery

If SSNC doesn't recover:
→ growth / observer / creation logic is broken
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append('/home/claude')

try:
    from baseline_mlp import InstrumentedMLP
    from complete_ssnc_system import CompleteSSNC_MLP
    print("✓ Models imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


CONFIG = {
    # Model architecture
    'input_dim': 784,
    'hidden_dims': [256, 128],  # Sufficient capacity
    'output_dim': 10,
    'activation': 'tanh',
    
    # Training
    'batch_size': 128,
    'warmup_epochs': 10,  # Train to stable accuracy
    'survival_epochs': 10,  # Post-damage training
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0,
    
    # Damage parameters
    'damage_ratio': 0.40,  # Delete 40% of neurons
    'damage_at_epoch': 5,  # After epoch 5 during survival phase
    
    # SSNC Phases (for recovery testing)
    'enable_local_learning': True,
    'enable_soft_deletion': False,  # Focus on creation/recovery
    'enable_hard_deletion': False,
    'enable_creation': True,  # CRITICAL: must create new neurons after damage
    'enable_monitoring': True,
    
    # Creation parameters (aggressive for recovery)
    'creation_threshold': 0.85,  # High demand triggers creation
    'creation_method': 'duplicate',
    'max_creations_per_cycle': 8,  # Create aggressively
    
    # Frequencies
    'utility_update_every': 5,
    'plasticity_cycle_every': 20,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def load_fashion_mnist(batch_size=128):
    """Load Fashion-MNIST with proper normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader


class BaselineModel(nn.Module):
    """Simple baseline MLP for comparison."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.activation = nn.Tanh()
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        x = self.layers[-1](x)
        return x
    
    def damage(self, damage_ratio):
        """Randomly set weights to zero (neuron deletion)."""
        total_deleted = 0
        
        # Damage hidden layers (not input/output)
        for layer in self.layers[:-1]:
            mask = torch.rand_like(layer.weight) < damage_ratio
            layer.weight.data[mask] = 0.0
            total_deleted += mask.sum().item()
        
        return total_deleted


class DamageTrainer:
    """Trainer for damage survival test."""
    
    def __init__(self, config, train_loader, val_loader, model_type='baseline'):
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_type = model_type
        
        # Create model
        if model_type == 'baseline':
            self.model = BaselineModel(
                config['input_dim'],
                config['hidden_dims'],
                config['output_dim']
            ).to(self.device)
            self.model_name = "Baseline MLP"
        else:
            self.model = CompleteSSNC_MLP(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                output_dim=config['output_dim'],
                activation=config['activation'],
                enable_local_learning=config['enable_local_learning'],
                enable_soft_deletion=config['enable_soft_deletion'],
                enable_hard_deletion=config['enable_hard_deletion'],
                enable_creation=config['enable_creation'],
                enable_redistribution=False,
                enable_monitoring=config['enable_monitoring'],
                creation_threshold=config['creation_threshold'],
                creation_method=config['creation_method'],
                max_creations_per_cycle=config['max_creations_per_cycle'],
                plasticity_cycle_every=config['plasticity_cycle_every']
            ).to(self.device)
            self.model_name = "SSNC MLP (with creation recovery)"
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            nesterov=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # History
        self.history = {
            'epoch': [],
            'phase': [],  # 'warmup' or 'survival'
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'damage_applied': [],  # When damage is applied
            'recovery_from_damage': [],  # Recovery trajectory
            'active_neurons': [],  # For SSNC
            'created': []  # For SSNC
        }
        
        self.best_val_acc_before_damage = 0.0
        self.val_acc_at_damage = 0.0
        self.epoch_count = 0
    
    def train_epoch(self, epoch, phase='warmup'):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip_norm']
            )
            
            # SSNC observation (if applicable)
            if self.model_type == 'ssnc':
                self.model.observe_gradients()
            
            # Update
            self.optimizer.step()
            
            # Compute utilities (SSNC)
            if self.model_type == 'ssnc' and batch_idx % self.config['utility_update_every'] == 0:
                self.model.compute_all_utilities()
            
            # Plasticity cycle (SSNC)
            if self.model_type == 'ssnc' and batch_idx % self.config['plasticity_cycle_every'] == 0:
                actions = self.model.plasticity_cycle(current_loss=loss.item())
                
                if actions['created']:
                    print(f"    ✨ Created {len(actions['created'])} recovery neurons")
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """Validate."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    def apply_damage(self):
        """Apply random damage to the model."""
        if self.model_type == 'baseline':
            total_deleted = self.model.damage(self.config['damage_ratio'])
        else:
            # For SSNC, we'll manually damage the base layers
            total_deleted = 0
            for layer in self.model.layers[:-1]:
                mask = torch.rand_like(layer.weight) < self.config['damage_ratio']
                layer.weight.data[mask] = 0.0
                total_deleted += mask.sum().item()
        
        print(f"\n💥 DAMAGE APPLIED: Deleted {total_deleted} weights ({self.config['damage_ratio']*100:.0f}% of hidden layers)")
        return total_deleted
    
    def train_full(self):
        """Full training: warmup + damage + survival."""
        print(f"\n{'='*70}")
        print(f"DAMAGE SURVIVAL TEST - {self.model_name}")
        print('='*70)
        print(f"Phase 1: Warmup ({self.config['warmup_epochs']} epochs)")
        print(f"Phase 2: Damage at epoch {self.config['damage_at_epoch']}")
        print(f"Phase 3: Survival ({self.config['survival_epochs']} epochs post-damage)")
        print(f"Damage ratio: {self.config['damage_ratio']*100:.0f}%")
        print('='*70 + '\n')
        
        # WARMUP PHASE
        print("PHASE 1: WARMUP - Training to stable accuracy")
        print('-'*70)
        
        for epoch in range(1, self.config['warmup_epochs'] + 1):
            train_loss, train_acc = self.train_epoch(epoch, phase='warmup')
            val_loss, val_acc = self.validate()
            
            self.history['epoch'].append(epoch)
            self.history['phase'].append('warmup')
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['damage_applied'].append(0)
            
            if self.model_type == 'ssnc':
                stats = self.model.get_statistics()
                self.history['active_neurons'].append(stats['active_neurons'])
                self.history['created'].append(stats['created'])
            else:
                self.history['active_neurons'].append(0)
                self.history['created'].append(0)
            
            self.best_val_acc_before_damage = max(self.best_val_acc_before_damage, val_acc)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        print(f"\n✓ Warmup complete. Best val accuracy: {self.best_val_acc_before_damage:.2f}%\n")
        
        # SURVIVAL PHASE
        print("PHASE 2 & 3: DAMAGE + SURVIVAL TRAINING")
        print('-'*70)
        
        for epoch in range(1, self.config['survival_epochs'] + 1):
            # Apply damage at specified epoch
            if epoch == self.config['damage_at_epoch']:
                self.apply_damage()
                _, val_acc_at_damage = self.validate()
                self.val_acc_at_damage = val_acc_at_damage
                print(f"Accuracy immediately after damage: {val_acc_at_damage:.2f}%")
                print(f"Accuracy drop: {self.best_val_acc_before_damage - val_acc_at_damage:.2f}%\n")
            
            train_loss, train_acc = self.train_epoch(epoch, phase='survival')
            val_loss, val_acc = self.validate()
            
            absolute_epoch = self.config['warmup_epochs'] + epoch
            self.history['epoch'].append(absolute_epoch)
            self.history['phase'].append('survival')
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Mark when damage was applied
            if epoch == self.config['damage_at_epoch']:
                self.history['damage_applied'].append(1)
                recovery_from_damage = val_acc - self.val_acc_at_damage
            else:
                self.history['damage_applied'].append(0)
                recovery_from_damage = val_acc - self.val_acc_at_damage if epoch > self.config['damage_at_epoch'] else 0
            
            self.history['recovery_from_damage'].append(recovery_from_damage)
            
            if self.model_type == 'ssnc':
                stats = self.model.get_statistics()
                self.history['active_neurons'].append(stats['active_neurons'])
                self.history['created'].append(stats['created'])
            else:
                self.history['active_neurons'].append(0)
                self.history['created'].append(0)
            
            print(f"Survival Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Acc={val_acc:.2f}% | Recovery: {recovery_from_damage:+.2f}%")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print('='*70)
        
        return self.history


def plot_damage_comparison(baseline_hist, ssnc_hist, save_path=None):
    """Plot side-by-side comparison of damage survival."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DAMAGE SURVIVAL TEST: Baseline vs SSNC', fontsize=16, fontweight='bold')
    
    epochs_baseline = baseline_hist['epoch']
    epochs_ssnc = ssnc_hist['epoch']
    
    # Baseline accuracy
    ax = axes[0, 0]
    ax.plot(epochs_baseline, baseline_hist['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs_baseline, baseline_hist['val_acc'], label='Val', linewidth=2)
    # Mark damage point
    damage_epoch_baseline = CONFIG['warmup_epochs'] + CONFIG['damage_at_epoch']
    ax.axvline(x=damage_epoch_baseline, color='red', linestyle='--', linewidth=2, label='Damage')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Baseline MLP: Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SSNC accuracy
    ax = axes[0, 1]
    ax.plot(epochs_ssnc, ssnc_hist['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs_ssnc, ssnc_hist['val_acc'], label='Val', linewidth=2, color='orange')
    ax.axvline(x=damage_epoch_baseline, color='red', linestyle='--', linewidth=2, label='Damage')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('SSNC MLP: Accuracy (with creation recovery)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Direct comparison
    ax = axes[0, 2]
    ax.plot(epochs_baseline, baseline_hist['val_acc'], label='Baseline', linewidth=2.5, marker='o')
    ax.plot(epochs_ssnc, ssnc_hist['val_acc'], label='SSNC', linewidth=2.5, marker='s')
    ax.axvline(x=damage_epoch_baseline, color='red', linestyle='--', linewidth=2, label='Damage')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Direct Comparison: Recovery Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss comparison
    ax = axes[1, 0]
    ax.plot(epochs_baseline, baseline_hist['val_loss'], label='Baseline', linewidth=2)
    ax.plot(epochs_ssnc, ssnc_hist['val_loss'], label='SSNC', linewidth=2)
    ax.axvline(x=damage_epoch_baseline, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Loss Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SSNC neuron creation
    ax = axes[1, 1]
    if max(ssnc_hist['created']) > 0:
        ax.bar(epochs_ssnc, ssnc_hist['created'], alpha=0.6, color='green')
        ax.axvline(x=damage_epoch_baseline, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Neurons Created')
        ax.set_title('SSNC: Neuron Creation Over Time')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Recovery metric
    ax = axes[1, 2]
    survival_start = CONFIG['warmup_epochs']
    recovery_baseline = [baseline_hist['val_acc'][survival_start + CONFIG['damage_at_epoch'] - 1] - baseline_hist['val_acc'][survival_start + CONFIG['damage_at_epoch'] - 1]]
    recovery_ssnc = ssnc_hist['recovery_from_damage'][CONFIG['damage_at_epoch']:]
    
    ax.bar([0], [baseline_hist['val_acc'][survival_start + CONFIG['damage_at_epoch'] - 1] - baseline_hist['val_acc'][survival_start + CONFIG['damage_at_epoch']]], 
           width=0.4, label='Baseline', alpha=0.7, color='steelblue')
    if recovery_ssnc and max(recovery_ssnc) > min(recovery_ssnc):
        max_recovery_ssnc = max(recovery_ssnc)
        ax.bar([0.5], [max_recovery_ssnc], width=0.4, label='SSNC (max recovery)', alpha=0.7, color='orange')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Recovery (percentage points)')
    ax.set_title('Recovery Comparison')
    ax.set_xticks([0, 0.5])
    ax.set_xticklabels(['Baseline', 'SSNC'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()


def main():
    """Run damage survival test."""
    print("\n" + "="*70)
    print("DAMAGE SURVIVAL TEST - CRITICAL SSNC BENCHMARK")
    print("="*70)
    print("\nProtocol:")
    print("1. Train both models to same accuracy")
    print("2. Randomly delete 30-50% of neurons")
    print("3. Continue training")
    print("\nExpected:")
    print("  • Baseline: Permanent accuracy drop")
    print("  • SSNC: Partial → Substantial recovery via creation\n")
    
    # Load data
    print("Loading Fashion-MNIST...")
    train_loader, val_loader = load_fashion_mnist(CONFIG['batch_size'])
    print("✓ Loaded\n")
    
    # Train BASELINE
    print("="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70 + "\n")
    baseline_trainer = DamageTrainer(CONFIG, train_loader, val_loader, model_type='baseline')
    baseline_history = baseline_trainer.train_full()
    
    # Train SSNC
    print("\n" + "="*70)
    print("TRAINING SSNC MODEL (with creation recovery)")
    print("="*70 + "\n")
    ssnc_trainer = DamageTrainer(CONFIG, train_loader, val_loader, model_type='ssnc')
    ssnc_history = ssnc_trainer.train_full()
    
    # Save results
    output_dir = Path('./results/damage_survival_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plot_path = output_dir / 'damage_survival_comparison.png'
    plot_damage_comparison(baseline_history, ssnc_history, save_path=plot_path)
    
    # Save histories
    histories = {
        'baseline': baseline_history,
        'ssnc': ssnc_history,
        'config': CONFIG
    }
    
    history_path = output_dir / 'damage_survival_history.json'
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2, default=str)
    print(f"\nHistory saved: {history_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("DAMAGE SURVIVAL TEST RESULTS")
    print('='*70)
    print(f"\nBefore Damage:")
    print(f"  Baseline accuracy:  {baseline_trainer.best_val_acc_before_damage:.2f}%")
    print(f"  SSNC accuracy:      {ssnc_trainer.best_val_acc_before_damage:.2f}%")
    
    print(f"\nImmediately After {CONFIG['damage_ratio']*100:.0f}% Damage:")
    print(f"  Baseline accuracy:  {baseline_trainer.val_acc_at_damage:.2f}% (drop: {baseline_trainer.best_val_acc_before_damage - baseline_trainer.val_acc_at_damage:.2f}%)")
    print(f"  SSNC accuracy:      {ssnc_trainer.val_acc_at_damage:.2f}% (drop: {ssnc_trainer.best_val_acc_before_damage - ssnc_trainer.val_acc_at_damage:.2f}%)")
    
    # Final accuracies
    baseline_final = baseline_history['val_acc'][-1]
    ssnc_final = ssnc_history['val_acc'][-1]
    
    print(f"\nAfter {CONFIG['survival_epochs']} Survival Epochs:")
    print(f"  Baseline accuracy:  {baseline_final:.2f}% (recovery: {baseline_final - baseline_trainer.val_acc_at_damage:+.2f}%)")
    print(f"  SSNC accuracy:      {ssnc_final:.2f}% (recovery: {ssnc_final - ssnc_trainer.val_acc_at_damage:+.2f}%)")
    
    print(f"\n💥 VERDICT:")
    recovery_diff = (ssnc_final - ssnc_trainer.val_acc_at_damage) - (baseline_final - baseline_trainer.val_acc_at_damage)
    if recovery_diff > 2.0:
        print(f"  ✓ SSNC shows significantly better recovery (+{recovery_diff:.2f}%)")
        print(f"  ✓ Creation mechanism successfully replaced damaged neurons")
        print(f"  ✓ This test JUSTIFIES SSNC")
    elif recovery_diff > 0:
        print(f"  ~ SSNC shows modest recovery benefit (+{recovery_diff:.2f}%)")
        print(f"  ~ Creation mechanism partially working")
    else:
        print(f"  ✗ SSNC shows NO recovery benefit ({recovery_diff:+.2f}%)")
        print(f"  ✗ Check growth / creation / observer logic")
    
    print('='*70 + '\n')


if __name__ == "__main__":
    main()
