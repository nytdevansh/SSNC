"""
COMPLETE SSNC TRAINING - ALL PHASES ACTIVE

Demonstrates the full self-organizing system with:
- Soft deletion (gradual gating)
- Hard deletion (permanent removal)
- Neuron creation (capacity restoration)
- Global monitoring (safety)
- Recovery testing (self-repair)

Expected behavior:
1. Early epochs: Exploration, balanced deletion/creation
2. Mid epochs: Stabilization, adaptive structure
3. Late epochs: Optimized, minimal changes
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
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.complete_ssnc import CompleteSSNC_MLP


CONFIG = {
    # Model
    'input_dim': 784,
    'hidden_dims': [384, 192],
    'output_dim': 10,
    'activation': 'relu',
    
    # Training
    'batch_size': 128,
    'epochs': 25,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0,
    
    # SSNC phases - BALANCED SETTINGS
    'deletion_threshold': 0.03,  # More selective (was 0.05)
    'marking_cycles': 30,  # Longer wait (was 20)
    'gating_rate': 0.06,  # Slower gating (was 0.08)
    'grace_period': 60,  # Longer protection (was 50)
    'hard_deletion_cycles': 60,  # Much longer (was 50)
    'creation_threshold': 0.70,  # Will use percentile instead
    
    'enable_hard_deletion': True,
    'enable_creation': True,
    'enable_monitoring': True,
    
    'max_deletions_per_cycle': 1,  # More conservative (was 2)
    'max_creations_per_cycle': 2,  # More aggressive (was 1)
    
    'utility_update_every': 10,
    'plasticity_cycle_every': 100,
    
    # Phase 11: Recovery test
    'test_recovery': True,
    'recovery_at_epoch': 18,  # Later (was 15)
    'damage_rate': 0.10,  # Less damage (was 0.15)
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def load_fashion_mnist(batch_size=128):
    """Load Fashion-MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


class CompleteSSNCTrainer:
    """Trainer for complete SSNC system."""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        self.model = CompleteSSNC_MLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            deletion_threshold=config['deletion_threshold'],
            marking_cycles=config['marking_cycles'],
            gating_rate=config['gating_rate'],
            grace_period=config['grace_period'],
            hard_deletion_cycles=config['hard_deletion_cycles'],
            creation_threshold=config['creation_threshold'],
            enable_hard_deletion=config['enable_hard_deletion'],
            enable_creation=config['enable_creation'],
            enable_monitoring=config['enable_monitoring'],
            max_deletions_per_cycle=config['max_deletions_per_cycle'],
            max_creations_per_cycle=config['max_creations_per_cycle']
        ).to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            nesterov=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'active_neurons': [],
            'soft_deleted': [],
            'hard_deleted': [],
            'created': [],
            'net_change': [],
            'utility_mean': [],
            'utility_std': [],
            'recovery_test': None
        }
        
        self.best_val_acc = 0.0
        self.recovery_tested = False
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip_norm']
            )
            
            self.model.observe_gradients()
            self.optimizer.step()
            
            # Utilities
            if batch_idx % self.config['utility_update_every'] == 0:
                self.model.compute_all_utilities()
            
            # Plasticity
            if batch_idx % self.config['plasticity_cycle_every'] == 0:
                actions = self.model.plasticity_cycle(current_loss=loss.item())
                
                # Log significant events
                if actions['hard_deleted']:
                    print(f"    ⚡ Batch {batch_idx}: Hard-deleted {len(actions['hard_deleted'])} neurons")
                if actions['created']:
                    print(f"    ✨ Batch {batch_idx}: Created {len(actions['created'])} neurons")
                if actions['monitor_warnings']:
                    print(f"    ⚠ Batch {batch_idx}: {', '.join(actions['monitor_warnings'])}")
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                stats = self.model.get_statistics()
                print(f"  [{batch_idx:3d}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100.0*correct/total:.2f}% "
                      f"| Active: {stats['active_neurons']} "
                      f"Soft: {stats['soft_deleted']} Hard: {stats['hard_deleted']} "
                      f"Created: {stats['created']} "
                      f"| Util: {stats['utility_mean']:.3f}±{stats['utility_std']:.3f}")
        
        return total_loss / len(train_loader), 100.0 * correct / total
    
    def validate(self, val_loader):
        """Validate."""
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
        
        return total_loss / len(val_loader), 100.0 * correct / total
    
    def test_recovery(self, train_loader, val_loader):
        """
        Phase 11: Recovery test.
        
        Induce damage and measure recovery capability.
        """
        print(f"\n{'='*70}")
        print("PHASE 11: RECOVERY TEST")
        print('='*70)
        
        # Measure before damage
        val_loss_before, val_acc_before = self.validate(val_loader)
        stats_before = self.model.get_statistics()
        
        print(f"Before damage:")
        print(f"  Active neurons: {stats_before['active_neurons']}")
        print(f"  Val accuracy: {val_acc_before:.2f}%")
        
        # Induce damage
        damaged = self.model.induce_damage(damage_rate=self.config['damage_rate'])
        stats_after_damage = self.model.get_statistics()
        val_loss_damaged, val_acc_damaged = self.validate(val_loader)
        
        print(f"\nAfter inducing {self.config['damage_rate']*100:.0f}% damage:")
        print(f"  Neurons deleted: {damaged}")
        print(f"  Active neurons: {stats_after_damage['active_neurons']}")
        print(f"  Val accuracy: {val_acc_damaged:.2f}%")
        print(f"  Accuracy drop: {val_acc_before - val_acc_damaged:.2f}%")
        
        # Allow recovery
        print(f"\nAllowing 5 epochs for recovery...")
        recovery_epochs = []
        
        for recovery_epoch in range(1, 6):
            train_loss, train_acc = self.train_epoch(train_loader, f"Recovery {recovery_epoch}")
            val_loss, val_acc = self.validate(val_loader)
            
            recovery_epochs.append({
                'epoch': recovery_epoch,
                'val_acc': val_acc,
                'active_neurons': self.model.get_statistics()['active_neurons']
            })
            
            print(f"  Recovery epoch {recovery_epoch}: "
                  f"Val Acc={val_acc:.2f}% "
                  f"Active={recovery_epochs[-1]['active_neurons']}")
        
        # Final recovery stats
        val_loss_recovered, val_acc_recovered = self.validate(val_loader)
        stats_recovered = self.model.get_statistics()
        
        print(f"\nRecovery results:")
        print(f"  Active neurons: {stats_recovered['active_neurons']} "
              f"(started: {stats_before['active_neurons']}, damaged: {stats_after_damage['active_neurons']})")
        print(f"  Val accuracy: {val_acc_recovered:.2f}% "
              f"(before: {val_acc_before:.2f}%, damaged: {val_acc_damaged:.2f}%)")
        print(f"  Recovery: {val_acc_recovered - val_acc_damaged:+.2f}%")
        print(f"  Final gap: {val_acc_before - val_acc_recovered:.2f}%")
        
        recovery_rate = (val_acc_recovered - val_acc_damaged) / (val_acc_before - val_acc_damaged)
        print(f"  Recovery rate: {recovery_rate*100:.1f}%")
        
        print('='*70 + '\n')
        
        self.history['recovery_test'] = {
            'before': {
                'active_neurons': stats_before['active_neurons'],
                'val_acc': val_acc_before
            },
            'damaged': {
                'neurons_deleted': damaged,
                'active_neurons': stats_after_damage['active_neurons'],
                'val_acc': val_acc_damaged,
                'accuracy_drop': val_acc_before - val_acc_damaged
            },
            'recovered': {
                'active_neurons': stats_recovered['active_neurons'],
                'val_acc': val_acc_recovered,
                'recovery': val_acc_recovered - val_acc_damaged,
                'final_gap': val_acc_before - val_acc_recovered,
                'recovery_rate': recovery_rate
            },
            'recovery_epochs': recovery_epochs
        }
        
        self.recovery_tested = True
    
    def train(self, train_loader, val_loader, epochs):
        """Full training with all phases."""
        print(f"\n{'='*70}")
        print("COMPLETE SSNC TRAINING - ALL PHASES")
        print('='*70)
        print(f"Phases enabled:")
        print(f"  ✓ Soft deletion (Phase 6)")
        print(f"  ✓ Hard deletion (Phase 7): after {self.config['hard_deletion_cycles']} cycles")
        print(f"  ✓ Creation (Phase 8): threshold {self.config['creation_threshold']}")
        print(f"  ✓ Monitoring (Phase 10): bounds [0.3x, 2.0x]")
        if self.config['test_recovery']:
            print(f"  ✓ Recovery test (Phase 11): epoch {self.config['recovery_at_epoch']}")
        print('='*70 + '\n')
        
        for epoch in range(1, epochs + 1):
            # Recovery test at specified epoch
            if (self.config['test_recovery'] and 
                epoch == self.config['recovery_at_epoch'] and 
                not self.recovery_tested):
                self.test_recovery(train_loader, val_loader)
            
            print(f"\nEpoch {epoch}/{epochs}")
            print('-'*70)
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            stats = self.model.get_statistics()
            self.history['active_neurons'].append(stats['active_neurons'])
            self.history['soft_deleted'].append(stats['soft_deleted'])
            self.history['hard_deleted'].append(stats['hard_deleted'])
            self.history['created'].append(stats['created'])
            self.history['net_change'].append(stats['net_change'])
            self.history['utility_mean'].append(stats['utility_mean'])
            self.history['utility_std'].append(stats['utility_std'])
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  Structure:")
            print(f"    Active: {stats['active_neurons']}/{sum(self.config['hidden_dims'])}")
            print(f"    Soft: {stats['soft_deleted']}, Hard: {stats['hard_deleted']}, Created: {stats['created']}")
            print(f"    Net change: {stats['net_change']:+d}")
            print(f"  Utility: {stats['utility_mean']:.4f} ± {stats['utility_std']:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"  ★ New best!")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print('='*70)
        
        return self.history
    
    def plot_results(self, save_path=None):
        """Plot comprehensive results."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history['train_acc'], label='Train', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], label='Val', linewidth=2)
        if self.config['test_recovery'] and self.history['recovery_test']:
            ax2.axvline(x=self.config['recovery_at_epoch'], color='red', 
                       linestyle='--', alpha=0.5, label='Recovery test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Active neurons
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.history['active_neurons'], linewidth=2, color='blue')
        ax3.axhline(y=sum(self.config['hidden_dims']), color='gray', 
                   linestyle='--', label='Initial')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Count')
        ax3.set_title('Active Neurons (Adaptive)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Net change
        ax4 = fig.add_subplot(gs[0, 3])
        colors = ['red' if x < 0 else 'green' for x in self.history['net_change']]
        ax4.bar(epochs, self.history['net_change'], color=colors, alpha=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Net Change')
        ax4.set_title('Net Structural Change')
        ax4.grid(True, alpha=0.3)
        
        # Soft deleted
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(epochs, self.history['soft_deleted'], linewidth=2, color='orange')
        ax5.fill_between(epochs, self.history['soft_deleted'], alpha=0.3, color='orange')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Count')
        ax5.set_title('Soft-Deleted (Phase 6)')
        ax5.grid(True, alpha=0.3)
        
        # Hard deleted
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.plot(epochs, self.history['hard_deleted'], linewidth=2, color='red')
        ax6.fill_between(epochs, self.history['hard_deleted'], alpha=0.3, color='red')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Count')
        ax6.set_title('Hard-Deleted (Phase 7)')
        ax6.grid(True, alpha=0.3)
        
        # Created
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.plot(epochs, self.history['created'], linewidth=2, color='green')
        ax7.fill_between(epochs, self.history['created'], alpha=0.3, color='green')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Count')
        ax7.set_title('Created Neurons (Phase 8)')
        ax7.grid(True, alpha=0.3)
        
        # All structural changes
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.plot(epochs, self.history['soft_deleted'], label='Soft', linewidth=2, alpha=0.7)
        ax8.plot(epochs, self.history['hard_deleted'], label='Hard', linewidth=2, alpha=0.7)
        ax8.plot(epochs, self.history['created'], label='Created', linewidth=2, alpha=0.7)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Count')
        ax8.set_title('All Structural Changes')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Utility mean
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.plot(epochs, self.history['utility_mean'], linewidth=2, color='purple')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Mean Utility')
        ax9.set_title('Average Utility')
        ax9.grid(True, alpha=0.3)
        
        # Utility std
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.plot(epochs, self.history['utility_std'], linewidth=2, color='brown')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Utility Std')
        ax10.set_title('Utility Variance')
        ax10.grid(True, alpha=0.3)
        
        # Recovery test results (if available)
        if self.history['recovery_test']:
            ax11 = fig.add_subplot(gs[2, 2:])
            recovery = self.history['recovery_test']
            
            stages = ['Before', 'Damaged', 'Recovered']
            accuracies = [
                recovery['before']['val_acc'],
                recovery['damaged']['val_acc'],
                recovery['recovered']['val_acc']
            ]
            neurons = [
                recovery['before']['active_neurons'],
                recovery['damaged']['active_neurons'],
                recovery['recovered']['active_neurons']
            ]
            
            x = np.arange(len(stages))
            width = 0.35
            
            ax11_acc = ax11
            ax11_neurons = ax11.twinx()
            
            bars1 = ax11_acc.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
            bars2 = ax11_neurons.bar(x + width/2, neurons, width, label='Neurons', color='green', alpha=0.7)
            
            ax11_acc.set_xlabel('Recovery Stage')
            ax11_acc.set_ylabel('Accuracy (%)', color='blue')
            ax11_neurons.set_ylabel('Active Neurons', color='green')
            ax11_acc.set_title('Phase 11: Recovery Test Results')
            ax11_acc.set_xticks(x)
            ax11_acc.set_xticks(x)
            ax11_acc.set_xticklabels(stages)
            ax11_acc.tick_params(axis='y', labelcolor='blue')
            ax11_neurons.tick_params(axis='y', labelcolor='green')
            ax11_acc.grid(True, alpha=0.3)
            
            # Add recovery rate text
            recovery_rate = recovery['recovered']['recovery_rate']
            ax11_acc.text(0.5, 0.95, f"Recovery rate: {recovery_rate*100:.1f}%",
                         transform=ax11_acc.transAxes, ha='center', va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()


def main():
    """Main training."""
    print("\n" + "="*70)
    print("COMPLETE SSNC - FASHION-MNIST")
    print("="*70)
    
    train_loader, val_loader = load_fashion_mnist(CONFIG['batch_size'])
    
    trainer = CompleteSSNCTrainer(CONFIG)
    history = trainer.train(train_loader, val_loader, CONFIG['epochs'])
    
    # Save results
    output_dir = Path('./results/complete_ssnc_all_phases')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'complete_results.png'
    trainer.plot_results(save_path=plot_path)
    
    torch.save(trainer.model.state_dict(), output_dir / 'model.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    final_stats = trainer.model.get_statistics()
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"Best Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"\nStructural Evolution:")
    print(f"  Initial: {sum(CONFIG['hidden_dims'])} neurons")
    print(f"  Final active: {final_stats['active_neurons']}")
    print(f"  Soft-deleted: {final_stats['soft_deleted']}")
    print(f"  Hard-deleted: {final_stats['hard_deleted']}")
    print(f"  Created: {final_stats['created']}")
    print(f"  Net change: {final_stats['net_change']:+d}")
    print(f"\nFinal Utility: {final_stats['utility_mean']:.4f} ± {final_stats['utility_std']:.4f}")
    
    if history['recovery_test']:
        print(f"\nRecovery Test (Phase 11):")
        rec = history['recovery_test']
        print(f"  Damage induced: {rec['damaged']['neurons_deleted']} neurons")
        print(f"  Accuracy drop: {rec['damaged']['accuracy_drop']:.2f}%")
        print(f"  Recovery: {rec['recovered']['recovery']:+.2f}%")
        print(f"  Recovery rate: {rec['recovered']['recovery_rate']*100:.1f}%")
    
    print('='*70 + '\n')


if __name__ == "__main__":
    main()
