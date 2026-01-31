"""
COMPLETE SSNC TRAINING - ALL PHASES ACTIVE

This demonstrates the full self-organizing system:
- Soft deletion removes low-utility neurons
- Hard deletion permanently reclaims resources
- Creation restores capacity for overloaded neurons
- Global monitoring prevents failures
- System self-stabilizes and adapts

Expected behavior:
1. Early epochs: Network explores, some deletions
2. Mid epochs: Stabilizes, balanced creation/deletion
3. Late epochs: Optimized structure, minimal changes
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
from complete_ssnc_system import CompleteSSNC_MLP


CONFIG = {
    # Model
    'input_dim': 784,
    'hidden_dims': [384, 192],  # Start large to demonstrate deletion
    'output_dim': 10,
    'activation': 'relu',
    
    # Training
    'batch_size': 128,
    'epochs': 25,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0,
    
    # SSNC Phases
    'enable_local_learning': True,
    'enable_soft_deletion': True,
    'enable_hard_deletion': True,  # Phase 7
    'enable_creation': True,  # Phase 8
    'enable_redistribution': False,  # Phase 9 (placeholder)
    'enable_monitoring': True,  # Phase 10
    
    # Deletion parameters
    'soft_deletion_threshold': 0.10,  # Higher = more aggressive
    'soft_marking_cycles': 12,
    'hard_deletion_cycles': 25,
    
    # Creation parameters
    'creation_threshold': 0.70,
    'creation_method': 'duplicate',
    
    # Budgets
    'max_deletions_per_cycle': 3,
    'max_creations_per_cycle': 2,
    
    # Frequencies
    'utility_update_every': 10,
    'plasticity_cycle_every': 50,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def load_fashion_mnist(batch_size=128):
    """Load Fashion-MNIST."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


class CompleteSSNCTrainer:
    """Trainer for complete SSNC system."""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Create complete model
        self.model = CompleteSSNC_MLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            enable_local_learning=config['enable_local_learning'],
            enable_soft_deletion=config['enable_soft_deletion'],
            enable_hard_deletion=config['enable_hard_deletion'],
            enable_creation=config['enable_creation'],
            enable_redistribution=config['enable_redistribution'],
            enable_monitoring=config['enable_monitoring'],
            soft_deletion_threshold=config['soft_deletion_threshold'],
            soft_marking_cycles=config['soft_marking_cycles'],
            hard_deletion_cycles=config['hard_deletion_cycles'],
            creation_threshold=config['creation_threshold'],
            creation_method=config['creation_method'],
            max_deletions_per_cycle=config['max_deletions_per_cycle'],
            max_creations_per_cycle=config['max_creations_per_cycle'],
            plasticity_cycle_every=config['plasticity_cycle_every']
        ).to(self.device)
        
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
            'plasticity_events': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
            
            # SSNC observation
            self.model.observe_gradients()
            
            # Update
            self.optimizer.step()
            
            # Compute utilities
            if batch_idx % self.config['utility_update_every'] == 0:
                self.model.compute_all_utilities()
            
            # Plasticity cycle
            if batch_idx % self.config['plasticity_cycle_every'] == 0:
                actions = self.model.plasticity_cycle(current_loss=loss.item())
                
                # Log significant events
                if (actions['soft_gated'] or actions['hard_deleted'] or 
                    actions['created'] or actions['monitor_warnings']):
                    
                    event = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'soft_gated': len(actions['soft_gated']),
                        'hard_deleted': len(actions['hard_deleted']),
                        'created': len(actions['created']),
                        'warnings': actions['monitor_warnings']
                    }
                    self.history['plasticity_events'].append(event)
                    
                    # Print important events
                    if actions['hard_deleted']:
                        print(f"    ⚡ Batch {batch_idx}: Hard-deleted {len(actions['hard_deleted'])} neurons")
                    if actions['created']:
                        print(f"    ✨ Batch {batch_idx}: Created {len(actions['created'])} neurons")
                    if actions['monitor_warnings']:
                        print(f"    ⚠ Batch {batch_idx}: {actions['monitor_warnings']}")
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Progress
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
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop."""
        print(f"\n{'='*70}")
        print("COMPLETE SSNC TRAINING - ALL PHASES ACTIVE")
        print('='*70)
        print(f"Phases enabled:")
        print(f"  ✓ Soft deletion (Phase 6)")
        print(f"  ✓ Hard deletion (Phase 7)")
        print(f"  ✓ Creation (Phase 8)")
        print(f"  ✓ Monitoring (Phase 10)")
        print(f"\nConfiguration:")
        print(f"  Initial neurons: {sum(self.config['hidden_dims'])}")
        print(f"  Soft threshold: {self.config['soft_deletion_threshold']}")
        print(f"  Hard cycles: {self.config['hard_deletion_cycles']}")
        print(f"  Creation threshold: {self.config['creation_threshold']}")
        print('='*70 + '\n')
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print('-'*70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Statistics
            stats = self.model.get_statistics()
            self.history['active_neurons'].append(stats['active_neurons'])
            self.history['soft_deleted'].append(stats['soft_deleted'])
            self.history['hard_deleted'].append(stats['hard_deleted'])
            self.history['created'].append(stats['created'])
            self.history['net_change'].append(stats['net_change'])
            self.history['utility_mean'].append(stats['utility_mean'])
            self.history['utility_std'].append(stats['utility_std'])
            
            # Summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  Structure:")
            print(f"    Active: {stats['active_neurons']}")
            print(f"    Soft-deleted: {stats['soft_deleted']}")
            print(f"    Hard-deleted: {stats['hard_deleted']}")
            print(f"    Created: {stats['created']}")
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
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
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
        ax3.set_title('Active Neurons (Adaptive Structure)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Soft deleted
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(epochs, self.history['soft_deleted'], linewidth=2, color='orange')
        ax4.fill_between(epochs, self.history['soft_deleted'], alpha=0.3, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Count')
        ax4.set_title('Soft-Deleted Neurons (Phase 6)')
        ax4.grid(True, alpha=0.3)
        
        # Hard deleted
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, self.history['hard_deleted'], linewidth=2, color='red')
        ax5.fill_between(epochs, self.history['hard_deleted'], alpha=0.3, color='red')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Count')
        ax5.set_title('Hard-Deleted Neurons (Phase 7)')
        ax5.grid(True, alpha=0.3)
        
        # Created
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(epochs, self.history['created'], linewidth=2, color='green')
        ax6.fill_between(epochs, self.history['created'], alpha=0.3, color='green')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Count')
        ax6.set_title('Created Neurons (Phase 8)')
        ax6.grid(True, alpha=0.3)
        
        # Net change
        ax7 = fig.add_subplot(gs[2, 0])
        colors = ['red' if x < 0 else 'green' for x in self.history['net_change']]
        ax7.bar(epochs, self.history['net_change'], color=colors, alpha=0.6)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Net Change')
        ax7.set_title('Net Neuron Change (Created - Deleted)')
        ax7.grid(True, alpha=0.3)
        
        # Utility mean
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(epochs, self.history['utility_mean'], linewidth=2, color='purple')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Mean Utility')
        ax8.set_title('Average Utility Evolution')
        ax8.grid(True, alpha=0.3)
        
        # Utility std
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(epochs, self.history['utility_std'], linewidth=2, color='brown')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Utility Std')
        ax9.set_title('Utility Variance (Differentiation)')
        ax9.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()


def main():
    """Main training."""
    print("\n" + "="*70)
    print("COMPLETE SSNC SYSTEM - FASHION-MNIST")
    print("="*70)
    
    # Load data
    print("\nLoading Fashion-MNIST...")
    train_loader, val_loader = load_fashion_mnist(CONFIG['batch_size'])
    print(f"✓ Loaded")
    
    # Create trainer
    trainer = CompleteSSNCTrainer(CONFIG)
    
    # Train
    history = trainer.train(train_loader, val_loader, CONFIG['epochs'])
    
    # Save results
    output_dir = Path('./results/complete_ssnc')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plot_path = output_dir / 'complete_results.png'
    trainer.plot_results(save_path=plot_path)
    
    # Save model
    model_path = output_dir / 'complete_model.pt'
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    
    # Save history
    history_path = output_dir / 'complete_history.json'
    json_history = {k: v for k, v in history.items() if k != 'plasticity_events'}
    json_history['plasticity_events'] = history['plasticity_events']
    json_history['config'] = CONFIG
    
    with open(history_path, 'w') as f:
        json.dump(json_history, f, indent=2)
    print(f"History saved: {history_path}")
    
    # Final summary
    final_stats = trainer.model.get_statistics()
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"\nStructural Evolution:")
    print(f"  Initial neurons: {sum(CONFIG['hidden_dims'])}")
    print(f"  Final active: {final_stats['active_neurons']}")
    print(f"  Soft-deleted: {final_stats['soft_deleted']}")
    print(f"  Hard-deleted: {final_stats['hard_deleted']}")
    print(f"  Created: {final_stats['created']}")
    print(f"  Net change: {final_stats['net_change']:+d}")
    print(f"\nFinal Utility: {final_stats['utility_mean']:.4f} ± {final_stats['utility_std']:.4f}")
    print('='*70 + '\n')


if __name__ == "__main__":
    main()
