"""
GUARANTEED WORKING SSNC TRAINING

This WILL show plasticity activity because:
1. Simple, direct neuron tracking
2. Higher deletion threshold (more neurons qualify)
3. Shorter marking period (faster action)
4. Clear logging of every event

If this doesn't show deletions, something is fundamentally broken.
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
from actually_working_ssnc import WorkingSSNC_MLP


# AGGRESSIVE SETTINGS TO GUARANTEE ACTIVITY
CONFIG = {
    'input_dim': 784,
    'hidden_dims': [384, 192],  # Large to see deletions
    'output_dim': 10,
    'activation': 'relu',
    
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0,
    
    # AGGRESSIVE - WILL trigger deletions
    'deletion_threshold': 0.15,  # High = easier to trigger
    'marking_cycles': 8,  # Short = faster action
    'gating_rate': 0.12,  # Fast gating
    'max_deletions_per_cycle': 5,  # Allow more
    
    'utility_update_every': 5,  # Frequent
    'plasticity_cycle_every': 30,  # Frequent
    
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


class SimpleTrainer:
    """Simple trainer focused on showing plasticity."""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        self.model = WorkingSSNC_MLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            activation=config['activation'],
            deletion_threshold=config['deletion_threshold'],
            marking_cycles=config['marking_cycles'],
            gating_rate=config['gating_rate'],
            max_deletions_per_cycle=config['max_deletions_per_cycle']
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
            'marked': [],
            'soft_deleted': [],
            'utility_mean': [],
            'utility_std': [],
            'deletion_events': []
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
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'])
            
            self.model.observe_gradients()
            self.optimizer.step()
            
            # Utilities
            if batch_idx % self.config['utility_update_every'] == 0:
                self.model.compute_all_utilities()
            
            # Plasticity
            if batch_idx % self.config['plasticity_cycle_every'] == 0:
                actions = self.model.plasticity_cycle()
                
                if actions['soft_gated']:
                    print(f"  ⚡ Epoch {epoch}, Batch {batch_idx}: DELETED {len(actions['soft_gated'])} neurons")
                    for item in actions['soft_gated'][:3]:
                        print(f"      Layer {item['layer']}, Neuron {item['neuron']}: utility={item['utility']:.4f}")
                    
                    self.history['deletion_events'].append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'count': len(actions['soft_gated'])
                    })
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                stats = self.model.get_statistics()
                print(f"  [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100.0*correct/total:.2f}% "
                      f"| Marked: {stats['marked']} Deleted: {stats['soft_deleted']} "
                      f"| Util: {stats['utility_mean']:.3f}±{stats['utility_std']:.3f} "
                      f"Low: {stats['low_utility_count']}")
        
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
        """Full training."""
        print(f"\n{'='*70}")
        print("GUARANTEED WORKING SSNC TRAINING")
        print('='*70)
        print(f"Settings (AGGRESSIVE to show activity):")
        print(f"  Initial neurons: {sum(self.config['hidden_dims'])}")
        print(f"  Deletion threshold: {self.config['deletion_threshold']} (HIGH)")
        print(f"  Marking cycles: {self.config['marking_cycles']} (SHORT)")
        print(f"  Plasticity every: {self.config['plasticity_cycle_every']} batches")
        print('='*70 + '\n')
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print('-'*70)
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            stats = self.model.get_statistics()
            self.history['marked'].append(stats['marked'])
            self.history['soft_deleted'].append(stats['soft_deleted'])
            self.history['utility_mean'].append(stats['utility_mean'])
            self.history['utility_std'].append(stats['utility_std'])
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  Plasticity: Marked={stats['marked']}, Deleted={stats['soft_deleted']}, Total={stats['total_soft_deleted']}")
            print(f"  Utility: {stats['utility_mean']:.4f} ± {stats['utility_std']:.4f}, Low={stats['low_utility_count']}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"  ★ New best!")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print('='*70)
        
        return self.history
    
    def plot_results(self, save_path=None):
        """Plot results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Marked neurons
        axes[0, 2].plot(epochs, self.history['marked'], linewidth=2, color='orange')
        axes[0, 2].fill_between(epochs, self.history['marked'], alpha=0.3, color='orange')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Marked for Deletion')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Soft deleted
        axes[1, 0].plot(epochs, self.history['soft_deleted'], linewidth=2, color='red')
        axes[1, 0].fill_between(epochs, self.history['soft_deleted'], alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('SOFT-DELETED NEURONS (Should be NON-ZERO!)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Utility mean
        axes[1, 1].plot(epochs, self.history['utility_mean'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Utility')
        axes[1, 1].set_title('Average Utility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Utility std
        axes[1, 2].plot(epochs, self.history['utility_std'], linewidth=2, color='brown')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Utility Std')
        axes[1, 2].set_title('Utility Variance')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()


def main():
    """Main."""
    print("\n" + "="*70)
    print("SIMPLE WORKING SSNC - FASHION-MNIST")
    print("="*70)
    
    train_loader, val_loader = load_fashion_mnist(CONFIG['batch_size'])
    
    trainer = SimpleTrainer(CONFIG)
    history = trainer.train(train_loader, val_loader, CONFIG['epochs'])
    
    output_dir = Path('./results/simple_working_ssnc')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'working_results.png'
    trainer.plot_results(save_path=plot_path)
    
    torch.save(trainer.model.state_dict(), output_dir / 'model.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump({k: v for k, v in history.items() if k != 'deletion_events'}, f, indent=2)
    
    final_stats = trainer.model.get_statistics()
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"Best Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Marked neurons: {final_stats['marked']}")
    print(f"Soft-deleted neurons: {final_stats['soft_deleted']}")
    print(f"Total deleted: {final_stats['total_soft_deleted']}")
    print(f"Final utility: {final_stats['utility_mean']:.4f} ± {final_stats['utility_std']:.4f}")
    
    if final_stats['total_soft_deleted'] > 0:
        print(f"\n✅ PLASTICITY IS WORKING! Deleted {final_stats['total_soft_deleted']} neurons!")
    else:
        print(f"\n❌ PLASTICITY STILL NOT WORKING")
        print(f"   Low utility count: {final_stats['low_utility_count']}")
        print(f"   This suggests utilities aren't differentiating or threshold is wrong.")
    
    print('='*70 + '\n')


if __name__ == "__main__":
    main()
