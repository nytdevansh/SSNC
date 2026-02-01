"""
SSNC PHASE 6: Training with Soft Deletion

First time Observer has execution authority!
Trains with gradual soft deletion of low-utility neurons.

CRITICAL VALIDATION:
- Performance shouldn't drop significantly
- Deletions should be gradual and reversible
- Recovery mechanism should work
- System should remain stable
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

from phases.ssnc_mlp_phase6 import SSNC_MLP_Phase6, print_phase6_stats


class Phase6Trainer:
    """Trainer for PHASE 6 with soft deletion."""
    
    def __init__(
        self,
        model: SSNC_MLP_Phase6,
        learning_rate: float = 0.01,
        device: str = None
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'ssnc_stats': [],
            'deletion_stats': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train one epoch with soft deletion."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Standard training step
            self.optimizer.zero_grad()
            output = self.model(data)  # Includes soft deletion gating
            loss = self.criterion(output, target)
            loss.backward()
            
            # SSNC observation
            self.model.observe_gradients()
            
            if batch_idx % 10 == 0:
                self.model.compute_all_utilities()
            
            # Local learning
            if self.model.enable_local_learning:
                self.model.apply_local_learning()
            
            # SGD update
            self.optimizer.step()
            
            # Observer cycle (includes soft deletion execution)
            if self.model.enable_observer:
                report = self.model.observer_cycle()
                
                # Log deletion actions
                if report and 'deletion_actions' in report:
                    actions = report['deletion_actions']
                    if actions['newly_gated'] or actions['recovered']:
                        self.history['deletion_stats'].append({
                            'epoch': epoch,
                            'batch': batch_idx,
                            'actions': actions
                        })
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            self.model.update_loss(loss.item())
            
            # Progress
            if batch_idx % 100 == 0:
                status = f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                status += f'Loss: {loss.item():.4f} Acc: {100.0*correct/total:.2f}%'
                
                if self.model.enable_soft_deletion:
                    del_summary = self.model.get_deletion_summary()
                    if del_summary:
                        stats = del_summary['statistics']
                        status += f' | Gated: {stats["soft_deleted"]}'
                        status += f' | Marked: {stats["marked_for_deletion"]}'
                
                if self.model.enable_observer:
                    obs = self.model.get_observer_summary()
                    if obs:
                        status += f' | Health: {obs["health_score"]:.2f}'
                
                print(status)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validation with soft deletion active."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass includes gating
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        save_dir: str = "checkpoints/phase6"
    ):
        """Full training loop with soft deletion."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING PHASE 6 (WITH SOFT DELETION)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Soft deletion: {self.model.enable_soft_deletion}")
        if self.model.enable_soft_deletion:
            dm = self.model.deletion_manager
            print(f"  Deletion threshold: {dm.deletion_utility_threshold}")
            print(f"  Marking cycles: {dm.marking_cycles_required}")
            print(f"  Max deletions/cycle: {dm.max_deletions_per_cycle}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # SSNC stats
            self.model.compute_all_utilities()
            ssnc_summary = self.model.get_ssnc_summary()
            
            epoch_stats = {
                'epoch': epoch,
                'ssnc': ssnc_summary
            }
            
            if self.model.enable_soft_deletion:
                epoch_stats['deletion'] = self.model.get_deletion_summary()
            
            if self.model.enable_observer:
                epoch_stats['observer'] = self.model.get_observer_summary()
            
            self.history['ssnc_stats'].append(epoch_stats)
            
            # Summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train: {train_loss:.4f} / {train_acc:.2f}%")
            print(f"  Val: {val_loss:.4f} / {val_acc:.2f}%")
            
            if self.model.enable_soft_deletion and 'deletion' in epoch_stats:
                del_stats = epoch_stats['deletion']['statistics']
                print(f"  Soft-deleted: {del_stats['soft_deleted']} "
                      f"(total: {del_stats['total_soft_deleted']}, "
                      f"recovered: {del_stats['total_recovered']})")
            
            # Validation checks
            self._phase6_validation_checks(epoch, epoch_stats)
            
            # Detailed stats every 5 epochs
            if epoch % 5 == 0:
                print_phase6_stats(self.model)
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(save_path / 'best_model.pt', epoch, val_acc, val_loss)
                print(f"  ⭐ New best! (Val Acc: {val_acc:.2f}%)")
        
        # Final save
        self._save_checkpoint(save_path / 'final_model.pt', epochs, val_acc, val_loss)
        
        # Save deletion log
        if self.model.enable_soft_deletion:
            with open(save_path / 'deletion_log.json', 'w') as f:
                json.dump(self.history['deletion_stats'], f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"PHASE 6 TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best val accuracy: {best_val_acc:.2f}%")
        print(f"Checkpoints: {save_path}")
        self._print_phase6_summary()
        print(f"{'='*60}\n")
        
        return self.history
    
    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float, val_loss: float):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'history': self.history,
            'deletion_manager_state': (
                self.model.deletion_manager.__dict__
                if self.model.deletion_manager else None
            )
        }, path)
    
    def _phase6_validation_checks(self, epoch: int, stats: dict):
        """PHASE 6 specific validation."""
        warnings = []
        
        # Check 1: Not too many deletions too fast
        if 'deletion' in stats:
            del_stats = stats['deletion']['statistics']
            total_deleted = del_stats['soft_deleted']
            total_neurons = self.model.get_ssnc_summary()['global']['total_neurons']
            
            deletion_rate = total_deleted / total_neurons
            if deletion_rate > 0.3:
                warnings.append(
                    f"⚠️  High deletion rate: {deletion_rate:.1%} of neurons gated"
                )
        
        # Check 2: Performance shouldn't collapse
        if len(self.history['val_acc']) > 1:
            curr_acc = self.history['val_acc'][-1]
            prev_acc = self.history['val_acc'][-2]
            
            if curr_acc < prev_acc - 2.0:
                warnings.append(
                    f"⚠️  Accuracy dropped {prev_acc - curr_acc:.1f}% in one epoch"
                )
        
        # Check 3: Recovery should work
        if 'deletion' in stats:
            del_stats = stats['deletion']['statistics']
            if del_stats['total_soft_deleted'] > 10:
                recovery_rate = del_stats['recovery_rate']
                if recovery_rate > 0:
                    print(f"  ✓ Recovery working: {recovery_rate:.1%} recovery rate")
        
        if warnings:
            print(f"\n  VALIDATION WARNINGS (Epoch {epoch}):")
            for warning in warnings:
                print(f"  {warning}")
    
    def _print_phase6_summary(self):
        """Print PHASE 6 summary."""
        print("\nPHASE 6 Training Summary:")
        print("-" * 60)
        
        final_stats = self.history['ssnc_stats'][-1]
        
        # Performance
        final_acc = self.history['val_acc'][-1]
        print(f"✓ Final accuracy: {final_acc:.2f}%")
        
        # Deletion impact
        if 'deletion' in final_stats:
            del_stats = final_stats['deletion']['statistics']
            print(f"✓ Neurons soft-deleted: {del_stats['soft_deleted']}")
            print(f"✓ Total deletions attempted: {del_stats['total_soft_deleted']}")
            print(f"✓ Successful recoveries: {del_stats['total_recovered']}")
            print(f"✓ Recovery rate: {del_stats['recovery_rate']:.1%}")
            
            # Effective network size
            total_neurons = self.model.get_ssnc_summary()['global']['total_neurons']
            active_neurons = total_neurons - del_stats['soft_deleted']
            print(f"✓ Effective network size: {active_neurons}/{total_neurons} neurons")
        
        # Stability
        if len(self.history['train_loss']) > 5:
            recent_losses = self.history['train_loss'][-5:]
            loss_std = np.std(recent_losses) / np.mean(recent_losses)
            if loss_std < 0.1:
                print("✓ Training remained stable")
        
        print("\nReady for PHASE 7: Hard Deletion")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves with deletion metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', alpha=0.7)
        axes[0, 1].plot(self.history['val_acc'], label='Val', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Utility
        utilities = [
            s['ssnc']['global']['mean_utility']
            for s in self.history['ssnc_stats']
        ]
        axes[0, 2].plot(utilities, color='purple', alpha=0.7)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Mean Utility')
        axes[0, 2].set_title('Utility Evolution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Soft-deleted neurons
        if 'deletion' in self.history['ssnc_stats'][0]:
            deleted = [
                s['deletion']['statistics']['soft_deleted']
                for s in self.history['ssnc_stats']
            ]
            axes[1, 0].plot(deleted, color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Soft-Deleted Neurons')
            axes[1, 0].set_title('Active Soft Deletions')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Recovery rate
            recovery_rates = [
                s['deletion']['statistics']['recovery_rate']
                for s in self.history['ssnc_stats']
            ]
            axes[1, 1].plot(recovery_rates, color='green', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recovery Rate')
            axes[1, 1].set_title('Neuron Recovery Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        # Health score
        if 'observer' in self.history['ssnc_stats'][0]:
            health = [
                s['observer']['health_score']
                for s in self.history['ssnc_stats']
            ]
            axes[1, 2].plot(health, color='blue', alpha=0.7)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Health Score')
            axes[1, 2].set_title('Structural Health')
            axes[1, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved: {save_path}")
        
        plt.show()


def load_mnist(batch_size: int = 64) -> tuple:
    """Load MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Main training for PHASE 6."""
    
    CONFIG = {
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'output_dim': 10,
        'activation': 'tanh',
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 20,
        'enable_local_learning': True,
        'local_lr': 0.0001,
        'enable_observer': True,
        'enable_soft_deletion': True,
        'deletion_utility_threshold': 0.05,
        'marking_cycles_required': 10,
        'max_deletions_per_cycle': 5
    }
    
    print("SSNC PHASE 6: Training with Soft Deletion")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create model
    model = SSNC_MLP_Phase6(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=CONFIG['output_dim'],
        activation=CONFIG['activation'],
        enable_local_learning=CONFIG['enable_local_learning'],
        local_lr=CONFIG['local_lr'],
        enable_observer=CONFIG['enable_observer'],
        enable_soft_deletion=CONFIG['enable_soft_deletion'],
        deletion_utility_threshold=CONFIG['deletion_utility_threshold'],
        marking_cycles_required=CONFIG['marking_cycles_required'],
        max_deletions_per_cycle=CONFIG['max_deletions_per_cycle']
    )
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"SSNC neurons: {sum(len(l.neurons) for l in model.ssnc_layers)}\n")
    
    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = load_mnist(batch_size=CONFIG['batch_size'])
    print(f"✓ Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}\n")
    
    # Train
    trainer = Phase6Trainer(model=model, learning_rate=CONFIG['learning_rate'])
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=CONFIG['epochs'],
        save_dir='checkpoints/phase6'
    )
    
    # Plot
    trainer.plot_training_curves(save_path='checkpoints/phase6/training_curves.png')
    
    # Final stats
    print("\nFinal Phase 6 Statistics:")
    print_phase6_stats(model)
    
    print("\n✅ PHASE 6 COMPLETE!")
    print("\nValidation points:")
    print("  1. Did soft deletion work? (neurons gated)")
    print("  2. Did recovery work? (some neurons ungated)")
    print("  3. Performance maintained? (should be ~95-96%)")
    print("  4. System stable? (no crashes)")
    print("\nIf all pass → Ready for PHASE 7 (Hard Deletion)")


if __name__ == "__main__":
    main()
