"""
SSNC PHASE 4 & 5: Training with Local Learning and Observer

Trains SSNC-MLP with:
- Standard SGD (global optimization)
- Local learning (utility-modulated)
- Observer monitoring (decision logging)

This is the first "complete" SSNC training loop.
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

from ssnc_mlp_advanced import SSNC_MLP_Advanced, print_advanced_stats


class AdvancedSSNCTrainer:
    """Trainer for PHASE 4 & 5 with hybrid learning and observer."""
    
    def __init__(
        self,
        model: SSNC_MLP_Advanced,
        learning_rate: float = 0.01,
        device: str = None,
        use_hybrid_learning: bool = True
    ):
        """
        Args:
            use_hybrid_learning: If True, combine SGD + local learning
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.use_hybrid_learning = use_hybrid_learning
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'ssnc_stats': [],
            'observer_reports': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train one epoch with hybrid learning."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # === HYBRID LEARNING STEP ===
            
            # 1. Standard forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 2. Standard backward pass
            loss.backward()
            
            # 3. SSNC: Neurons observe gradients
            self.model.observe_gradients()
            
            # 4. SSNC: Compute utilities
            if batch_idx % 10 == 0:
                self.model.compute_all_utilities()
            
            # 5. PHASE 4: Apply local learning (if enabled)
            if self.use_hybrid_learning and self.model.enable_local_learning:
                # Note: Local learning happens in parallel with SGD
                # This is the "hybrid" part - both mechanisms work together
                self.model.apply_local_learning()
            
            # 6. Standard SGD update
            self.optimizer.step()
            
            # 7. PHASE 5: Observer cycle (periodic)
            if self.model.enable_observer:
                report = self.model.observer_cycle()
                if report:
                    self.history['observer_reports'].append(report)
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            self.model.update_loss(loss.item())
            
            # Print progress
            if batch_idx % 100 == 0:
                status = f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                status += f'Loss: {loss.item():.4f} '
                status += f'Acc: {100.0 * correct / total:.2f}%'
                
                if self.model.enable_local_learning:
                    learning_stats = self.model.get_learning_summary()
                    if learning_stats['layers']:
                        total_updates = sum(
                            l.get('total_updates', 0) for l in learning_stats['layers']
                        )
                        status += f' | Local updates: {total_updates}'
                
                if self.model.enable_observer:
                    observer_stats = self.model.get_observer_summary()
                    if observer_stats:
                        status += f' | Health: {observer_stats["health_score"]:.2f}'
                
                print(status)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        save_dir: str = "checkpoints/phase4_5"
    ):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING ADVANCED SSNC (PHASE 4 & 5)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Local learning: {self.model.enable_local_learning}")
        print(f"Observer: {self.model.enable_observer}")
        print(f"Hybrid learning: {self.use_hybrid_learning}")
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
            
            # Record SSNC statistics
            self.model.compute_all_utilities()
            ssnc_summary = self.model.get_ssnc_summary()
            
            epoch_stats = {
                'epoch': epoch,
                'ssnc': ssnc_summary
            }
            
            if self.model.enable_local_learning:
                epoch_stats['learning'] = self.model.get_learning_summary()
            
            if self.model.enable_observer:
                epoch_stats['observer'] = self.model.get_observer_summary()
            
            self.history['ssnc_stats'].append(epoch_stats)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Mean Utility: {ssnc_summary['global']['mean_utility']:.4f}")
            
            if self.model.enable_observer:
                obs_summary = epoch_stats['observer']
                print(f"  Health Score: {obs_summary['health_score']:.3f} "
                      f"({obs_summary['health_trend']['trend']})")
                print(f"  Observer Decisions: {obs_summary['decisions']['total_decisions']}")
            
            # Validation checks
            self._phase4_5_validation_checks(epoch, epoch_stats)
            
            # Print detailed stats every 5 epochs
            if epoch % 5 == 0:
                print_advanced_stats(self.model)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(save_path / 'best_model.pt', epoch, val_acc, val_loss)
                print(f"  ⭐ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        self._save_checkpoint(save_path / 'final_model.pt', epochs, val_acc, val_loss)
        
        # Save observer log if available
        if self.model.enable_observer:
            self.model.observer.export_log(save_path / 'observer_log.json')
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Checkpoints saved to: {save_path}")
        self._print_phase4_5_summary()
        print(f"{'='*60}\n")
        
        return self.history
    
    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'history': self.history,
            'config': {
                'local_learning': self.model.enable_local_learning,
                'observer': self.model.enable_observer,
                'hybrid_learning': self.use_hybrid_learning
            }
        }
        torch.save(checkpoint, path)
    
    def _phase4_5_validation_checks(self, epoch: int, stats: dict):
        """PHASE 4 & 5 specific validation."""
        warnings = []
        
        # Check 1: Local learning shouldn't destabilize
        if self.model.enable_local_learning:
            if len(self.history['train_loss']) > 3:
                recent_losses = self.history['train_loss'][-3:]
                if any(recent_losses[i] > recent_losses[i-1] * 1.5 for i in range(1, len(recent_losses))):
                    warnings.append("⚠️  Loss increased significantly - possible instability")
        
        # Check 2: Observer health shouldn't crash
        if self.model.enable_observer and 'observer' in stats:
            health = stats['observer']['health_score']
            if health < 0.5:
                warnings.append(f"⚠️  Structural health critical ({health:.2f})")
        
        # Check 3: Not too many neurons should have clipped updates
        if self.model.enable_local_learning and 'learning' in stats:
            for layer_stats in stats['learning']['layers']:
                if 'total_clipped' in layer_stats and layer_stats.get('total_updates', 0) > 0:
                    clip_rate = layer_stats['total_clipped'] / layer_stats['total_updates']
                    if clip_rate > 0.5:
                        warnings.append(
                            f"⚠️  Layer {layer_stats['layer_id']}: "
                            f"{clip_rate*100:.1f}% updates clipped"
                        )
        
        if warnings:
            print(f"\n  VALIDATION WARNINGS (Epoch {epoch}):")
            for warning in warnings:
                print(f"  {warning}")
    
    def _print_phase4_5_summary(self):
        """Print final summary."""
        print("\nPHASE 4 & 5 Training Summary:")
        print("-" * 60)
        
        final_stats = self.history['ssnc_stats'][-1]
        
        # Performance vs baseline
        final_acc = self.history['val_acc'][-1]
        if final_acc > 95:
            print(f"✓ Performance maintained ({final_acc:.1f}% val accuracy)")
        else:
            print(f"⚠️  Performance below baseline ({final_acc:.1f}% val accuracy)")
        
        # Local learning impact
        if self.model.enable_local_learning and 'learning' in final_stats:
            total_updates = sum(
                l.get('total_updates', 0)
                for l in final_stats['learning']['layers']
            )
            print(f"✓ Local learning updates applied: {total_updates}")
        
        # Observer effectiveness
        if self.model.enable_observer and 'observer' in final_stats:
            obs = final_stats['observer']
            print(f"✓ Observer health: {obs['health_score']:.2f} ({obs['health_trend']['trend']})")
            print(f"✓ Decisions logged: {obs['decisions']['total_decisions']}")
        
        # Stability
        if len(self.history['train_loss']) > 5:
            recent_losses = self.history['train_loss'][-5:]
            loss_std = np.std(recent_losses) / np.mean(recent_losses)
            if loss_std < 0.1:
                print("✓ Training was stable")
            else:
                print("⚠️  Training showed variability")
        
        print("\nReady for PHASE 6: Soft Deletion (if validation passes)")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot comprehensive training curves."""
        n_plots = 3 if self.model.enable_observer else 2
        fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 8))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1, 0].plot(self.history['train_acc'], label='Train', alpha=0.7)
        axes[1, 0].plot(self.history['val_acc'], label='Val', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Accuracy Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Utility evolution
        mean_utilities = [
            entry['ssnc']['global']['mean_utility']
            for entry in self.history['ssnc_stats']
        ]
        axes[0, 1].plot(mean_utilities, color='purple', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Utility')
        axes[0, 1].set_title('Neuron Utility Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Inactive neurons
        inactive = [
            entry['ssnc']['global']['inactive_neurons']
            for entry in self.history['ssnc_stats']
        ]
        axes[1, 1].plot(inactive, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Inactive Neurons')
        axes[1, 1].set_title('Inactive Neuron Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Observer health (if available)
        if self.model.enable_observer and n_plots >= 3:
            health_scores = [
                entry['observer']['health_score']
                for entry in self.history['ssnc_stats']
                if 'observer' in entry
            ]
            axes[0, 2].plot(health_scores, color='green', alpha=0.7)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Health Score')
            axes[0, 2].set_title('Structural Health')
            axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            axes[0, 2].grid(True, alpha=0.3)
            
            # Observer decisions
            decision_counts = []
            for entry in self.history['ssnc_stats']:
                if 'observer' in entry:
                    decision_counts.append(entry['observer']['decisions']['total_decisions'])
            
            if decision_counts:
                axes[1, 2].plot(np.diff([0] + decision_counts), color='orange', alpha=0.7)
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('New Decisions')
                axes[1, 2].set_title('Observer Decisions per Epoch')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()


def load_mnist(batch_size: int = 64) -> tuple:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def main():
    """Main training script for PHASE 4 & 5."""
    
    CONFIG = {
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'output_dim': 10,
        'activation': 'tanh',
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 20,
        # PHASE 4 & 5 specific
        'enable_local_learning': True,
        'local_lr': 0.0001,
        'enable_observer': True,
        'observer_readonly': True,
        'use_hybrid_learning': True
    }
    
    print("SSNC PHASE 4 & 5: Training with Local Learning and Observer")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create advanced model
    model = SSNC_MLP_Advanced(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=CONFIG['output_dim'],
        activation=CONFIG['activation'],
        enable_local_learning=CONFIG['enable_local_learning'],
        local_lr=CONFIG['local_lr'],
        enable_observer=CONFIG['enable_observer'],
        observer_readonly=CONFIG['observer_readonly']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"SSNC tracking: {sum(len(layer.neurons) for layer in model.ssnc_layers)} neurons")
    print(f"Local learning: {'ENABLED' if CONFIG['enable_local_learning'] else 'DISABLED'}")
    print(f"Observer: {'ENABLED (read-only)' if CONFIG['enable_observer'] else 'DISABLED'}")
    print()
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=CONFIG['batch_size'])
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}\n")
    
    # Create trainer
    trainer = AdvancedSSNCTrainer(
        model=model,
        learning_rate=CONFIG['learning_rate'],
        use_hybrid_learning=CONFIG['use_hybrid_learning']
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=CONFIG['epochs'],
        save_dir='checkpoints/phase4_5'
    )
    
    # Plot results
    trainer.plot_training_curves(save_path='checkpoints/phase4_5/training_curves.png')
    
    # Final statistics
    print("\nFinal Advanced SSNC Statistics:")
    print_advanced_stats(model)
    
    print("\n✅ PHASE 4 & 5 COMPLETE!")
    print("\nKey validation points:")
    print("  1. Does performance match PHASE 2? (should be ~96%)")
    print("  2. Did local learning help or hurt?")
    print("  3. Are observer decisions sensible?")
    print("  4. Is structural health stable?")
    print("\nIf all checks pass → Ready for PHASE 6 (Soft Deletion)")


if __name__ == "__main__":
    main()
