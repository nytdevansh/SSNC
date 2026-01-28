"""
SSNC PHASE 2: Training with Neuron Observation

Trains SSNC-MLP where neurons observe but don't act.
This validates that observation doesn't break training.

CRITICAL VALIDATION:
- Performance should match baseline (PHASE 1)
- Utilities should differentiate between neurons
- No instability should appear
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
from ssnc_mlp import SSNC_MLP, print_ssnc_stats


class SSNC_Trainer:
    """Trains SSNC-MLP with full neuron observation."""
    
    def __init__(
        self,
        model: SSNC_MLP,
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
            'ssnc_stats': []  # SSNC-specific tracking
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch with SSNC observation."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass (neurons observe activations automatically)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # SSNC: Let neurons observe error signals
            self.model.observe_gradients()
            
            # SSNC: Compute utilities periodically (not every step - expensive)
            if batch_idx % 10 == 0:
                self.model.compute_all_utilities()
            
            # Standard SGD update (neurons don't interfere yet)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            self.model.update_loss(loss.item())
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.0 * correct / total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate with SSNC observation."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass (neurons still observe in eval mode)
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
        epochs: int = 10,
        save_dir: str = "checkpoints/phase2"
    ):
        """Full training loop with SSNC validation checks."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING SSNC MODEL (PHASE 2)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
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
            self.model.compute_all_utilities()  # Final update for epoch
            ssnc_summary = self.model.get_ssnc_summary()
            self.history['ssnc_stats'].append({
                'epoch': epoch,
                'summary': ssnc_summary
            })
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Mean Neuron Utility: {ssnc_summary['global']['mean_utility']:.4f}")
            print(f"  Inactive Neurons: {ssnc_summary['global']['inactive_neurons']}")
            
            # PHASE 2 VALIDATION CHECKS
            self._phase2_validation_checks(epoch, ssnc_summary)
            
            # Print detailed stats every 5 epochs
            if epoch % 5 == 0:
                print_ssnc_stats(self.model)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'ssnc_summary': ssnc_summary
                }, save_path / 'best_model.pt')
                print(f"  ⭐ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, save_path / 'final_model.pt')
        
        # Save detailed neuron states
        neuron_states = self.model.get_detailed_neuron_states()
        with open(save_path / 'final_neuron_states.json', 'w') as f:
            json.dump(neuron_states, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"PHASE 2 TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Checkpoints saved to: {save_path}")
        self._print_phase2_summary()
        print(f"{'='*60}\n")
        
        return self.history
    
    def _phase2_validation_checks(self, epoch: int, ssnc_summary: dict):
        """
        PHASE 2 specific validation checks.
        
        These ensure observation doesn't break training.
        """
        warnings = []
        
        # Check 1: Utilities should differ across neurons
        for layer_data in ssnc_summary['layers']:
            utility_std = layer_data['utilities']['std']
            if utility_std < 0.01:
                warnings.append(
                    f"⚠️  Layer {layer_data['layer_id']}: Low utility variance ({utility_std:.4f})"
                )
        
        # Check 2: Not too many inactive neurons
        inactive_pct = (
            100.0 * ssnc_summary['global']['inactive_neurons'] /
            ssnc_summary['global']['total_neurons']
        )
        if inactive_pct > 30:
            warnings.append(
                f"⚠️  {inactive_pct:.1f}% neurons inactive (threshold: 30%)"
            )
        
        # Check 3: Utilities should not all be near zero
        if ssnc_summary['global']['mean_utility'] < 0.05:
            warnings.append(
                f"⚠️  Mean utility very low ({ssnc_summary['global']['mean_utility']:.4f})"
            )
        
        # Print warnings if any
        if warnings:
            print(f"\n  VALIDATION WARNINGS (Epoch {epoch}):")
            for warning in warnings:
                print(f"  {warning}")
    
    def _print_phase2_summary(self):
        """Print final PHASE 2 validation summary."""
        final_stats = self.history['ssnc_stats'][-1]['summary']
        
        print("\nPHASE 2 Validation Summary:")
        print("-" * 60)
        
        # Check if utilities differentiate
        utility_ranges = []
        for layer_data in final_stats['layers']:
            utility_range = (
                layer_data['utilities']['max'] - 
                layer_data['utilities']['min']
            )
            utility_ranges.append(utility_range)
        
        avg_range = np.mean(utility_ranges)
        if avg_range > 0.1:
            print("✓ Utilities differentiate between neurons")
        else:
            print("✗ Utilities don't differentiate well (range < 0.1)")
        
        # Check stability
        if len(self.history['train_loss']) > 5:
            recent_losses = self.history['train_loss'][-5:]
            loss_variation = np.std(recent_losses) / np.mean(recent_losses)
            if loss_variation < 0.1:
                print("✓ Training is stable")
            else:
                print("⚠️  Training shows some instability")
        
        # Check performance vs expected
        final_acc = self.history['val_acc'][-1]
        if final_acc > 95:
            print(f"✓ Good performance ({final_acc:.1f}% val accuracy)")
        elif final_acc > 90:
            print(f"⚠️  Acceptable performance ({final_acc:.1f}% val accuracy)")
        else:
            print(f"✗ Poor performance ({final_acc:.1f}% val accuracy)")
        
        print("\nReady for PHASE 3: Utility computation refinement")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves with SSNC metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(self.history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train', alpha=0.7)
        axes[0, 1].plot(self.history['val_acc'], label='Val', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean utility over time
        mean_utilities = [
            entry['summary']['global']['mean_utility']
            for entry in self.history['ssnc_stats']
        ]
        axes[1, 0].plot(mean_utilities, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mean Utility')
        axes[1, 0].set_title('Neuron Utility Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inactive neurons over time
        inactive_neurons = [
            entry['summary']['global']['inactive_neurons']
            for entry in self.history['ssnc_stats']
        ]
        axes[1, 1].plot(inactive_neurons, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Inactive Neurons')
        axes[1, 1].set_title('Inactive Neuron Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        
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
    """Main training script for PHASE 2."""
    
    CONFIG = {
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'output_dim': 10,
        'activation': 'tanh',
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 20
    }
    
    print("SSNC PHASE 2: Training with Neuron Observation")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create SSNC model
    model = SSNC_MLP(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=CONFIG['output_dim'],
        activation=CONFIG['activation']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"SSNC tracking: {sum(len(layer.neurons) for layer in model.ssnc_layers)} neurons\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=CONFIG['batch_size'])
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}\n")
    
    # Create trainer
    trainer = SSNC_Trainer(
        model=model,
        learning_rate=CONFIG['learning_rate']
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=CONFIG['epochs'],
        save_dir='checkpoints/phase2'
    )
    
    # Plot results
    trainer.plot_training_curves(save_path='checkpoints/phase2/training_curves.png')
    
    # Final SSNC statistics
    print("\nFinal SSNC Statistics:")
    print_ssnc_stats(model)
    
    print("\n✅ PHASE 2 COMPLETE!")
    print("\nKey validation points:")
    print("  1. Does val accuracy match PHASE 1 baseline? (should be ~97%)")
    print("  2. Do utilities differentiate between neurons? (check std > 0.1)")
    print("  3. Are statistics stable? (no NaNs, no explosions)")
    print("  4. Do neurons track correctly? (check neuron states)")
    print("\nIf all checks pass → Ready for PHASE 3")


if __name__ == "__main__":
    main()
