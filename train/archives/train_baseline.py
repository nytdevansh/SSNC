"""
SSNC PHASE 1: Training Script for Baseline MLP

Trains the instrumented MLP on MNIST to establish baseline performance.
Saves checkpoints and statistics for later phases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
from pathlib import Path
from baseline_mlp import InstrumentedMLP, print_model_stats


class BaselineTrainer:
    """Trains the baseline MLP and tracks all statistics."""
    
    def __init__(
        self,
        model: InstrumentedMLP,
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
            'layer_stats': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update model's loss tracking
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
        """Validate on test set."""
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
        epochs: int = 10,
        save_dir: str = "checkpoints"
    ):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING BASELINE MODEL")
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
            
            # Record layer statistics
            layer_stats = self.model.get_layer_stats()
            self.history['layer_stats'].append({
                'epoch': epoch,
                'stats': layer_stats
            })
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Print detailed stats every 5 epochs
            if epoch % 5 == 0:
                print_model_stats(self.model)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path / 'best_model.pt')
                print(f"  ⭐ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save final model and history
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, save_path / 'final_model.pt')
        
        # Save history as JSON
        with open(save_path / 'training_history.json', 'w') as f:
            # Convert layer_stats to JSON-serializable format
            history_copy = self.history.copy()
            history_copy['layer_stats'] = [
                {
                    'epoch': entry['epoch'],
                    'stats': self._convert_stats_to_json(entry['stats'])
                }
                for entry in self.history['layer_stats']
            ]
            json.dump(history_copy, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Checkpoints saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _convert_stats_to_json(self, stats: dict) -> dict:
        """Convert stats to JSON-serializable format."""
        def convert_value(val):
            """Recursively convert numpy types to Python native types."""
            if isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [convert_value(v) for v in val]
            elif hasattr(val, 'item'):  # numpy scalar
                return val.item()
            else:
                return val
        
        json_stats = {
            'per_layer': {},
            'per_neuron': {}
        }
        
        for layer_idx, layer_data in stats['per_layer'].items():
            json_stats['per_layer'][str(layer_idx)] = convert_value(layer_data)
        
        for layer_idx, neuron_data in stats['per_neuron'].items():
            json_stats['per_neuron'][str(layer_idx)] = {
                str(n): convert_value(data) for n, data in neuron_data.items()
            }
        
        return json_stats
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss', alpha=0.7)
        axes[0].plot(self.history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(self.history['train_acc'], label='Train Acc', alpha=0.7)
        axes[1].plot(self.history['val_acc'], label='Val Acc', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
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
    """Main training script for PHASE 1."""
    
    # Configuration
    CONFIG = {
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'output_dim': 10,
        'activation': 'tanh',
        'learning_rate': 0.01,
        'batch_size': 128,
        'epochs': 20
    }
    
    print("SSNC PHASE 1: Baseline Model Training")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create model
    model = InstrumentedMLP(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=CONFIG['output_dim'],
        activation=CONFIG['activation']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=CONFIG['batch_size'])
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}\n")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        learning_rate=CONFIG['learning_rate']
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=CONFIG['epochs'],
        save_dir='checkpoints/phase1'
    )
    
    # Plot results
    trainer.plot_training_curves(save_path='checkpoints/phase1/training_curves.png')
    
    # Final statistics
    print("\nFinal Model Statistics:")
    print_model_stats(model)
    
    print("\n✅ PHASE 1 COMPLETE!")
    print("Next steps:")
    print("  1. Verify model trained successfully (check val accuracy)")
    print("  2. Examine neuron statistics for anomalies")
    print("  3. Ready to move to PHASE 2 (add local neuron state)")


if __name__ == "__main__":
    main()
