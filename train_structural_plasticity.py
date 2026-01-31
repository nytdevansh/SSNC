"""
STRUCTURAL PLASTICITY INTEGRATED WITH MNIST TRAINING

Combines SSNC Phase 6 (soft deletion) with complete structural plasticity
(phases 7-9: hard deletion, creation, redistribution).

Tests two scenarios:
1. NORMAL: Reasonable architecture that adapts gradually
2. EXTREME: Massively over-parameterized that prunes aggressively
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
from typing import Dict, List, Tuple

from ssnc_mlp_phase6 import SSNC_MLP_Phase6, print_phase6_stats
from structural_plasticity import StructuralPlasticityManager, CreationEvent


class PlasticityTrainer:
    """Trains SSNC network with full structural plasticity."""
    
    def __init__(
        self,
        model: SSNC_MLP_Phase6,
        hidden_dims: List[int],
        learning_rate: float = 0.01,
        device: str = None,
        scenario_name: str = "normal"
    ):
        self.model = model
        self.hidden_dims = hidden_dims
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Use momentum optimizer for better convergence
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        self.criterion = nn.CrossEntropyLoss()
        self.scenario_name = scenario_name
        
        # Structural plasticity manager
        self.plasticity_manager = StructuralPlasticityManager(
            n_layers=len(hidden_dims),
            initial_neurons_per_layer=hidden_dims,
            deletion_utility_threshold=0.05,
            soft_deletion_cycles=5,
            hard_deletion_cycles=10,
            enable_creation=True,
            overload_utility_threshold=0.7,
            max_creations_per_cycle=3,
            max_deletions_per_cycle=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'neurons_per_layer': [],
            'hard_deletions_per_epoch': [],
            'creations_per_epoch': [],
            'total_neurons': [],
            'plasticity_events': []
        }
    
    def _get_all_utilities(self) -> Dict:
        """Collect utilities from all neurons in model."""
        utilities = {}
        
        # Only compute if we have SSNC layers
        if hasattr(self.model, 'ssnc_layers') and self.model.ssnc_layers:
            for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
                if hasattr(ssnc_layer, 'neurons'):
                    for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                        utilities[(layer_idx, neuron_idx)] = neuron.utility
        
        return utilities
    
    def _handle_structural_changes(self, epoch: int, batch_idx: int):
        """
        Handle neuron creation and deletion.
        This is called periodically during training.
        """
        utilities = self._get_all_utilities()
        
        if not utilities:
            return
        
        # Update plasticity manager
        actions = self.plasticity_manager.update_cycle(utilities, [])
        
        # Log significant events
        if actions['hard_deletions'] or actions['creations']:
            event = {
                'epoch': epoch,
                'batch': batch_idx,
                'hard_deletions': len(actions['hard_deletions']),
                'creations': len(actions['creations']),
                'neurons_before': self.history['total_neurons'][-1] if self.history['total_neurons'] else sum(self.hidden_dims),
                'neurons_after': sum(self.plasticity_manager.current_neurons_per_layer)
            }
            self.history['plasticity_events'].append(event)
            
            # Print significant changes
            if len(actions['hard_deletions']) > 0 or len(actions['creations']) > 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}:")
                if len(actions['hard_deletions']) > 0:
                    print(f"    Hard deleted: {len(actions['hard_deletions'])} neurons")
                if len(actions['creations']) > 0:
                    print(f"    Created: {len(actions['creations'])} neurons")
                print(f"    Network: {self.plasticity_manager.current_neurons_per_layer}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train one epoch with structural plasticity."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        hard_deletions_epoch = 0
        creations_epoch = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Standard training step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # SSNC observation (BEFORE weights updated - capture raw signal)
            self.model.observe_gradients()
            
            if batch_idx % 10 == 0:
                self.model.compute_all_utilities()
            
            # Local learning - very small to gently regularize (not override SGD)
            if self.model.enable_local_learning:
                self.model.apply_local_learning()
            
            # SGD update (LAST - dominant force, not overridden by local learning)
            self.optimizer.step()
            
            # Observer cycle (soft deletion)
            if self.model.enable_observer:
                report = self.model.observer_cycle()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            self.model.update_loss(loss.item())
            
            # Structural plasticity check (every 10 batches)
            if batch_idx % 10 == 0:
                prev_total = sum(self.plasticity_manager.current_neurons_per_layer)
                self._handle_structural_changes(epoch, batch_idx)
                new_total = sum(self.plasticity_manager.current_neurons_per_layer)
                creations_epoch += max(0, new_total - prev_total)
                hard_deletions_epoch += max(0, prev_total - new_total)
            
            # Progress
            if batch_idx % 100 == 0:
                status = f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                status += f'Loss: {loss.item():.4f} Acc: {100.0*correct/total:.2f}%'
                status += f' | Neurons: {self.plasticity_manager.current_neurons_per_layer}'
                print(status)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        # Record epoch statistics
        self.history['neurons_per_layer'].append(
            self.plasticity_manager.current_neurons_per_layer.copy()
        )
        self.history['total_neurons'].append(
            sum(self.plasticity_manager.current_neurons_per_layer)
        )
        self.history['hard_deletions_per_epoch'].append(hard_deletions_epoch)
        self.history['creations_per_epoch'].append(creations_epoch)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validation with plasticity active."""
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
        epochs: int = 20
    ):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"STRUCTURAL PLASTICITY TRAINING - {self.scenario_name.upper()}")
        print(f"{'='*70}")
        print(f"Network: {self.hidden_dims}")
        print(f"Total epochs: {epochs}")
        print(f"{'='*70}\n")
        
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 3
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Current Architecture: {self.plasticity_manager.current_neurons_per_layer}")
            print(f"  Total Neurons: {sum(self.plasticity_manager.current_neurons_per_layer)}")
            
            if len(self.history['plasticity_events']) > 0:
                print(f"  Plasticity Events This Epoch: {len([e for e in self.history['plasticity_events'] if e['epoch'] == epoch])}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        return self.history
    
    def plot_results(self, save_path: Path = None):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Total neurons over time
        axes[1, 0].plot(self.history['total_neurons'], marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Total Neurons')
        axes[1, 0].set_title('Network Size Over Time')
        axes[1, 0].grid(True)
        
        # Creations vs Deletions
        epochs = range(len(self.history['creations_per_epoch']))
        axes[1, 1].bar(epochs, self.history['creations_per_epoch'], alpha=0.6, label='Created')
        axes[1, 1].bar(epochs, [-x for x in self.history['hard_deletions_per_epoch']], alpha=0.6, label='Hard Deleted')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Number of Neurons')
        axes[1, 1].set_title('Creations vs Deletions Per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: Path):
        """Save training results and analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert lists to serializable format
            history = {k: v for k, v in self.history.items()}
            json.dump(history, f, indent=2)
        
        # Save plasticity analysis
        analysis_path = output_dir / 'plasticity_analysis.json'
        analysis = {
            'scenario': self.scenario_name,
            'initial_architecture': self.hidden_dims,
            'final_architecture': self.plasticity_manager.current_neurons_per_layer,
            'initial_total_neurons': sum(self.hidden_dims),
            'final_total_neurons': sum(self.plasticity_manager.current_neurons_per_layer),
            'total_created': self.plasticity_manager.total_created,
            'total_hard_deleted': self.plasticity_manager.total_hard_deleted,
            'plasticity_events': self.history['plasticity_events'],
            'best_val_accuracy': max(self.history['val_acc']) if self.history['val_acc'] else 0.0
        }
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        return analysis


def ensure_dataset_downloaded(dataset_name: str = "mnist", max_retries: int = 3) -> Path:
    """
    Ensure dataset is downloaded with retry logic.
    
    Args:
        dataset_name: "mnist" or "fashion"
        max_retries: Number of download attempts
    
    Returns:
        Path to dataset directory
    """
    data_path = Path('./data').absolute() / dataset_name.upper()
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Select dataset class
    if dataset_name.lower() == "fashion":
        dataset_class = datasets.FashionMNIST
        dataset_display_name = "Fashion-MNIST"
    else:
        dataset_class = datasets.MNIST
        dataset_display_name = "MNIST"
    
    # Check if already downloaded
    try:
        # Try loading without download
        test_dataset = dataset_class(
            root=str(data_path),
            train=True,
            download=False
        )
        print(f"✓ {dataset_display_name} already downloaded ({len(test_dataset)} samples)")
        return data_path
    except (FileNotFoundError, RuntimeError):
        print(f"→ {dataset_display_name} not found locally, downloading...")
    
    # Download with retries
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: Downloading {dataset_display_name}...")
            
            # Download training set
            train_dataset = dataset_class(
                root=str(data_path),
                train=True,
                download=True
            )
            
            # Download test set
            test_dataset = dataset_class(
                root=str(data_path),
                train=False,
                download=True
            )
            
            print(f"✓ {dataset_display_name} downloaded successfully!")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Test samples: {len(test_dataset)}")
            return data_path
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  ✗ Attempt {attempt} failed: {type(e).__name__}")
                print(f"    Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                print(f"\n❌ FAILED: Could not download {dataset_display_name} after {max_retries} attempts")
                print(f"   Last error: {e}")
                print(f"\n   To fix this, try one of:")
                print(f"   1. Check your internet connection")
                print(f"   2. Manually download from:")
                if dataset_name.lower() == "fashion":
                    print(f"      https://github.com/zalandoresearch/fashion-mnist")
                else:
                    print(f"      http://yann.lecun.com/exdb/mnist/")
                print(f"   3. Run with --skip-fashion flag to use MNIST only")
                raise RuntimeError(f"Could not download {dataset_display_name}") from e
    
    return data_path


def load_dataset(dataset_name: str = "mnist", batch_size: int = 64):
    """Load dataset (MNIST or Fashion-MNIST) with automatic download."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Ensure dataset is downloaded
    data_path = ensure_dataset_downloaded(dataset_name)
    
    # Select dataset class
    if dataset_name.lower() == "fashion":
        dataset_class = datasets.FashionMNIST
        dataset_display_name = "Fashion-MNIST"
    else:
        dataset_class = datasets.MNIST
        dataset_display_name = "MNIST"
    
    print(f"Loading {dataset_display_name} dataset...")
    
    train_dataset = dataset_class(
        root=str(data_path),
        train=True,
        transform=transform,
        download=False
    )
    
    val_dataset = dataset_class(
        root=str(data_path),
        train=False,
        transform=transform,
        download=False
    )
    
    print(f"✓ {dataset_display_name} loaded: {len(train_dataset)} train, {len(val_dataset)} test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def run_scenario(scenario_name: str, hidden_dims: List[int], epochs: int = 10, dataset: str = "mnist"):
    """Run a training scenario."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO: {scenario_name.upper()}")
    print(f"# Architecture: {hidden_dims}")
    print(f"# Dataset: {dataset.upper()}")
    print(f"{'#'*70}")
    
    # Create model
    model = SSNC_MLP_Phase6(
        input_dim=784,
        hidden_dims=hidden_dims,
        output_dim=10,
        activation='tanh',
        enable_local_learning=True,  # Enabled throughout training
        local_lr=0.0001,  # 10x smaller - acts as regularizer
        enable_observer=True,  # Enabled throughout training
        enable_soft_deletion=True,
        deletion_utility_threshold=0.05,
        marking_cycles_required=10,
        max_deletions_per_cycle=5
    )
    
    # Load data
    train_loader, val_loader = load_dataset(dataset_name=dataset, batch_size=64)
    
    # Create trainer with higher learning rate for better convergence
    trainer = PlasticityTrainer(
        model=model,
        hidden_dims=hidden_dims,
        learning_rate=0.1,  # Increased from 0.01 for Fashion-MNIST
        scenario_name=scenario_name
    )
    
    # Train
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save results
    output_dir = Path('./checkpoints/structural_plasticity') / dataset / scenario_name
    analysis = trainer.save_results(output_dir)
    
    # Plot
    plot_path = output_dir / 'training_plots.png'
    trainer.plot_results(save_path=plot_path)
    
    return analysis


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STRUCTURAL PLASTICITY INTEGRATION WITH FASHION-MNIST TRAINING")
    print("="*70)
    
    # Test 1: Fashion-MNIST - Normal scenario
    print("\n[1/2] FASHION-MNIST - NORMAL SCENARIO")
    fashion_normal = run_scenario(
        scenario_name="normal",
        hidden_dims=[128, 64],
        epochs=10,
        dataset="fashion"
    )
    
    # Test 2: Fashion-MNIST - Extreme scenario (reveals real plasticity!)
    print("\n[2/2] FASHION-MNIST - EXTREME SCENARIO")
    fashion_extreme = run_scenario(
        scenario_name="extreme",
        hidden_dims=[512, 256],
        epochs=10,
        dataset="fashion"
    )
    
    # Comparison
    print("\n" + "="*70)
    print("FASHION-MNIST STRUCTURAL PLASTICITY RESULTS")
    print("="*70)
    
    scenarios = {
        'Normal (128/64)': fashion_normal,
        'Extreme (512/256)': fashion_extreme
    }
    
    print(f"\n{'Scenario':<25} {'Init Neurons':<15} {'Final':<15} {'Created':<12} {'Deleted':<12} {'Val Acc':<10}")
    print("-" * 90)
    
    for name, analysis in scenarios.items():
        init = analysis['initial_total_neurons']
        final = analysis['final_total_neurons']
        created = analysis['total_created']
        deleted = analysis['total_hard_deleted']
        acc = analysis['best_val_accuracy']
        print(f"{name:<25} {init:<15} {final:<15} {created:<12} {deleted:<12} {acc:>8.2f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Compare scenarios
    print("\n1. ARCHITECTURE IMPACT (Normal vs Extreme):")
    normal_init = fashion_normal['initial_total_neurons']
    normal_final = fashion_normal['final_total_neurons']
    extreme_init = fashion_extreme['initial_total_neurons']
    extreme_final = fashion_extreme['final_total_neurons']
    
    print(f"   Normal:  {normal_init} → {normal_final} ({normal_final-normal_init:+d} neurons)")
    print(f"   Extreme: {extreme_init} → {extreme_final} ({extreme_final-extreme_init:+d} neurons)")
    
    normal_net = normal_final - normal_init
    extreme_net = extreme_final - extreme_init
    print(f"   → Extreme scenario pruned {abs(extreme_net)} vs Normal {abs(normal_net)}")
    
    print("\n2. PLASTICITY ACTIVITY (Total created + deleted):")
    for name, analysis in scenarios.items():
        activity = analysis['total_created'] + analysis['total_hard_deleted']
        created = analysis['total_created']
        deleted = analysis['total_hard_deleted']
        print(f"   {name:<25} Created: {created:>3d} | Deleted: {deleted:>3d} | Total: {activity:>3d}")
    
    print("\n3. DELETION-TO-CREATION RATIO (Higher = more aggressive pruning):")
    for name, analysis in scenarios.items():
        ratio = analysis['total_hard_deleted'] / max(1, analysis['total_created']) if analysis['total_created'] > 0 else 0
        print(f"   {name:<25} {ratio:>6.2f}x")
    
    print("\n4. PERFORMANCE RESILIENCE (Accuracy maintained during restructuring):")
    normal_acc = fashion_normal['best_val_accuracy']
    extreme_acc = fashion_extreme['best_val_accuracy']
    print(f"   Normal:  {normal_acc:>6.2f}%")
    print(f"   Extreme: {extreme_acc:>6.2f}%")
    print(f"   → Difference: {extreme_acc - normal_acc:+.2f}%")
    if abs(extreme_acc - normal_acc) < 2.0:
        print(f"   ✓ System maintains performance despite aggressive restructuring!")
    
    print("\n" + "="*70)
    print("✓ Full integration complete!")
    print("✓ Results saved to: checkpoints/structural_plasticity/fashion/{normal,extreme}/")
    print("="*70 + "\n")
