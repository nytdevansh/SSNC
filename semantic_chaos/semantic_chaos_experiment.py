"""
Semantic Chaos Experiment: Rule-Switching Fashion-MNIST

Tests whether SSNC can maintain structural memory when task semantics change.

Phase 1 (Epochs 1-10):  Normal training
Phase 2 (Epochs 11-20): Label swap (T-shirt↔Trouser, Pullover↔Shirt, Coat↔Bag)
Phase 3 (Epochs 21-30): Restore original labels

Expected: SSNC recovers 2x faster in Phase 3 than base model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.complete_ssnc import CompleteSSNC_MLP


class LabelSwappingDataset(Dataset):
    """Wrapper to dynamically swap labels."""
    
    def __init__(self, base_dataset, swap_pairs: List[Tuple[int, int]] = None):
        self.base_dataset = base_dataset
        self.swap_pairs = swap_pairs or []
        
        # Create swap mapping
        self.label_map = {i: i for i in range(10)}
        for a, b in self.swap_pairs:
            self.label_map[a] = b
            self.label_map[b] = a
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        new_label = self.label_map[label]
        return img, new_label


class BaselineMLP(nn.Module):
    """Standard MLP for comparison."""
    
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SemanticChaosExperiment:
    """Three-phase experiment with rule switching."""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Experiment phases
        self.phases = {
            'phase1': {'epochs': (1, 10), 'swaps': []},
            'phase2': {'epochs': (11, 20), 'swaps': [(0, 1), (2, 6), (4, 8)]},
            'phase3': {'epochs': (21, 30), 'swaps': []}  # Restore original
        }
        
        self.results = {
            'ssnc': {'phase1': [], 'phase2': [], 'phase3': []},
            'baseline': {'phase1': [], 'phase2': [], 'phase3': []}
        }
    
    def prepare_data(self, swap_pairs: List[Tuple[int, int]]):
        """Prepare dataloaders with label swapping."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_base = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        val_base = datasets.FashionMNIST('./data', train=False, transform=transform)
        
        train_dataset = LabelSwappingDataset(train_base, swap_pairs)
        val_dataset = LabelSwappingDataset(val_base, swap_pairs)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, is_ssnc=False, epoch=1):
        """Train for one epoch."""
        model.train()
        
        # Set current epoch for burn-in tracking
        if is_ssnc:
            model.set_epoch(epoch)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # SSNC plasticity
            if is_ssnc and batch_idx % 10 == 0:
                model.observe_gradients()
                model.plasticity_cycle(current_loss=loss.item())
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, model, val_loader):
        """Evaluate model."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.view(data.size(0), -1).to(self.device)
                target = target.to(self.device)
                
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100. * correct / total
    
    def run_phase(self, model, optimizer, criterion, phase_name, is_ssnc=False):
        """Run a single phase of training."""
        phase_config = self.phases[phase_name]
        start_epoch, end_epoch = phase_config['epochs']
        swap_pairs = phase_config['swaps']
        
        print(f"\n{'='*70}")
        print(f"{phase_name.upper()}: Epochs {start_epoch}-{end_epoch}")
        if swap_pairs:
            print(f"Label swaps: {swap_pairs}")
        else:
            print("Original labels")
        print(f"{'='*70}")
        
        # Prepare data with current swap configuration
        train_loader, val_loader = self.prepare_data(swap_pairs)
        
        phase_results = []
        
        for epoch in range(start_epoch, end_epoch + 1):
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, is_ssnc, epoch=epoch
            )
            val_acc = self.evaluate(model, val_loader)
            
            phase_results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            print(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f} "
                  f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        return phase_results
    
    def run_experiment(self, model_type='ssnc'):
        """Run full 3-phase experiment."""
        print(f"\n{'#'*70}")
        print(f"RUNNING {model_type.upper()} MODEL")
        print(f"{'#'*70}")
        
        # Create model
        if model_type == 'ssnc':
            model = CompleteSSNC_MLP(
                input_dim=784,
                hidden_dims=[256, 128],
                output_dim=10,
                activation='relu',
                deletion_threshold=0.03,
                marking_cycles=30,
                hard_deletion_cycles=60,
                grace_period=50,
                enable_hard_deletion=True,
                enable_creation=True,
                enable_monitoring=True,
                max_deletions_per_cycle=1,
                max_creations_per_cycle=2,
                burn_in_epochs=8  # 🔥 CRITICAL: No SSNC actions for first 8 epochs
            ).to(self.device)
            is_ssnc = True
        else:
            model = BaselineMLP(
                input_dim=784,
                hidden_dims=[256, 128],
                output_dim=10
            ).to(self.device)
            is_ssnc = False
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Run all phases
        for phase_name in ['phase1', 'phase2', 'phase3']:
            results = self.run_phase(model, optimizer, criterion, phase_name, is_ssnc)
            self.results[model_type][phase_name] = results
        
        return model
    
    def analyze_results(self):
        """Analyze and compare results."""
        print(f"\n{'='*70}")
        print("ANALYSIS: Semantic Memory Test")
        print(f"{'='*70}")
        
        # Phase 3 analysis (key test)
        print("\nPhase 3 Recovery Analysis:")
        print("-" * 70)
        
        for model_type in ['baseline', 'ssnc']:
            phase3 = self.results[model_type]['phase3']
            
            initial_acc = phase3[0]['val_acc']  # First epoch of phase 3
            final_acc = phase3[-1]['val_acc']   # Last epoch of phase 3
            
            # Find recovery speed (epochs to reach 85%)
            recovery_epochs = None
            for i, result in enumerate(phase3):
                if result['val_acc'] >= 85.0:
                    recovery_epochs = i + 1
                    break
            
            print(f"\n{model_type.upper()}:")
            print(f"  Initial accuracy (epoch 21): {initial_acc:.2f}%")
            print(f"  Final accuracy (epoch 30):   {final_acc:.2f}%")
            print(f"  Recovery gain:                {final_acc - initial_acc:.2f}%")
            print(f"  Epochs to 85%:                {recovery_epochs if recovery_epochs else '>10'}")
        
        # Determine winner
        print("\n" + "="*70)
        baseline_initial = self.results['baseline']['phase3'][0]['val_acc']
        ssnc_initial = self.results['ssnc']['phase3'][0]['val_acc']
        
        baseline_final = self.results['baseline']['phase3'][-1]['val_acc']
        ssnc_final = self.results['ssnc']['phase3'][-1]['val_acc']
        
        if ssnc_initial > baseline_initial + 10:
            print("✓ WIN: SSNC shows structural memory (>10% initial advantage)")
        elif ssnc_final > baseline_final + 1:
            print("✓ WIN: SSNC recovers faster (>1% final advantage)")
        else:
            print("⚠ INCONCLUSIVE: Differences within noise margin")
        print("="*70)
    
    def save_results(self, filepath='semantic_chaos_results.json'):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filepath}")


def main():
    print("="*70)
    print("SEMANTIC CHAOS EXPERIMENT")
    print("Testing Structural Memory via Rule Switching")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    experiment = SemanticChaosExperiment(device=device)
    
    # Run both models
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline Model")
    print("="*70)
    experiment.run_experiment(model_type='baseline')
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: SSNC Model")
    print("="*70)
    experiment.run_experiment(model_type='ssnc')
    
    # Analyze
    experiment.analyze_results()
    experiment.save_results()


if __name__ == "__main__":
    main()
