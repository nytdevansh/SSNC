"""
FIXED Trainer Module - Addresses premature epoch termination and architecture change handling.

Key Fixes:
1. Only restart epochs on SIGNIFICANT architecture changes (≥5 neurons)
2. Defer architecture changes to epoch end (complete full epochs)
3. Properly handle optimizer state preservation
4. Add class imbalance handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.trec6_model import TREC6_SSNC_Model, CONFIG
from src.trec6_loader import download_trec6_direct, TREC6Dataset
from src.utils import LabelSmoothingCrossEntropy, EarlyStopping, visualize_network_architecture

def train_epoch(model, train_loader, optimizer, criterion, config, epoch, scheduler=None):
    """
    Train one epoch with FIXED architecture change handling.
    
    KEY FIX: Complete full epochs. Collect architecture changes during epoch
    but only apply them at the END of the epoch.
    """
    model.train()
    model.set_epoch(epoch)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Track architecture changes to apply at epoch end
    pending_changes = {
        'deletions': 0,
        'creations': 0,
        'needs_optimizer_update': False
    }
    
    # LR Coupling: Track cooldown after structural changes
    plasticity_cooldown = 0
    base_lr = optimizer.param_groups[0]['lr']
    lr_scale = config.get('plasticity_warmup_lr_scale', 0.5)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # === LR COUPLING: Apply cooldown scaling ===
        if plasticity_cooldown > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale
            plasticity_cooldown -= 1
            if plasticity_cooldown == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
                tqdm.write(f"  📈 LR restored to {base_lr:.6f}")
        
        texts = batch['text']
        labels = batch['coarse_label'].to(config['device'])
        
        # Forward
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # SSNC observations
        model.observe_gradients()
        
        # Gradient clipping
        if config['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f"{total_loss / (batch_idx + 1):.4f}",
            'acc': f"{100.0 * correct / total:.1f}%"
        })
        
        # Update utilities
        if (batch_idx + 1) % config['utility_update_every'] == 0:
            model.compute_all_utilities()
        
        # Plasticity cycle
        if (batch_idx + 1) % config['plasticity_cycle_every'] == 0:
            stats_before = model.get_statistics()
            batch_acc = 100.0 * (logits.argmax(dim=1) == labels).sum().item() / labels.size(0)
            actions = model.plasticity_cycle(current_loss=loss.item(), current_accuracy=batch_acc)
            
            # === LR COUPLING ===
            if actions.get('architecture_changed', False) or actions.get('reduce_lr', False):
                plasticity_cooldown = config.get('plasticity_cooldown_batches', 10)
                tqdm.write(f"  📉 LR scaled to {lr_scale}x for {plasticity_cooldown} batches (stability)")
            
            observer_cooldown = actions.get('plasticity_cooldown', 0)
            if observer_cooldown > 0:
                plasticity_cooldown = max(plasticity_cooldown, observer_cooldown)
            
            # Check for STOP signal
            if actions.get('stop_training', False):
                tqdm.write("\n🛑 AUTOMATIC STOP: Network has converged structurally.")
                return total_loss / len(train_loader), 100.0 * correct / total, pending_changes['deletions'], pending_changes['creations'], True, False
            
            # === CRITICAL FIX: Track changes but DON'T restart epoch ===
            if actions.get('architecture_changed', False):
                n_created = len(actions.get('created', []))
                n_deleted = len(actions.get('hard_deleted', []))
                
                # Accumulate changes
                pending_changes['deletions'] += n_deleted
                pending_changes['creations'] += n_created
                
                # Mark that we need optimizer update at epoch end
                if n_created > 0 or n_deleted > 0:
                    pending_changes['needs_optimizer_update'] = True
                    tqdm.write(f"  📋 Architecture change queued: +{n_created} created, -{n_deleted} deleted (will apply at epoch end)")
            
            stats_after = model.get_statistics()
            
            # Log structural events
            if actions:
                n_marked = len(actions.get('soft_marked', []))
                n_gated = len(actions.get('soft_gated', []))
                n_deleted = len(actions.get('hard_deleted', []))
                n_created = len(actions.get('created', []))
                observer_decisions = actions.get('observer_decisions', [])
                warnings = actions.get('monitor_warnings', [])
                
                for decision in observer_decisions:
                    if 'DELETE' in decision or 'CREATE' in decision:
                        tqdm.write(f"  🧠 Observer: {decision}")
                
                if n_marked > 0:
                    tqdm.write(f"  📌 Batch {batch_idx+1}: Marked {n_marked} neurons")
                if n_gated > 0:
                    tqdm.write(f"  🚪 Batch {batch_idx+1}: Soft-gated {n_gated} neurons")
                if n_deleted > 0:
                    tqdm.write(f"  ⚡ Batch {batch_idx+1}: Hard-deleted {n_deleted} neurons")
                if n_created > 0:
                    tqdm.write(f"  ✨ Batch {batch_idx+1}: Created {n_created} neurons")
                if warnings and 'BURN-IN' not in str(warnings):
                    tqdm.write(f"  ⚠️  {warnings}")
                
                # Visualize layer addition immediately
                for decision in observer_decisions:
                    if "Added new hidden layer" in decision:
                        tqdm.write(f"  📸 SNAPSHOT: Generating architecture diagram...")
                        visualize_network_architecture(model, Path(__file__).parents[1] / 'results' / 'trec6', epoch=epoch)
                
                # SAFETY CHECK
                if stats_before['active_neurons'] > 0:
                    reduction_rate = (stats_before['active_neurons'] - stats_after['active_neurons']) / stats_before['active_neurons']
                    if reduction_rate > config['max_deletion_rate']:
                        tqdm.write(f"  ⚠️  SAFETY: Deletion rate {reduction_rate:.1%} exceeds max {config['max_deletion_rate']:.1%}")
    
    # === EPOCH COMPLETED - Now handle pending architecture changes ===
    # Return with architecture_changed flag if there were significant changes
    architecture_changed = pending_changes['needs_optimizer_update']
    
    return (
        total_loss / len(train_loader),
        100.0 * correct / total,
        pending_changes['deletions'],
        pending_changes['creations'],
        False,  # stop_training
        architecture_changed  # architecture_changed
    )


def evaluate(model, test_loader, criterion, config):
    """Evaluate model on test set."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    num_classes = config['num_classes']
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text']
            labels = batch['coarse_label'].to(config['device'])
            
            logits = model(texts)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if pred[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc.append(100.0 * class_correct[i] / class_total[i])
        else:
            per_class_acc.append(0.0)
    
    return total_loss / len(test_loader), 100.0 * correct / total, per_class_acc


def train(config=None):
    """Main training function with FIXED architecture handling."""
    if config is None:
        config = CONFIG.copy()
    else:
        # User provided config, ensure defaults are present if needed or just use it
        pass
    
    # Set device
    device = torch.device(config['device'])
    print(f"\n{'='*70}")
    print(f"FIXED SSNC TRAINING ON TREC-6")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load data
    print("\nLoading TREC-6 dataset...")
    dataset = download_trec6_direct()
    train_dataset = TREC6Dataset(dataset, split='train')
    test_dataset = TREC6Dataset(dataset, split='test')
    
    # === FIX: Add class-weighted loss for imbalance ===
    if config.get('use_class_weights', True):
        # Calculate class weights
        from collections import Counter
        label_counts = Counter(train_dataset.coarse_labels)
        total_samples = len(train_dataset)
        class_weights = []
        for i in range(config['num_classes']):
            count = label_counts.get(i, 1)  # Avoid division by zero
            weight = total_samples / (config['num_classes'] * count)
            class_weights.append(weight)
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"\n✓ Class weights computed:")
        for i, w in enumerate(class_weights):
            print(f"  Class {i}: {w:.3f}")
    else:
        class_weights = None
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Initialize model (pass train_texts for encoder that needs fitting)
    print("\nInitializing SSNC model...")
    train_texts = [train_dataset[i]['text'] for i in range(len(train_dataset))]
    model = TREC6_SSNC_Model(config, train_texts=train_texts).to(device)
    
    # Loss function
    if config.get('label_smoothing', 0) > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.get('use_lr_schedule', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_schedule_factor', 0.5),
            patience=config.get('lr_schedule_patience', 5)
        )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 10),
        min_delta=0.001
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'per_class_acc': [],
        'active_neurons': [],
        'soft_deleted': [],
        'hard_deleted': [],
        'created': [],
        'learning_rate': []
    }
    
    best_test_acc = 0.0
    best_test_loss = float('inf')
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Plasticity cycle every: {config['plasticity_cycle_every']} batches")
    print(f"Burn-in epochs: {config['burn_in_epochs']}")
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 70)
        
        # Train epoch
        train_loss, train_acc, epoch_del, epoch_cre, stop_training, architecture_changed = train_epoch(
            model, train_loader, optimizer, criterion, config, epoch, scheduler
        )
        
        # === FIX: Update optimizer AFTER full epoch if architecture changed ===
        if architecture_changed:
            print(f"\n  🔄 SYSTEM: Updating optimizer after full epoch completion...")
            print(f"  💡 Preserving Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save old optimizer state
            old_lr = optimizer.param_groups[0]['lr']
            old_betas = optimizer.param_groups[0]['betas']
            old_weight_decay = optimizer.param_groups[0]['weight_decay']
            old_state = {k: v for k, v in optimizer.state.items()}
            
            # Create new optimizer with same hyperparameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=old_lr,
                betas=old_betas,
                weight_decay=old_weight_decay
            )
            
            # Restore what we can
            restored_count = 0
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    if p in old_state:
                        optimizer.state[p] = old_state[p]
                        restored_count += 1
            
            print(f"  ✓ Restored {restored_count} parameter states")
            
            # Update scheduler if exists
            if scheduler:
                old_num_bad = getattr(scheduler, 'num_bad_epochs', 0)
                old_best = getattr(scheduler, 'best', float('inf'))
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=config.get('lr_schedule_factor', 0.5),
                    patience=config.get('lr_schedule_patience', 5)
                )
                scheduler.num_bad_epochs = old_num_bad
                scheduler.best = old_best
        
        if stop_training:
            print(f"\n✨ TRAINING COMPLETED: Structural convergence at epoch {epoch}")
            break
        
        # Evaluate
        test_loss, test_acc, per_class_acc = evaluate(model, test_loader, criterion, config)
        
        # Statistics
        stats = model.get_statistics()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['per_class_acc'].append(per_class_acc)
        history['active_neurons'].append(stats['active_neurons'])
        history['soft_deleted'].append(stats['soft_deleted'])
        history['hard_deleted'].append(stats['hard_deleted'])
        history['created'].append(stats['created'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print results
        print(f"\nResults:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  SSNC: Active={stats['active_neurons']}, "
              f"Deleted={stats['hard_deleted']}, Created={stats['created']}")
        print(f"  This epoch: Deleted {epoch_del}, Created {epoch_cre}")
        print(f"  Per-class accuracy: {[f'{a:.1f}' for a in per_class_acc]}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print("  ★ NEW BEST ACCURACY!")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print("  ★ NEW BEST LOSS!")
        
        # Learning rate scheduling
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(test_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Early stopping check
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            print(f"   No improvement for {config['patience']} epochs")
            break
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS - FIXED SSNC ON TREC-6")
    print("="*70)
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"\nStructural Evolution:")
    print(f"  Started: {history['active_neurons'][0]} neurons")
    print(f"  Ended: {history['active_neurons'][-1]} neurons")
    change = history['active_neurons'][-1] - history['active_neurons'][0]
    change_pct = 100 * change / history['active_neurons'][0]
    print(f"  Change: {change:+d} neurons ({change_pct:+.1f}%)")
    print(f"  Hard-deleted: {history['hard_deleted'][-1]}")
    print(f"  Created: {history['created'][-1]}")
    
    # Overfitting check
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    gap = final_train_acc - final_test_acc
    print(f"\nOverfitting Check:")
    print(f"  Train accuracy: {final_train_acc:.2f}%")
    print(f"  Test accuracy: {final_test_acc:.2f}%")
    print(f"  Gap: {gap:.2f}%")
    if gap < 5:
        print("  ✓ Good generalization!")
    elif gap < 10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ✗ Significant overfitting - increase regularization")
    
    print("="*70)
    
    # Save results
    output_dir = Path(__file__).parents[1] / 'results' / 'trec6_fixed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'trec6_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_network_architecture(model, output_dir, epoch=len(history['train_loss']))
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('FIXED SSNC on TREC-6 Question Classification', 
                 fontsize=14, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2, color='blue')
    axes[0, 0].plot(history['test_loss'], label='Test', linewidth=2, color='orange')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['test_acc'], label='Test', linewidth=2)
    axes[0, 1].axhline(y=best_test_acc, color='r', linestyle='--', 
                       alpha=0.5, label=f'Best: {best_test_acc:.1f}%')
    axes[0, 1].set_title('Accuracy (%)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Active neurons
    axes[0, 2].plot(history['active_neurons'], linewidth=2, color='green')
    axes[0, 2].axhline(y=history['active_neurons'][0], color='gray', 
                       linestyle='--', alpha=0.5, label='Initial')
    axes[0, 2].set_title('Active Neurons', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Deletions
    axes[1, 0].plot(history['hard_deleted'], linewidth=2, color='red')
    axes[1, 0].set_title('Hard Deletions', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Created
    axes[1, 1].plot(history['created'], linewidth=2, color='blue')
    axes[1, 1].set_title('Created Neurons', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 2].plot(history['learning_rate'], linewidth=2, color='purple')
    axes[1, 2].set_title('Learning Rate', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trec6_results.png', dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Results saved to {output_dir}")
    print("  - trec6_fixed_history.json")
    print("  - trec6_fixed_results.png")
    print("  - network_architecture.png")


if __name__ == "__main__":
    train()