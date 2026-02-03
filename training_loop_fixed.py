"""
Example Training Loop - FIXED VERSION
This shows how to properly integrate SSNC plasticity into your training
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_with_ssnc(
    model, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    optimizer,
    epochs: int = 50,
    plasticity_interval: int = 50,  # Run plasticity every N batches
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Training loop with SSNC plasticity properly integrated.
    
    Args:
        model: CompleteSSNC_MLP instance
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: PyTorch optimizer
        epochs: Number of epochs
        plasticity_interval: How often to run plasticity (in batches)
        device: Device to use
    """
    
    print("🚀 Starting SSNC Training with Plasticity\n")
    
    for epoch in range(epochs):
        # CRITICAL: Update epoch counter
        model.current_epoch = epoch
        model.train()
        
        # Training metrics
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Standard training step
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # === EXECUTE PLASTICITY ===
            if batch_idx > 0 and batch_idx % plasticity_interval == 0:
                current_acc = 100.0 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                
                # THIS IS THE CRITICAL CALL!
                plasticity_result = model.execute_plasticity_decisions(
                    current_loss=avg_loss,
                    current_accuracy=current_acc
                )
                
                # Print what happened
                stats = plasticity_result.get('stats', {})
                print(f"Batch {batch_idx:3d}: Loss={avg_loss:.4f}, Acc={current_acc:.1f}% | "
                      f"Active={stats.get('active', 0)}, "
                      f"Soft={stats.get('soft', 0)}, "
                      f"Hard={stats.get('hard', 0)}")
                
                # Show structural changes
                if plasticity_result.get('created'):
                    print(f"   ➕ Created {len(plasticity_result['created'])} neurons")
                
                if plasticity_result.get('hard_deleted'):
                    print(f"   🗑️  Deleted {len(plasticity_result['hard_deleted'])} neurons")
                
                if plasticity_result.get('observer_decisions'):
                    decisions = plasticity_result['observer_decisions'][:2]
                    print(f"   📊 {' | '.join(decisions)}")
        
        # Evaluate on test set
        test_acc, test_loss = evaluate(model, test_loader, device)
        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        # Get network statistics
        stats = model.get_statistics()
        print(f"Network: {stats['active_neurons']} active, "
              f"{stats['hard_deleted']} deleted, "
              f"{stats['created']} created")
        print(f"{'='*70}\n")
        
        # Early stopping if converged
        if plasticity_result.get('stop_training'):
            print("🏁 Training stopped: Network converged")
            break
    
    return model


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    test_acc = 100.0 * correct / total
    
    return test_acc, test_loss


def diagnostic_test(model, train_loader, device):
    """
    Quick diagnostic to verify SSNC is working.
    Run this BEFORE full training to catch issues early.
    """
    print("\n" + "="*70)
    print("🔬 SSNC DIAGNOSTIC TEST")
    print("="*70)
    
    model.train()
    model.current_epoch = 1  # Past burn-in
    
    # Get one batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    # Forward + backward
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    # Check utilities
    print("\n1️⃣ Checking utility computation...")
    utilities = model._collect_utilities()
    if not utilities:
        print("   ❌ FAILED: No utilities collected!")
        return False
    
    util_vals = list(utilities.values())
    print(f"   ✅ Collected {len(util_vals)} utilities")
    print(f"   Range: [{min(util_vals):.3f}, {max(util_vals):.3f}]")
    print(f"   Mean:  {sum(util_vals)/len(util_vals):.3f}")
    print(f"   Std:   {(sum((x-sum(util_vals)/len(util_vals))**2 for x in util_vals)/len(util_vals))**0.5:.3f}")
    
    # Check if utilities are in usable range
    if max(util_vals) < 0.1:
        print("   ⚠️  WARNING: Utilities very low - scaling may be wrong!")
    
    # Try plasticity
    print("\n2️⃣ Testing plasticity execution...")
    result = model.execute_plasticity_decisions(
        current_loss=loss.item(),
        current_accuracy=50.0  # Fake mid-training accuracy
    )
    
    print(f"   Observer decisions: {result.get('observer_decisions', ['None'])[:2]}")
    print(f"   Created: {len(result.get('created', []))}")
    print(f"   Hard deleted: {len(result.get('hard_deleted', []))}")
    print(f"   Soft deleted: {len(result.get('soft_gated', []))}")
    print(f"   Stats: {result.get('stats', {})}")
    
    # Verdict
    print("\n3️⃣ Verdict:")
    if len(result.get('created', [])) == 0 and len(result.get('hard_deleted', [])) == 0:
        print("   ⚠️  No structural changes yet (may be normal if utilities not ready)")
        print("   💡 If this persists after 5+ epochs, check threshold configuration")
        return True  # Not necessarily a failure
    else:
        print("   ✅ SSNC IS WORKING - Structural changes detected!")
        return True
    
    print("="*70 + "\n")


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    from src.ssnc import CompleteSSNC_MLP
    import torch.optim as optim
    
    # Create model with PROPER CONFIGURATION
    model = CompleteSSNC_MLP(
        input_dim=300,  # Your embedding dimension
        hidden_dims=[576],  # Start with one hidden layer
        output_dim=6,  # TREC-6 classes
        
        # CRITICAL: Plasticity settings
        burn_in_epochs=0,  # Start plasticity immediately
        enable_soft_deletion=True,
        enable_hard_deletion=True,
        enable_creation=True,
        enable_monitoring=True,
        
        # Budgets (increased for 576 neurons)
        max_deletions_per_cycle=15,  # ~2.6% of network
        max_creations_per_cycle=8,   # ~1.4% of network
        
        # Thresholds (will be adapted by percentile logic)
        deletion_threshold=0.05,
        creation_threshold=0.25,
        
        # Cycles
        marking_cycles=4,  # Faster confirmation
        hard_deletion_cycles=8,
        
        # Optional: Autonomous mode
        use_autonomous_mode=False,  # Use Observer for now
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Assuming you have train_loader and test_loader defined...
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Run diagnostic FIRST
    # diagnostic_test(model, train_loader, 'cuda')
    
    # Then train
    # trained_model = train_with_ssnc(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     optimizer=optimizer,
    #     epochs=50,
    #     plasticity_interval=50
    # )
