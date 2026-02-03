
import torch
import torch.nn.functional as F
from src.trec6_model import TREC6_SSNC_Model, CONFIG
from src.trec6_loader import download_trec6_direct, TREC6Dataset
from torch.utils.data import DataLoader

def diagnostic_test():
    print("\n" + "="*70)
    print("🔬 SSNC DIAGNOSTIC TEST")
    print("="*70)
    
    # Load Config
    config = CONFIG.copy()
    config['device'] = 'cpu' # Use CPU for diagnostic
    
    # Create Model
    model = TREC6_SSNC_Model(config)
    model.train()
    model.set_epoch(1)
    
    # Load Data (Minimal)
    data_dict = download_trec6_direct()
    dataset = TREC6Dataset(data_dict, split='train')
    loader = DataLoader(dataset, batch_size=32, collate_fn=lambda b: {'text': [x['text'] for x in b], 'coarse_label': torch.tensor([x['coarse_label'] for x in b])})
    
    # Get one batch
    batch = next(iter(loader))
    texts = batch['text']
    targets = batch['coarse_label']
    
    # Forward
    output = model(texts)
    loss = F.cross_entropy(output, targets)
    loss.backward()
    
    # Update utilities
    model.compute_all_utilities()
    
    # Check utilities
    print("\n1️⃣ Checking utility computation...")
    utilities = model.ssnc_mlp._collect_utilities()
    if not utilities:
        print("   ❌ FAILED: No utilities collected!")
        return
    
    util_vals = list(utilities.values())
    print(f"   ✅ Collected {len(util_vals)} utilities")
    print(f"   Range: [{min(util_vals):.3f}, {max(util_vals):.3f}]")
    print(f"   Mean:  {sum(util_vals)/len(util_vals):.3f}")
    
    # Check if utilities are in usable range (should be >0.1 due to scaling)
    if max(util_vals) < 0.1:
        print("   ⚠️  WARNING: Utilities very low - scaling might be wrong!")
    else:
        print("   ✅ Utilities scaled correctly")
        
    # Check Observer Thresholds (if using Observer)
    if not config['use_autonomous_mode'] and hasattr(model.ssnc_mlp, 'monitor'):
        obs = model.ssnc_mlp.monitor
        # Run one analysis to set thresholds
        model.ssnc_mlp._observe_layer(output, 0) # Dummy observation
        # Just manually trigger analyze
        layer_sizes = [len(l) for l in model.ssnc_mlp.ssnc_neurons]
        decisions = obs.analyze_network_state(utilities, sum(layer_sizes), loss.item(), 1, layer_sizes, 50.0)
        
        print(f"\n2️⃣ Observer Thresholds:")
        print(f"   Deletion Threshold (15%): {obs.deletion_threshold:.3f}")
        print(f"   Creation Threshold (85%): {obs.creation_threshold:.3f}")
        
    # Execution Test
    print("\n3️⃣ Testing Plasticity Execution...")
    result = model.ssnc_mlp.execute_plasticity_decisions(loss.item(), 50.0)
    
    print(f"   Observer decisions: {result.get('observer_decisions', [])}")
    print(f"   Created: {len(result.get('created', []))}")
    print(f"   Hard deleted: {len(result.get('hard_deleted', []))}")
    print(f"   Soft marked: {len(result.get('soft_marked', []))}")
    
    if len(result.get('soft_marked', [])) > 0 or len(result.get('created', [])) > 0:
        print("   ✅ ACTION DETECTED!")
    else:
        print("   ⚠️  No action taken (might be normal for first step)")

if __name__ == "__main__":
    diagnostic_test()
