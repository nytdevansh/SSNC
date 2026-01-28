"""
SSNC PHASE 3: Utility Analysis Training Script

Loads trained PHASE 2 model and runs comprehensive utility validation.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

from ssnc_mlp import SSNC_MLP
from utility_analyzer import UtilityAnalyzer


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


def load_phase2_model(checkpoint_path: str = "checkpoints/phase2/best_model.pt") -> SSNC_MLP:
    """Load trained PHASE 2 model."""
    
    # Create model architecture
    model = SSNC_MLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        activation='tanh'
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded PHASE 2 model from: {checkpoint_path}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    return model


def main():
    """Run PHASE 3 utility validation."""
    
    print("="*60)
    print("SSNC PHASE 3: UTILITY VALIDATION & REFINEMENT")
    print("="*60)
    print()
    print("This phase validates that utility scores are meaningful")
    print("by testing correlation with ground truth importance.")
    print()
    print("Tests:")
    print("  1. Ablation test - do utilities predict importance?")
    print("  2. Stability test - are utilities consistent?")
    print("  3. Component analysis - which parts matter most?")
    print("  4. Noise sensitivity - are utilities robust?")
    print("="*60)
    print()
    
    # Check if PHASE 2 checkpoint exists
    checkpoint_path = Path("checkpoints/phase2/best_model.pt")
    if not checkpoint_path.exists():
        print("❌ ERROR: PHASE 2 checkpoint not found!")
        print(f"   Expected: {checkpoint_path}")
        print()
        print("Please run PHASE 2 training first:")
        print("  python train_ssnc_phase2.py")
        return
    
    # Load model
    print("Loading PHASE 2 model...")
    model = load_phase2_model(str(checkpoint_path))
    print()
    
    # Load data
    print("Loading MNIST dataset...")
    _, test_loader = load_mnist(batch_size=128)
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print()
    
    # Create analyzer
    analyzer = UtilityAnalyzer(model)
    
    # Run all validation tests
    results = analyzer.run_all_tests(
        test_loader=test_loader,
        save_dir="analysis/phase3"
    )
    
    print("\n" + "="*60)
    print("PHASE 3 VALIDATION COMPLETE")
    print("="*60)
    print()
    print("Results saved to: analysis/phase3/")
    print("  - validation_report.txt   (detailed report)")
    print("  - utility_analysis.png    (visualizations)")
    print("  - utility_analysis_results.json (raw data)")
    print()
    
    # Decision point
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    
    # Check if utility passed validation
    ablation_corr = results['ablation_test']['correlation'] if results['ablation_test'] else 0
    stability_cv = results['stability_test']['coefficient_of_variation'] if results['stability_test'] else 1
    noise_corr = results['noise_sensitivity']['avg_correlation'] if results['noise_sensitivity'] else 0
    
    passes = sum([
        ablation_corr > 0.3,
        stability_cv < 0.1,
        noise_corr > 0.7
    ])
    
    if passes >= 2:
        print()
        print("✅ UTILITY METRIC VALIDATED!")
        print()
        print("The utility metric is meaningful and robust.")
        print("Ready to proceed to PHASE 4: Enable Neuron Learning")
        print()
        print("Run next:")
        print("  python train_ssnc_phase4.py")
    else:
        print()
        print("⚠️  UTILITY METRIC NEEDS REFINEMENT")
        print()
        print("Current results:")
        print(f"  Ablation correlation: {ablation_corr:.3f} (want >0.3)")
        print(f"  Stability CV: {stability_cv:.4f} (want <0.1)")
        print(f"  Noise resistance: {noise_corr:.3f} (want >0.7)")
        print()
        print("Consider:")
        print("  1. Adjusting utility formula weights (in ssnc_neuron.py)")
        print("  2. Using longer smoothing windows")
        print("  3. Adding more training epochs before testing")
        print()
        print("Or proceed to PHASE 4 anyway to test local learning.")
    
    print("="*60)


if __name__ == "__main__":
    main()
