"""
SSNC PHASE 3: Utility Computation Refinement & Validation

Goal: Ensure utility metric is meaningful before neurons act on it.

Key questions this phase answers:
1. Do useful neurons score higher?
2. Does utility predict neuron importance?
3. Is noise dominating the signal?
4. Can we improve the formula?

This is pure analysis - no changes to training yet.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

from ssnc_mlp import SSNC_MLP, print_ssnc_stats
from torch.utils.data import DataLoader


class UtilityAnalyzer:
    """
    Analyzes utility metrics to validate they're meaningful.
    
    Tests:
    1. Correlation with ablation impact (remove neuron → loss change)
    2. Stability over time (does utility drift randomly?)
    3. Component analysis (which parts of utility matter most?)
    4. Noise sensitivity (random perturbations)
    """
    
    def __init__(self, model: SSNC_MLP, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.results = {
            'ablation_test': None,
            'stability_test': None,
            'component_analysis': None,
            'noise_sensitivity': None
        }
    
    def run_all_tests(
        self,
        test_loader: DataLoader,
        save_dir: str = "analysis/phase3"
    ):
        """Run complete utility validation suite."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("PHASE 3: UTILITY VALIDATION SUITE")
        print("="*60)
        
        # Test 1: Ablation test (ground truth)
        print("\n[1/4] Running ablation test (this will take time)...")
        self.results['ablation_test'] = self.ablation_test(test_loader)
        
        # Test 2: Stability test
        print("\n[2/4] Running stability test...")
        self.results['stability_test'] = self.stability_test(test_loader)
        
        # Test 3: Component analysis
        print("\n[3/4] Running component analysis...")
        self.results['component_analysis'] = self.component_analysis()
        
        # Test 4: Noise sensitivity
        print("\n[4/4] Running noise sensitivity test...")
        self.results['noise_sensitivity'] = self.noise_sensitivity_test(test_loader)
        
        # Generate report
        print("\n" + "="*60)
        print("GENERATING ANALYSIS REPORT")
        print("="*60)
        
        self._generate_report(save_path)
        self._plot_results(save_path)
        
        # Save raw results
        with open(save_path / 'utility_analysis_results.json', 'w') as f:
            json.dump(self._serialize_results(), f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to: {save_path}")
        
        return self.results
    
    def ablation_test(
        self,
        test_loader: DataLoader,
        n_samples: int = 20
    ) -> Dict:
        """
        Test 1: Does utility predict ablation impact?
        
        For random neurons, measure:
        - Their utility score
        - Impact on loss when removed (ground truth importance)
        
        High correlation = utility is meaningful.
        """
        print("  Testing if utility predicts neuron importance...")
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Get baseline loss
        baseline_loss = self._compute_loss(test_loader, criterion)
        print(f"  Baseline loss: {baseline_loss:.4f}")
        
        # Sample neurons to test
        all_neurons = []
        for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
            for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                all_neurons.append({
                    'layer': layer_idx,
                    'neuron': neuron_idx,
                    'utility': neuron.utility
                })
        
        # Sample diverse utilities (stratified sampling)
        sorted_neurons = sorted(all_neurons, key=lambda x: x['utility'])
        step = len(sorted_neurons) // n_samples
        sampled_neurons = [sorted_neurons[i * step] for i in range(n_samples)]
        
        # Test each neuron
        results = []
        for i, neuron_info in enumerate(sampled_neurons):
            layer_idx = neuron_info['layer']
            neuron_idx = neuron_info['neuron']
            utility = neuron_info['utility']
            
            # Ablate neuron (zero out its output)
            impact = self._ablate_neuron_and_measure(
                layer_idx, neuron_idx, test_loader, criterion, baseline_loss
            )
            
            results.append({
                'layer': layer_idx,
                'neuron': neuron_idx,
                'utility': utility,
                'loss_impact': impact
            })
            
            print(f"  [{i+1}/{n_samples}] Layer {layer_idx}, Neuron {neuron_idx}: "
                  f"Utility={utility:.3f}, Impact={impact:.4f}")
        
        # Compute correlation
        utilities = [r['utility'] for r in results]
        impacts = [r['loss_impact'] for r in results]
        
        correlation = np.corrcoef(utilities, impacts)[0, 1]
        
        print(f"\n  ✓ Correlation (utility vs impact): {correlation:.3f}")
        
        if correlation > 0.5:
            print("  ✓ STRONG correlation - utility is meaningful!")
        elif correlation > 0.3:
            print("  ⚠️  MODERATE correlation - utility has signal")
        else:
            print("  ✗ WEAK correlation - utility needs refinement")
        
        return {
            'correlation': correlation,
            'baseline_loss': baseline_loss,
            'results': results
        }
    
    def _ablate_neuron_and_measure(
        self,
        layer_idx: int,
        neuron_idx: int,
        test_loader: DataLoader,
        criterion: nn.Module,
        baseline_loss: float
    ) -> float:
        """
        Temporarily zero out a neuron and measure loss change.
        
        Returns:
            Loss delta (positive = neuron was helping)
        """
        # Register hook to zero out neuron
        def ablation_hook(module, input, output):
            output[:, neuron_idx] = 0.0
            return output
        
        # Attach hook
        layer = self.model.layers[layer_idx]
        handle = layer.register_forward_hook(ablation_hook)
        
        # Measure loss with neuron ablated
        ablated_loss = self._compute_loss(test_loader, criterion)
        
        # Remove hook
        handle.remove()
        
        # Return impact (positive = neuron was reducing loss)
        return ablated_loss - baseline_loss
    
    def stability_test(
        self,
        test_loader: DataLoader,
        n_runs: int = 10
    ) -> Dict:
        """
        Test 2: Is utility stable across forward passes?
        
        Run multiple forward passes with same data.
        Utility should be stable (low variance).
        """
        print("  Testing utility stability over forward passes...")
        
        self.model.eval()
        
        # Get a batch
        data_iter = iter(test_loader)
        data, _ = next(data_iter)
        data = data.to(self.device)
        
        # Track utilities over multiple passes
        utility_history = defaultdict(list)
        
        with torch.no_grad():
            for run in range(n_runs):
                # Forward pass
                _ = self.model(data)
                self.model.compute_all_utilities()
                
                # Record utilities
                for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
                    for neuron_idx, neuron in enumerate(ssnc_layer.neurons):
                        key = f"L{layer_idx}_N{neuron_idx}"
                        utility_history[key].append(neuron.utility)
        
        # Compute stability metrics
        variances = []
        mean_utilities = []
        
        for key, utilities in utility_history.items():
            mean_util = np.mean(utilities)
            variance = np.var(utilities)
            
            variances.append(variance)
            mean_utilities.append(mean_util)
        
        avg_variance = np.mean(variances)
        max_variance = np.max(variances)
        
        # Coefficient of variation (normalized)
        cvs = [
            np.sqrt(var) / (mean + 1e-6)
            for var, mean in zip(variances, mean_utilities)
        ]
        avg_cv = np.mean(cvs)
        
        print(f"  Mean variance: {avg_variance:.6f}")
        print(f"  Max variance: {max_variance:.6f}")
        print(f"  Coefficient of variation: {avg_cv:.4f}")
        
        if avg_cv < 0.05:
            print("  ✓ EXCELLENT stability")
        elif avg_cv < 0.1:
            print("  ✓ GOOD stability")
        else:
            print("  ⚠️  Utilities are noisy")
        
        return {
            'mean_variance': avg_variance,
            'max_variance': max_variance,
            'coefficient_of_variation': avg_cv,
            'n_runs': n_runs
        }
    
    def component_analysis(self) -> Dict:
        """
        Test 3: Which utility components matter most?
        
        Break down utility into components:
        - Activation magnitude
        - Activation variance
        - Error correlation
        - Inactivity penalty
        
        See which components have highest variance (signal).
        """
        print("  Analyzing utility components...")
        
        # Recompute all utilities with component tracking
        component_data = {
            'activation_magnitude': [],
            'activation_variance': [],
            'error_correlation': [],
            'inactivity_penalty': [],
            'final_utility': []
        }
        
        for ssnc_layer in self.model.ssnc_layers:
            for neuron in ssnc_layer.neurons:
                # Manually compute components (same as neuron.compute_utility)
                act_mag = abs(neuron.activation_mean)
                act_score = np.tanh(act_mag)
                
                var_score = np.tanh(np.sqrt(neuron.activation_var))
                
                err_score = neuron.error_correlation
                
                if neuron.inactivity_counter > 50:
                    inact_penalty = np.exp(-neuron.inactivity_counter / 100.0)
                else:
                    inact_penalty = 1.0
                
                component_data['activation_magnitude'].append(act_score)
                component_data['activation_variance'].append(var_score)
                component_data['error_correlation'].append(err_score)
                component_data['inactivity_penalty'].append(inact_penalty)
                component_data['final_utility'].append(neuron.utility)
        
        # Compute statistics for each component
        analysis = {}
        for component, values in component_data.items():
            analysis[component] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }
        
        # Print summary
        print("\n  Component Statistics:")
        for component, stats in analysis.items():
            print(f"    {component:25s}: "
                  f"μ={stats['mean']:.3f}, σ={stats['std']:.3f}, "
                  f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Identify most informative component (highest std)
        stds = {comp: analysis[comp]['std'] for comp in component_data.keys()}
        most_informative = max(stds.items(), key=lambda x: x[1])
        
        print(f"\n  Most informative component: {most_informative[0]} "
              f"(σ={most_informative[1]:.3f})")
        
        return analysis
    
    def noise_sensitivity_test(
        self,
        test_loader: DataLoader,
        noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1]
    ) -> Dict:
        """
        Test 4: How sensitive is utility to input noise?
        
        Add noise to inputs and see if utility rankings change.
        Robust utility should be noise-resistant.
        """
        print("  Testing noise sensitivity...")
        
        self.model.eval()
        
        # Get clean utilities (no noise)
        data_iter = iter(test_loader)
        data, _ = next(data_iter)
        data = data.to(self.device)
        
        clean_utilities = self._get_all_utilities(data)
        
        # Test with different noise levels
        results = {}
        
        for noise_level in noise_levels:
            if noise_level == 0.0:
                continue
            
            # Add noise
            noisy_data = data + torch.randn_like(data) * noise_level
            noisy_utilities = self._get_all_utilities(noisy_data)
            
            # Compute rank correlation (Spearman)
            from scipy.stats import spearmanr
            
            correlation, _ = spearmanr(clean_utilities, noisy_utilities)
            
            results[noise_level] = {
                'correlation': float(correlation),
                'mean_change': float(np.mean(np.abs(
                    np.array(clean_utilities) - np.array(noisy_utilities)
                )))
            }
            
            print(f"    Noise σ={noise_level:.2f}: "
                  f"rank_correlation={correlation:.3f}, "
                  f"mean_Δ={results[noise_level]['mean_change']:.4f}")
        
        # Overall noise resistance score
        correlations = [r['correlation'] for r in results.values()]
        avg_correlation = np.mean(correlations)
        
        if avg_correlation > 0.9:
            print(f"\n  ✓ EXCELLENT noise resistance (avg ρ={avg_correlation:.3f})")
        elif avg_correlation > 0.7:
            print(f"  ✓ GOOD noise resistance (avg ρ={avg_correlation:.3f})")
        else:
            print(f"  ⚠️  POOR noise resistance (avg ρ={avg_correlation:.3f})")
        
        return {
            'noise_levels': noise_levels,
            'results': results,
            'avg_correlation': avg_correlation
        }
    
    def _compute_loss(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        """Compute average loss over test set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                n_batches += 1
                
                if n_batches >= 10:  # Limit for speed
                    break
        
        return total_loss / n_batches
    
    def _get_all_utilities(self, data: torch.Tensor) -> List[float]:
        """Get utilities for all neurons after forward pass."""
        with torch.no_grad():
            _ = self.model(data)
            self.model.compute_all_utilities()
        
        utilities = []
        for ssnc_layer in self.model.ssnc_layers:
            for neuron in ssnc_layer.neurons:
                utilities.append(neuron.utility)
        
        return utilities
    
    def _generate_report(self, save_path: Path):
        """Generate human-readable report."""
        report = []
        report.append("="*60)
        report.append("PHASE 3: UTILITY VALIDATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Test 1: Ablation
        if self.results['ablation_test']:
            report.append("TEST 1: Ablation Test (Ground Truth Validation)")
            report.append("-" * 60)
            corr = self.results['ablation_test']['correlation']
            report.append(f"Correlation (utility vs impact): {corr:.3f}")
            
            if corr > 0.5:
                report.append("✓ PASS - Utility strongly predicts importance")
            elif corr > 0.3:
                report.append("⚠️  MARGINAL - Utility has signal but could improve")
            else:
                report.append("✗ FAIL - Utility doesn't predict importance")
            report.append("")
        
        # Test 2: Stability
        if self.results['stability_test']:
            report.append("TEST 2: Stability Test")
            report.append("-" * 60)
            cv = self.results['stability_test']['coefficient_of_variation']
            report.append(f"Coefficient of variation: {cv:.4f}")
            
            if cv < 0.05:
                report.append("✓ PASS - Excellent stability")
            elif cv < 0.1:
                report.append("✓ PASS - Good stability")
            else:
                report.append("⚠️  WARNING - Utilities are noisy")
            report.append("")
        
        # Test 3: Components
        if self.results['component_analysis']:
            report.append("TEST 3: Component Analysis")
            report.append("-" * 60)
            
            # Find most informative
            stds = {
                comp: data['std']
                for comp, data in self.results['component_analysis'].items()
            }
            most_informative = max(stds.items(), key=lambda x: x[1])
            
            report.append(f"Most informative component: {most_informative[0]}")
            report.append(f"  (std = {most_informative[1]:.3f})")
            report.append("")
        
        # Test 4: Noise
        if self.results['noise_sensitivity']:
            report.append("TEST 4: Noise Sensitivity")
            report.append("-" * 60)
            avg_corr = self.results['noise_sensitivity']['avg_correlation']
            report.append(f"Average rank correlation: {avg_corr:.3f}")
            
            if avg_corr > 0.9:
                report.append("✓ PASS - Excellent noise resistance")
            elif avg_corr > 0.7:
                report.append("✓ PASS - Good noise resistance")
            else:
                report.append("⚠️  WARNING - Sensitive to noise")
            report.append("")
        
        # Overall verdict
        report.append("="*60)
        report.append("OVERALL VERDICT")
        report.append("="*60)
        
        # Count passes
        passes = 0
        warnings = 0
        
        if self.results['ablation_test']:
            if self.results['ablation_test']['correlation'] > 0.3:
                passes += 1
            else:
                warnings += 1
        
        if self.results['stability_test']:
            if self.results['stability_test']['coefficient_of_variation'] < 0.1:
                passes += 1
            else:
                warnings += 1
        
        if self.results['noise_sensitivity']:
            if self.results['noise_sensitivity']['avg_correlation'] > 0.7:
                passes += 1
            else:
                warnings += 1
        
        report.append(f"Tests passed: {passes}/3")
        report.append(f"Warnings: {warnings}")
        report.append("")
        
        if passes >= 2:
            report.append("✅ UTILITY METRIC IS VALID")
            report.append("Ready to proceed to PHASE 4 (neuron learning)")
        else:
            report.append("⚠️  UTILITY METRIC NEEDS REFINEMENT")
            report.append("Consider adjusting weights in utility formula")
        
        report.append("="*60)
        
        # Save report
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open(save_path / 'validation_report.txt', 'w') as f:
            f.write(report_text)
    
    def _plot_results(self, save_path: Path):
        """Generate visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Utility vs Impact (ablation test)
        if self.results['ablation_test']:
            results = self.results['ablation_test']['results']
            utilities = [r['utility'] for r in results]
            impacts = [r['loss_impact'] for r in results]
            
            axes[0, 0].scatter(utilities, impacts, alpha=0.6)
            axes[0, 0].set_xlabel('Utility Score')
            axes[0, 0].set_ylabel('Loss Impact (Ablation)')
            axes[0, 0].set_title('Utility vs Ground Truth Importance')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line (with error handling)
            try:
                if len(utilities) > 1 and len(set(utilities)) > 1 and len(set(impacts)) > 1:
                    z = np.polyfit(utilities, impacts, 1)
                    p = np.poly1d(z)
                    axes[0, 0].plot(utilities, p(utilities), "r--", alpha=0.8)
            except (np.linalg.LinAlgError, RuntimeWarning):
                # Skip trend line if data is degenerate or SVD fails
                pass
        
        # Plot 2: Component distributions
        if self.results['component_analysis']:
            components = ['activation_magnitude', 'activation_variance', 
                         'error_correlation', 'final_utility']
            means = [self.results['component_analysis'][c]['mean'] for c in components]
            stds = [self.results['component_analysis'][c]['std'] for c in components]
            
            x = range(len(components))
            axes[0, 1].bar(x, means, yerr=stds, alpha=0.7, capsize=5)
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([c.replace('_', '\n') for c in components], 
                                       fontsize=8)
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_title('Utility Component Statistics')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Noise sensitivity
        if self.results['noise_sensitivity']:
            noise_levels = self.results['noise_sensitivity']['noise_levels'][1:]
            correlations = [
                self.results['noise_sensitivity']['results'][nl]['correlation']
                for nl in noise_levels
            ]
            
            axes[1, 0].plot(noise_levels, correlations, 'o-', linewidth=2)
            axes[1, 0].set_xlabel('Noise Level (σ)')
            axes[1, 0].set_ylabel('Rank Correlation with Clean')
            axes[1, 0].set_title('Noise Sensitivity')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0.7, color='r', linestyle='--', 
                              label='Acceptable threshold', alpha=0.5)
            axes[1, 0].legend()
        
        # Plot 4: Utility distribution per layer
        layer_utilities = []
        layer_labels = []
        
        for layer_idx, ssnc_layer in enumerate(self.model.ssnc_layers):
            utilities = [n.utility for n in ssnc_layer.neurons]
            layer_utilities.append(utilities)
            layer_labels.append(f'Layer {layer_idx}')
        
        axes[1, 1].boxplot(layer_utilities, labels=layer_labels)
        axes[1, 1].set_ylabel('Utility Score')
        axes[1, 1].set_title('Utility Distribution by Layer')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path / 'utility_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {save_path / 'utility_analysis.png'}")
        plt.close()
    
    def _serialize_results(self) -> Dict:
        """Convert results to JSON-serializable format."""
        serialized = {}
        
        for test_name, result in self.results.items():
            if result is None:
                continue
            
            # Convert numpy types to Python types
            serialized[test_name] = self._convert_to_serializable(result)
        
        return serialized
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy/torch types to Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


if __name__ == "__main__":
    print("SSNC PHASE 3: Utility Validation")
    print("="*60)
    print("\nThis module requires a trained PHASE 2 model.")
    print("Run from train_ssnc_phase3.py instead.")
