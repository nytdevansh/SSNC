"""
SSNC PHASE 5: The Observer

Global coordinator that:
- Monitors all neurons
- Detects structural issues
- Makes deletion/creation decisions
- Maintains system health

PHASE 5: READ-ONLY MODE
Observer watches and logs decisions but doesn't act yet.
This validates decision logic before giving it authority.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class StructuralDecision:
    """Records a decision the Observer would make."""
    decision_type: str  # 'delete', 'create', 'redistribute'
    layer_id: int
    neuron_id: Optional[int]
    reason: str
    utility: Optional[float]
    priority: float  # How urgent (0-1)
    timestamp: int


class SSNCObserver:
    """
    Global observer that monitors network health and makes structural decisions.
    
    PHASE 5: Read-only mode - observes and logs but doesn't act.
    PHASE 6+: Active mode - can execute decisions.
    
    The Observer is the "brain" of SSNC - it sees everything and decides
    when structural changes are needed.
    """
    
    def __init__(
        self,
        readonly: bool = True,
        deletion_threshold: float = 0.05,
        inactivity_threshold: int = 100,
        overload_threshold: float = 0.8
    ):
        """
        Args:
            readonly: If True, log decisions but don't execute
            deletion_threshold: Utility below this = deletion candidate
            inactivity_threshold: Inactive steps before deletion
            overload_threshold: Utility above this = possible overload
        """
        self.readonly = readonly
        self.deletion_threshold = deletion_threshold
        self.inactivity_threshold = inactivity_threshold
        self.overload_threshold = overload_threshold
        
        # Decision tracking
        self.decisions = []
        self.decision_history = defaultdict(list)
        
        # Metrics tracking
        self.observation_count = 0
        self.metrics_history = []
        
        # Structural health
        self.health_score = 1.0
        self.health_history = []
    
    def observe(self, model_state: Dict) -> Dict:
        """
        Main observation cycle.
        
        Args:
            model_state: Complete state from SSNC_MLP
                {
                    'layers': [{neuron states}, ...],
                    'global': {aggregate stats},
                    'performance': {loss, accuracy}
                }
        
        Returns:
            Observation report with decisions
        """
        self.observation_count += 1
        
        # Compute structural health
        health = self._assess_structural_health(model_state)
        self.health_score = health['overall_score']
        self.health_history.append(self.health_score)
        
        # Identify issues
        issues = self._identify_issues(model_state)
        
        # Generate decisions
        decisions = self._generate_decisions(model_state, issues)
        
        # Log decisions
        for decision in decisions:
            self.decisions.append(decision)
            self.decision_history[decision.decision_type].append(decision)
        
        # Create report
        report = {
            'observation_id': self.observation_count,
            'health': health,
            'issues': issues,
            'decisions': [self._decision_to_dict(d) for d in decisions],
            'readonly': self.readonly
        }
        
        self.metrics_history.append(report)
        
        return report
    
    def _assess_structural_health(self, model_state: Dict) -> Dict:
        """
        Compute overall structural health score.
        
        Healthy network:
        - Utility well distributed (not all similar)
        - Few inactive neurons
        - No single neuron carrying too much load
        - Balanced across layers
        """
        health = {
            'overall_score': 1.0,
            'components': {}
        }
        
        global_stats = model_state['global']
        layer_stats = model_state['layers']
        
        # Component 1: Utility distribution (want high variance)
        all_utilities = []
        for layer in layer_stats:
            all_utilities.extend(layer['utilities']['distribution'])
        
        utility_std = np.std(all_utilities)
        utility_mean = np.mean(all_utilities)
        
        # Good: std > 0.1, Bad: std < 0.05
        utility_health = np.clip(utility_std / 0.1, 0.0, 1.0)
        health['components']['utility_diversity'] = utility_health
        
        # Component 2: Inactive neurons (want few)
        inactive_rate = global_stats['inactive_neurons'] / global_stats['total_neurons']
        # Good: <5%, Bad: >20%
        inactive_health = 1.0 - np.clip(inactive_rate / 0.2, 0.0, 1.0)
        health['components']['activity_health'] = inactive_health
        
        # Component 3: Layer balance
        layer_utilities = [layer['utilities']['mean'] for layer in layer_stats]
        layer_balance = 1.0 - (np.std(layer_utilities) / (np.mean(layer_utilities) + 1e-6))
        layer_balance = np.clip(layer_balance, 0.0, 1.0)
        health['components']['layer_balance'] = layer_balance
        
        # Component 4: No extreme outliers (overloaded neurons)
        max_utility = max(all_utilities)
        if max_utility > self.overload_threshold:
            overload_penalty = (max_utility - self.overload_threshold) / (1.0 - self.overload_threshold)
            overload_health = 1.0 - np.clip(overload_penalty, 0.0, 0.5)
        else:
            overload_health = 1.0
        health['components']['overload_health'] = overload_health
        
        # Overall score (weighted average)
        health['overall_score'] = (
            0.3 * utility_health +
            0.3 * inactive_health +
            0.2 * layer_balance +
            0.2 * overload_health
        )
        
        return health
    
    def _identify_issues(self, model_state: Dict) -> List[Dict]:
        """
        Identify specific structural issues.
        
        Returns list of issues with severity.
        """
        issues = []
        
        layer_stats = model_state['layers']
        
        for layer_idx, layer in enumerate(layer_stats):
            # Issue 1: Dead neurons (low utility + inactive)
            utilities = layer['utilities']['distribution']
            
            for neuron_idx, utility in enumerate(utilities):
                if utility < self.deletion_threshold:
                    issues.append({
                        'type': 'low_utility',
                        'layer': layer_idx,
                        'neuron': neuron_idx,
                        'utility': utility,
                        'severity': 1.0 - (utility / self.deletion_threshold)
                    })
            
            # Issue 2: Overloaded neurons
            max_utility = layer['utilities']['max']
            if max_utility > self.overload_threshold:
                # Find which neuron
                for neuron_idx, utility in enumerate(utilities):
                    if utility >= max_utility * 0.95:  # Top neurons
                        issues.append({
                            'type': 'overload',
                            'layer': layer_idx,
                            'neuron': neuron_idx,
                            'utility': utility,
                            'severity': (utility - self.overload_threshold) / (1.0 - self.overload_threshold)
                        })
            
            # Issue 3: Layer imbalance
            if layer_idx > 0:
                prev_mean = layer_stats[layer_idx - 1]['utilities']['mean']
                curr_mean = layer['utilities']['mean']
                
                if curr_mean < prev_mean * 0.5:  # Much lower than previous
                    issues.append({
                        'type': 'layer_imbalance',
                        'layer': layer_idx,
                        'neuron': None,
                        'severity': 1.0 - (curr_mean / prev_mean)
                    })
        
        return issues
    
    def _generate_decisions(
        self,
        model_state: Dict,
        issues: List[Dict]
    ) -> List[StructuralDecision]:
        """
        Generate structural decisions based on observed issues.
        
        PHASE 5: Only logging, not executing.
        """
        decisions = []
        
        # Sort issues by severity
        sorted_issues = sorted(issues, key=lambda x: x['severity'], reverse=True)
        
        for issue in sorted_issues:
            if issue['type'] == 'low_utility':
                # Decision: Mark for deletion
                decision = StructuralDecision(
                    decision_type='delete',
                    layer_id=issue['layer'],
                    neuron_id=issue['neuron'],
                    reason=f"Low utility ({issue['utility']:.3f}) below threshold",
                    utility=issue['utility'],
                    priority=issue['severity'],
                    timestamp=self.observation_count
                )
                decisions.append(decision)
            
            elif issue['type'] == 'overload':
                # Decision: Consider duplication or redistribution
                decision = StructuralDecision(
                    decision_type='create',
                    layer_id=issue['layer'],
                    neuron_id=issue['neuron'],
                    reason=f"Overloaded neuron ({issue['utility']:.3f}), consider duplication",
                    utility=issue['utility'],
                    priority=issue['severity'],
                    timestamp=self.observation_count
                )
                decisions.append(decision)
            
            elif issue['type'] == 'layer_imbalance':
                # Decision: Redistribute capacity
                decision = StructuralDecision(
                    decision_type='redistribute',
                    layer_id=issue['layer'],
                    neuron_id=None,
                    reason=f"Layer {issue['layer']} underutilized relative to previous",
                    utility=None,
                    priority=issue['severity'],
                    timestamp=self.observation_count
                )
                decisions.append(decision)
        
        return decisions
    
    def get_decision_summary(self) -> Dict:
        """Get summary of all decisions made."""
        summary = {
            'total_decisions': len(self.decisions),
            'by_type': {},
            'by_priority': {
                'high': 0,    # priority > 0.7
                'medium': 0,  # 0.3-0.7
                'low': 0      # < 0.3
            }
        }
        
        for decision_type, decisions in self.decision_history.items():
            summary['by_type'][decision_type] = len(decisions)
        
        for decision in self.decisions:
            if decision.priority > 0.7:
                summary['by_priority']['high'] += 1
            elif decision.priority > 0.3:
                summary['by_priority']['medium'] += 1
            else:
                summary['by_priority']['low'] += 1
        
        return summary
    
    def get_health_trend(self, window: int = 10) -> Dict:
        """Analyze health trend over recent observations."""
        if len(self.health_history) < 2:
            return {'trend': 'unknown', 'recent_health': 1.0}
        
        recent = self.health_history[-window:]
        
        # Linear trend
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        trend = 'improving' if slope > 0.01 else ('degrading' if slope < -0.01 else 'stable')
        
        return {
            'trend': trend,
            'slope': float(slope),
            'recent_health': float(np.mean(recent)),
            'health_std': float(np.std(recent))
        }
    
    def should_intervene(self) -> Tuple[bool, str]:
        """
        Decide if Observer should take action.
        
        Returns:
            (should_act, reason)
        """
        if self.readonly:
            return False, "Observer in read-only mode"
        
        # Check health
        if self.health_score < 0.5:
            return True, "Structural health critical"
        
        # Check trend
        trend = self.get_health_trend()
        if trend['trend'] == 'degrading' and trend['recent_health'] < 0.7:
            return True, "Health degrading"
        
        # Check pending decisions
        high_priority = sum(1 for d in self.decisions[-10:] if d.priority > 0.7)
        if high_priority >= 3:
            return True, "Multiple high-priority issues"
        
        return False, "System healthy"
    
    def export_log(self, filepath: str):
        """Export complete observation log to file."""
        log_data = {
            'observation_count': self.observation_count,
            'health_history': self.health_history,
            'decisions': [self._decision_to_dict(d) for d in self.decisions],
            'decision_summary': self.get_decision_summary(),
            'health_trend': self.get_health_trend(),
            'readonly_mode': self.readonly
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _decision_to_dict(self, decision: StructuralDecision) -> Dict:
        """Convert decision to dictionary."""
        return {
            'type': decision.decision_type,
            'layer': decision.layer_id,
            'neuron': decision.neuron_id,
            'reason': decision.reason,
            'utility': decision.utility,
            'priority': decision.priority,
            'timestamp': decision.timestamp
        }


if __name__ == "__main__":
    print("SSNC PHASE 5: The Observer")
    print("="*60)
    print()
    
    # Create observer
    observer = SSNCObserver(readonly=True)
    print(f"Observer created in {'READ-ONLY' if observer.readonly else 'ACTIVE'} mode")
    print(f"Deletion threshold: {observer.deletion_threshold}")
    print(f"Inactivity threshold: {observer.inactivity_threshold}")
    print(f"Overload threshold: {observer.overload_threshold}")
    print()
    
    # Simulate model state
    mock_state = {
        'global': {
            'total_neurons': 100,
            'inactive_neurons': 5,
            'mean_utility': 0.25
        },
        'layers': [
            {
                'layer_id': 0,
                'n_neurons': 50,
                'utilities': {
                    'mean': 0.30,
                    'std': 0.10,
                    'min': 0.05,
                    'max': 0.60,
                    'distribution': [0.3 + 0.1 * np.random.randn() for _ in range(50)]
                }
            },
            {
                'layer_id': 1,
                'n_neurons': 50,
                'utilities': {
                    'mean': 0.20,
                    'std': 0.08,
                    'min': 0.02,
                    'max': 0.45,
                    'distribution': [0.2 + 0.08 * np.random.randn() for _ in range(50)]
                }
            }
        ],
        'performance': {
            'loss': 0.5,
            'accuracy': 0.95
        }
    }
    
    # Run observation
    print("Running observation cycle...")
    report = observer.observe(mock_state)
    
    print(f"\nObservation #{report['observation_id']}")
    print("-" * 60)
    print(f"Structural health: {report['health']['overall_score']:.3f}")
    print(f"  Utility diversity: {report['health']['components']['utility_diversity']:.3f}")
    print(f"  Activity health: {report['health']['components']['activity_health']:.3f}")
    print(f"  Layer balance: {report['health']['components']['layer_balance']:.3f}")
    print()
    
    print(f"Issues identified: {len(report['issues'])}")
    for issue in report['issues'][:5]:  # Show first 5
        print(f"  [{issue['type']}] Layer {issue['layer']}, "
              f"Severity={issue['severity']:.2f}")
    print()
    
    print(f"Decisions generated: {len(report['decisions'])}")
    for decision in report['decisions'][:5]:  # Show first 5
        print(f"  [{decision['type']}] {decision['reason']}")
    print()
    
    # Decision summary
    summary = observer.get_decision_summary()
    print("Decision Summary:")
    print(f"  Total: {summary['total_decisions']}")
    print(f"  By priority: High={summary['by_priority']['high']}, "
          f"Med={summary['by_priority']['medium']}, Low={summary['by_priority']['low']}")
    print()
    
    # Intervention check
    should_act, reason = observer.should_intervene()
    print(f"Should intervene? {should_act}")
    print(f"Reason: {reason}")
    
    print("\n✅ PHASE 5 Observer implementation complete!")
    print("Next: Integrate with SSNC-MLP and train with observation")
