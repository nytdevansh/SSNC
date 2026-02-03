"""
Autonomous SSNC Controller

A self-learning system that discovers optimal plasticity rules through experience,
without predefined thresholds. Uses:
1. Statistical Anomaly Detection (z-score based)
2. Multi-Armed Bandit (UCB for action selection)
3. Experience-based causal learning
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math
import random


class Action(Enum):
    """Possible structural actions."""
    DO_NOTHING = "do_nothing"
    DELETE_WEAK = "delete_weak"
    DELETE_REDUNDANT = "delete_redundant"
    CREATE_AT_BOTTLENECK = "create_at_bottleneck"
    DUPLICATE_STRONG = "duplicate_strong"


@dataclass
class NetworkState:
    """Captures the current state of the network for decision making."""
    epoch: int
    batch_idx: int
    train_accuracy: float
    train_loss: float
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    
    # Utility statistics
    utility_mean: float = 0.0
    utility_std: float = 0.0
    utility_min: float = 0.0
    utility_max: float = 0.0
    
    # Structural info
    total_neurons: int = 0
    neurons_per_layer: List[int] = field(default_factory=list)
    
    # Gradient info
    gradient_mean: float = 0.0
    gradient_std: float = 0.0
    
    # Per-class performance (optional)
    class_accuracies: Dict[int, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for similarity comparison."""
        return np.array([
            self.train_accuracy,
            self.train_loss,
            self.utility_mean,
            self.utility_std,
            self.gradient_mean,
            self.total_neurons / 1000.0,  # Normalize
        ])


@dataclass
class Experience:
    """A single experience tuple (state, action, reward)."""
    state: NetworkState
    action: Action
    reward: float  # Performance change after action
    timestamp: int  # Global step


class AnomalyDetector:
    """
    Detects statistical anomalies without fixed thresholds.
    Uses z-scores and adaptive windows.
    """
    
    def __init__(self, window_size: int = 100, sigma_threshold: float = 2.0):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        
        # Rolling windows for different metrics
        self.utility_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)
        
        # Per-neuron utility histories
        self.neuron_histories: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
    
    def update(self, state: NetworkState, utilities: Dict[Tuple[int, int], float]):
        """Update history with new observations."""
        self.loss_history.append(state.train_loss)
        self.accuracy_history.append(state.train_accuracy)
        self.utility_history.append(state.utility_mean)
        self.gradient_history.append(state.gradient_mean)
        
        for key, utility in utilities.items():
            self.neuron_histories[key].append(utility)
    
    def compute_z_score(self, value: float, history: deque) -> float:
        """Compute z-score relative to history."""
        if len(history) < 5:
            return 0.0
        
        mean = np.mean(history)
        std = np.std(history) + 1e-8
        return (value - mean) / std
    
    def is_anomaly(self, value: float, history: deque) -> bool:
        """Check if value is anomalously far from history."""
        z = self.compute_z_score(value, history)
        return abs(z) > self.sigma_threshold
    
    def get_utility_anomalies(self, utilities: Dict[Tuple[int, int], float]) -> Dict[str, List]:
        """
        Find neurons with anomalously low or high utility.
        
        Returns dict with:
        - 'anomaly_low': neurons with unusually low utility
        - 'anomaly_high': neurons with unusually high utility
        """
        anomaly_low = []
        anomaly_high = []
        
        for key, utility in utilities.items():
            history = self.neuron_histories[key]
            if len(history) < 10:
                continue
            
            z = self.compute_z_score(utility, history)
            
            if z < -self.sigma_threshold:
                anomaly_low.append((key, z, utility))
            elif z > self.sigma_threshold:
                anomaly_high.append((key, z, utility))
        
        # Sort by severity
        anomaly_low.sort(key=lambda x: x[1])  # Most negative first
        anomaly_high.sort(key=lambda x: x[1], reverse=True)  # Most positive first
        
        return {'anomaly_low': anomaly_low, 'anomaly_high': anomaly_high}
    
    def is_loss_unstable(self) -> bool:
        """Check if loss is oscillating (unstable)."""
        if len(self.loss_history) < 20:
            return False
        
        recent = list(self.loss_history)[-20:]
        diffs = np.diff(recent)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        # More than 10 sign changes in 20 samples = oscillating
        return sign_changes > 10
    
    def is_plateaued(self) -> bool:
        """Check if accuracy has stagnated."""
        if len(self.accuracy_history) < 30:
            return False
        
        recent = list(self.accuracy_history)[-30:]
        std = np.std(recent)
        
        # Very low variance = plateau
        return std < 0.5  # Less than 0.5% change
    
    def get_diagnostic(self) -> str:
        """Return current network diagnostic."""
        if len(self.accuracy_history) < 10:
            return "WARMUP"
        
        if self.is_loss_unstable():
            return "UNSTABLE"
        
        if self.is_plateaued():
            return "PLATEAU"
        
        # Check recent trend
        recent_acc = list(self.accuracy_history)[-10:]
        trend = np.polyfit(range(len(recent_acc)), recent_acc, 1)[0]
        
        if trend > 0.5:
            return "IMPROVING"
        elif trend < -0.5:
            return "DEGRADING"
        else:
            return "STABLE"


class ExperienceBuffer:
    """
    Stores experience tuples for learning from past decisions.
    Enables causal inference: "When I did X, what happened?"
    """
    
    def __init__(self, max_size: int = 1000):
        self.buffer: deque = deque(maxlen=max_size)
        self.action_stats: Dict[Action, List[float]] = defaultdict(list)
    
    def add(self, state: NetworkState, action: Action, reward: float, step: int):
        """Add new experience."""
        exp = Experience(state=state, action=action, reward=reward, timestamp=step)
        self.buffer.append(exp)
        self.action_stats[action].append(reward)
    
    def get_action_statistics(self) -> Dict[Action, Dict[str, float]]:
        """Get mean and std reward for each action."""
        stats = {}
        for action in Action:
            rewards = self.action_stats[action]
            if rewards:
                stats[action] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'count': len(rewards)
                }
            else:
                stats[action] = {'mean': 0.0, 'std': 1.0, 'count': 0}
        return stats
    
    def find_similar_experiences(self, current_state: NetworkState, k: int = 10) -> List[Experience]:
        """Find k most similar past experiences."""
        if len(self.buffer) < k:
            return list(self.buffer)
        
        current_vec = current_state.to_vector()
        
        similarities = []
        for exp in self.buffer:
            exp_vec = exp.state.to_vector()
            # Cosine similarity
            sim = np.dot(current_vec, exp_vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(exp_vec) + 1e-8
            )
            similarities.append((exp, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, _ in similarities[:k]]
    
    def predict_action_outcome(self, current_state: NetworkState, action: Action) -> float:
        """Predict expected reward for action based on similar experiences."""
        similar = self.find_similar_experiences(current_state, k=20)
        
        # Filter for same action
        same_action = [exp for exp in similar if exp.action == action]
        
        if not same_action:
            # Use global statistics
            stats = self.get_action_statistics()
            return stats[action]['mean']
        
        return np.mean([exp.reward for exp in same_action])


class ActionBandit:
    """
    Multi-Armed Bandit for action selection using UCB.
    Balances exploration vs exploitation.
    """
    
    def __init__(self, exploration_constant: float = 2.0):
        self.c = exploration_constant
        self.action_counts: Dict[Action, int] = defaultdict(int)
        self.action_rewards: Dict[Action, float] = defaultdict(float)
        self.total_steps = 0
    
    def select_action(self, valid_actions: List[Action]) -> Action:
        """Select action using UCB (Upper Confidence Bound)."""
        self.total_steps += 1
        
        # If any action hasn't been tried, try it
        untried = [a for a in valid_actions if self.action_counts[a] == 0]
        if untried:
            return random.choice(untried)
        
        # Compute UCB score for each action
        ucb_scores = {}
        for action in valid_actions:
            n = self.action_counts[action]
            mean_reward = self.action_rewards[action] / n
            exploration_bonus = self.c * math.sqrt(math.log(self.total_steps) / n)
            ucb_scores[action] = mean_reward + exploration_bonus
        
        # Select action with highest UCB
        return max(ucb_scores.keys(), key=lambda a: ucb_scores[a])
    
    def update(self, action: Action, reward: float):
        """Update statistics after observing outcome."""
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
    
    def get_exploitation_ranking(self) -> List[Tuple[Action, float]]:
        """Get actions ranked by mean reward (pure exploitation)."""
        rankings = []
        for action in Action:
            n = self.action_counts[action]
            if n > 0:
                mean = self.action_rewards[action] / n
                rankings.append((action, mean))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class AutonomousSSNCController:
    """
    Main controller for autonomous SSNC decisions.
    No predefined thresholds - learns from experience.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        sigma_threshold: float = 2.0,
        exploration_constant: float = 2.0,
        min_warmup_steps: int = 50
    ):
        self.anomaly_detector = AnomalyDetector(window_size, sigma_threshold)
        self.experience_buffer = ExperienceBuffer()
        self.action_bandit = ActionBandit(exploration_constant)
        
        self.min_warmup_steps = min_warmup_steps
        self.global_step = 0
        
        # Track pending action for reward computation
        self.pending_action: Optional[Action] = None
        self.pending_state: Optional[NetworkState] = None
        self.pre_action_accuracy: float = 0.0
    
    def observe(self, state: NetworkState, utilities: Dict[Tuple[int, int], float]):
        """
        Observe current network state.
        Call this every plasticity cycle.
        """
        self.global_step += 1
        self.anomaly_detector.update(state, utilities)
        
        # If there's a pending action, compute its reward
        if self.pending_action is not None:
            reward = state.train_accuracy - self.pre_action_accuracy
            self.experience_buffer.add(
                self.pending_state, self.pending_action, reward, self.global_step
            )
            self.action_bandit.update(self.pending_action, reward)
            self.pending_action = None
    
    def decide(self, state: NetworkState, utilities: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """
        Make autonomous decision about structural changes.
        
        Returns dict with:
        - 'action': Selected Action enum
        - 'targets': List of (layer_idx, neuron_idx) to act on
        - 'reasoning': Human-readable explanation
        """
        # Warmup: don't act yet, just observe
        if self.global_step < self.min_warmup_steps:
            return {
                'action': Action.DO_NOTHING,
                'targets': [],
                'reasoning': f"Warmup phase ({self.global_step}/{self.min_warmup_steps})"
            }
        
        # Get diagnostic
        diagnostic = self.anomaly_detector.get_diagnostic()
        
        # Get utility anomalies
        anomalies = self.anomaly_detector.get_utility_anomalies(utilities)
        
        # Determine valid actions based on context
        valid_actions = self._get_valid_actions(diagnostic, anomalies)
        
        # Use bandit to select action
        selected_action = self.action_bandit.select_action(valid_actions)
        
        # Get targets for the action (pass utilities for fallback targeting)
        targets, reasoning = self._get_action_targets(selected_action, anomalies, state, utilities)
        
        # Store for reward computation
        if selected_action != Action.DO_NOTHING:
            self.pending_action = selected_action
            self.pending_state = state
            self.pre_action_accuracy = state.train_accuracy
        
        return {
            'action': selected_action,
            'targets': targets,
            'reasoning': f"[{diagnostic}] {reasoning}",
            'diagnostic': diagnostic
        }
    
    def _get_valid_actions(self, diagnostic: str, anomalies: Dict) -> List[Action]:
        """Determine which actions are valid given current context."""
        # ALWAYS allow all actions for exploration (except when unstable)
        if diagnostic == "UNSTABLE":
            return [Action.DO_NOTHING]
        
        # Proactive: always allow structural actions
        valid = [
            Action.DO_NOTHING,
            Action.DELETE_WEAK,
            Action.DELETE_REDUNDANT,
            Action.CREATE_AT_BOTTLENECK,
            Action.DUPLICATE_STRONG
        ]
        
        return valid
    
    def _get_action_targets(
        self, action: Action, anomalies: Dict, state: NetworkState, utilities: Dict = None
    ) -> Tuple[List[Tuple[int, int]], str]:
        """Get specific neuron targets for an action."""
        
        if action == Action.DO_NOTHING:
            return [], "No action needed"
        
        elif action == Action.DELETE_WEAK:
            # Use anomalies if available, else use bottom percentile
            if anomalies['anomaly_low']:
                candidates = anomalies['anomaly_low'][:2]
                targets = [key for key, z, util in candidates]
            elif utilities:
                # Fallback: target bottom 1% by raw utility
                sorted_utils = sorted(utilities.items(), key=lambda x: x[1])
                n_target = max(1, len(sorted_utils) // 100)
                targets = [key for key, _ in sorted_utils[:n_target]]
            else:
                targets = []
            return targets, f"Deleting {len(targets)} weak neurons"
        
        elif action == Action.DELETE_REDUNDANT:
            return [], "Signal to prune redundant neurons"
        
        elif action == Action.CREATE_AT_BOTTLENECK:
            # Use anomalies if available, else use top percentile
            if anomalies['anomaly_high']:
                candidates = anomalies['anomaly_high'][:1]
                targets = [key for key, z, util in candidates]
            elif utilities:
                # Fallback: target top 1% by raw utility
                sorted_utils = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
                n_target = max(1, len(sorted_utils) // 100)
                targets = [key for key, _ in sorted_utils[:n_target]]
            else:
                targets = []
            return targets, f"Creating at {len(targets)} bottleneck neurons"
        
        elif action == Action.DUPLICATE_STRONG:
            if anomalies['anomaly_high']:
                candidates = anomalies['anomaly_high'][:1]
                targets = [key for key, z, util in candidates]
            elif utilities:
                sorted_utils = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
                n_target = max(1, len(sorted_utils) // 100)
                targets = [key for key, _ in sorted_utils[:n_target]]
            else:
                targets = []
            return targets, f"Duplicating {len(targets)} high-utility neurons"
        
        return [], "Unknown action"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics for logging."""
        action_stats = self.experience_buffer.get_action_statistics()
        exploitation_ranking = self.action_bandit.get_exploitation_ranking()
        
        return {
            'global_step': self.global_step,
            'total_experiences': len(self.experience_buffer.buffer),
            'action_statistics': {
                a.value: stats for a, stats in action_stats.items()
            },
            'best_action': exploitation_ranking[0] if exploitation_ranking else None,
            'diagnostic': self.anomaly_detector.get_diagnostic()
        }
