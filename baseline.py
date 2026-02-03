"""
Baseline Model Training.

Trains the TREC-6 model with all SSNC features disabled (Fixed Topology).
"""

from src.trec6_model import CONFIG
from src.trainer import train

if __name__ == "__main__":
    # Create baseline config by disabling all SSNC features
    baseline_config = CONFIG.copy()
    
    baseline_config.update({
        'enable_soft_deletion': False,
        'enable_hard_deletion': False,
        'enable_creation': False,
        'enable_redistribution': False,
        'enable_monitoring': False,
        
        # Disable burn-in just in case
        'burn_in_epochs': 0
    })
    
    print("STARTING BASELINE TRAINING (FIXED TOPOLOGY)")
    print("All SSNC features have been disabled.")
    
    train(baseline_config)
