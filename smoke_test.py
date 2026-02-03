import sys
import os

# Ensure we can import src
sys.path.append(os.getcwd())

# Mutate config BEFORE importing trainer to ensure it picks up changes
# (Or if trainer imports CONFIG object, mutating it works)
from src.trec6_model import CONFIG
CONFIG['epochs'] = 1
CONFIG['plasticity_cycle_every'] = 10 # Frequent checks to force execution
CONFIG['utility_update_every'] = 1

print("🔥 STARTING SMOKE TEST (1 Epoch)...")

from src.trainer import train

if __name__ == "__main__":
    try:
        train()
        print("\n✅ SMOKE TEST PASSED")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
