# Self-Stabilizing Neuronal Control (SSNC) on TREC-6

## Overview

SSNC is a biologically-inspired neural network architecture that self-organizes during training. Unlike standard networks with fixed architectures, SSNC adapts its structure dynamically:

*   **Self-Pruning**: Neurons with low utility are gradually "soft deleted" (gated) and eventually "hard deleted" (removed).
*   **Neurogenesis**: When the network is overloaded, new neurons are created to add capacity.
*   **Stability**: An Adaptive Observer monitors the network to ensure changes are principled and safe.

This project demonstrates SSNC on the **TREC-6** Question Classification dataset.

## Project Structure

The project is modularized for clarity:

```text
/
├── main.py                 # Entry point to run training
├── train.py                # (Legacy/Alias)
├── data/                   # Dataset storage
├── results/                # Logs and visualization outputs
└── src/
    ├── ssnc_mlp.py         # Core SSNC network implementation
    ├── observer.py         # Adaptive Observer logic (The "Brain")
    ├── metrics.py          # Neuron state tracking
    ├── trec6_model.py      # TREC-6 specific model wrapper
    ├── trec6_loader.py     # Data loading utilities
    ├── trainer.py          # Training loop
    └── utils.py            # Helpers
```

## Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install torch numpy matplotlib tqdm
    ```

2.  **Run Training**:
    ```bash
    python main.py
    ```

3.  **View Results**:
    Check the `results/trec6/` directory for:
    *   `trec6_stable_results.png`: Plots of accuracy, loss, and neuron counts.
    *   `network_architecture.png`: Visualization of the final network structure.
    *   `trec6_stable_history.json`: Raw training logs.

## Documentation

For a deep dive into how SSNC works, read [ARCHITECTURE.md](Documentation/ARCHITECTURE.md).
