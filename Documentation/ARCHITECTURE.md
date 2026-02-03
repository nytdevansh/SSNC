# SSNC Architecture

## Core Concepts

Self-Stabilizing Neuronal Control (SSNC) is built on four main pillars:

1.  **Neuron Local State (`metrics.py`)**:
    Each neuron tracks its own "health" validation stats:
    *   `utility`: A score (0-1) representing usefulness (activation magnitude, variance, error correlation).
    *   `age`: How long the neuron has existed.
    *   `marked_for_deletion`: A flag indicating low utility.

2.  **Adaptive Observer (`observer.py`)**:
    A global module that analyzes the aggregate state of the network. It diagnoses the "Regime" (e.g., Converging, Unstable, Underfitting) and issues commands:
    *   `should_delete`: If redundancy is high.
    *   `should_create`: If the network is struggling (stagnation).
    *   `should_add_layer`: If the network lacks depth complexity.

3.  **SSNC MLP (`ssnc_mlp.py`)**:
    The neural network container. It executes the observer's commands through distinct cycles:
    *   **Soft Deletion**: Low-utility neurons are "gated" (multiplied by a decreasing factor). This is reversible.
    *   **Hard Deletion**: Neurons that remain soft-deleted for too long are permanently removed to reclaim resources.
    *   **Neurogenesis**: High-utility neurons are duplicated to share the load when the network is under pressure.

4.  **Safety & Stability**:
    *   **Burn-in**: No changes in the first few epochs to let weights stabilize.
    *   **Grace Period**: New neurons are immune to deletion for a while.
    *   **Circuit Breakers**: The observer stops deletion if accuracy drops too fast.

## Plasticity Cycle

Every `plasticity_cycle_every` batches (e.g., 20), the following happens:

1.  **Update Utilities**: Neurons compute their current utility based on recent activations and gradients.
2.  **Analyze**: The Observer looks at the utility distribution and loss history.
3.  **Decide**: The Observer sets `max_deletions` and `max_creations` budgets.
4.  **Execute**:
    *   Mark neurons for soft deletion.
    *   Gate soft-deleted neurons.
    *   Hard-delete confirmed dead neurons.
    *   Create new neurons if authorized.

## Configuration

The system is configured via `CONFIG` in `src/trec6_model.py`. Key parameters:

*   `soft_deletion_threshold`: Utility below this (e.g., 0.05) triggers marking.
*   `creation_threshold`: Utility above this (e.g., 0.40) suggests overload.
*   `max_deletions_per_cycle`: Limits how fast the network can shrink.
