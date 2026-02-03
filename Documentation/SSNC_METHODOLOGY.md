# Self-Stabilizing Neuronal Control (SSNC): Methodology & Functioning

## 1. Introduction & Philosophy

**Self-Stabilizing Neuronal Control (SSNC)** is a biologically-inspired neural network architecture designed to overcome the rigidity of traditional fixed-topology deep learning models. In biological brains, neurons are not static; synaptic pruning and neurogenesis occur continuously to optimize energy efficiency and learning capability. SSNC mimics this by allowing the network to **self-structure** during training.

The core philosophy of SSNC is:
> *"A network should only contain the neurons it needs to solve the task, and should grow only when it lacks the capacity to learn."*

This is achieved through a **Dual-Process System**:
1.  **The Body (SSNC_MLP)**: A dynamic Multi-Layer Perceptron that can add, remove, and gate neurons.
2.  **The Brain (Adaptive Observer)**: A meta-controller that analyzes network statistics (utility, loss, accuracy) to make principled decisions about structural changes.

---

## 2. The SSNC Lifecycle

Every neuron in an SSNC network goes through a dynamic lifecycle, tracked by the system:

1.  **Creation (Birth)**:
    *   **Initial**: Created at initialization.
    *   **Neurogenesis**: Created by duplicating a high-utility "parent" neuron when the network needs more capacity.
    *   **Layer Addition**: A completely new layer can be inserted if the network is too shallow.

2.  **Probation (Burn-in/Grace Period)**:
    *   New neurons are protected from deletion for a set number of cycles (e.g., `grace_period = 50`).
    *   The entire network has a `burn_in_epochs` phase where *no* structural changes occur, allowing initial weights to stabilize.

3.  **Active Phase**:
    *   The neuron participates in forward/backward passes.
    *   Its **Utility** is continuously updated.
    *   **Utility Metric**: primarily based on **Average Activation** (how often and how strongly it fires). A neuron that never fires is considered "dead" or "useless".

4.  **Marked for Deletion**:
    *   If `Utility < deletion_threshold` for `marking_cycles` consecutively, the neuron is "Marked".
    *   This is a warning state. If utility rises, the mark is removed.

5.  **Soft Deletion (Gating)**:
    *   If a neuron remains marked, it enters **Soft Deletion**.
    *   The neuron is **Gated**: its output is multiplied by a scalar mask that ramps down from 1.0 to 0.0.
    *   Once the gate hits 0.0, the neuron is effectively "off" but still exists in memory. This allows for **hysteresis**—it can be quickly revived if needed without data loss.

6.  **Hard Deletion (Pruning)**:
    *   If a neuron remains soft-deleted for `hard_deletion_cycles` (e.g., 5 cycles), it is **Permanently Removed**.
    *   The weight matrices of the layer are physically resized (columns/rows removed).
    *   The indices of subsequent neurons are shifted.

---

## 3. The Plasticity Loop (Functioning)

SSNC operates on a periodic cycle (e.g., every 50 batches). This guarantees stability and prevents chaos.

### Step 1: Observation
During standard training batches, neurons track their **Utility**:
$$ U_i \leftarrow \alpha U_i + (1-\alpha) \cdot \text{Activation}_i $$
Where $\alpha$ is a momentum term (e.g., 0.95).

### Step 2: The Observer's Analysis
At the end of a cycle, the **Adaptive Observer** takes a snapshot of the system:
*   **Utility Distribution**: Are most neurons useful? Are many useless?
*   **Pressure**:
    *   *Deletion Pressure*: % of neurons with near-zero utility.
    *   *Creation Pressure*: % of neurons with maxed-out utility (overloaded).
*   **Trend**: Is the loss decreasing, stagnating, or exploding?

### Step 3: Principled Decision Making
The Observer classifies the network into a **Regime** and sets the policy:

| Regime | Condition | Policy |
| :--- | :--- | :--- |
| **Converging** | High Acc, Low Variance | **Stop Plasticity**. Lock the structure. |
| **Stable** | Improving Loss | **Balanced**. Allow moderate pruning and creation. |
| **Underfitting** | High Loss, Stagnant | **Growth**. Boost creation, lower thresholds. |
| **Unstable** | Spiking Loss | **Freeze**. Stop all changes to let weights settle. |
| **Bloated** | Layer size > limit | **Split/Prune**. Force reduction or add layer. |

### Step 4: Execution
Based on the policy, the system executes:
1.  **Update Thresholds**: The `deletion_threshold` might auto-adjust (e.g., set to the bottom 10th percentile of utility).
2.  **Soft Delete**: Gate the weakest neurons.
3.  **Hard Delete**: Physically remove neurons that have failed probation.
4.  **Create**: Duplicate the strongest neurons (Activity-Driven Growth) to share the load.

---

## 4. Key Mechanisms

### Adaptive Observer
The "Brain" (`src/observer.py`) is what makes SSNC "Self-Stabilizing". Instead of fixed rules (e.g., "prune 10% every epoch"), it uses:
*   **Safety Brakes**: If `Active Neurons < Min Capacity`, force creation. If `Loss` spikes, block deletion.
*   **Circuit Breakers**: If deletion happens too fast (>10% drop), panic stop to prevent collapse.
*   **Reasoning Log**: Every decision produces a human-readable log (e.g., *"STAGNATION: Forcing growth +5 neurons"*).

### Vectorized Gating
To ensure speed, Soft Deletion does not use `if` statements in the forward pass. It uses a **Gate Mask**:
*   `mask = [1.0, 1.0, 0.0, 1.0, ...]`
*   `output = activation(input) * mask`
*   This allows "deleting" a neuron without changing tensor shapes, which is computationally cheap and reversible.

### Neuron Duplication (Mitosis)
When creating a neuron, SSNC doesn't initialize it randomly. It uses **Mitosis**:
1.  Select a high-utility "parent" neuron.
2.  Copy its incoming and outgoing weights to a new slot.
3.  Add slight noise to break symmetry.
4.  (Optional) Halve the outgoing weights of both parent and child so the downstream sum remains constant.
This preserves the "knowledge" of the parent while doubling the processing capacity for that feature.

---

## 5. Technical Implementation Details

*   **File Structure**:
    *   `ssnc_mlp.py`: Contains `CompleteSSNC_MLP` (The Body). Handles the complex tensor manipulations for resizing layers.
    *   `observer.py`: Contains `AdaptiveObserver` (The Brain). Pure logic, no PyTorch dependencies.
    *   `trainer.py`: The loop that synchronizes the two.

*   **Hyperparameters**:
    *   `plasticity_cycle_every`: Batches between structural checks (e.g., 50).
    *   `hard_deletion_cycles`: How long a soft-deleted neuron waits before death (e.g., 5 cycles).
    *   `creation_threshold`: Utility level required to spawn a child (e.g., 0.25).

## 6. Summary of Benefits

1.  **Parameter Efficiency**: Automatically finds a smaller sub-network (often 30-50% reduction) that solves the task.
2.  **Dynamic Capacity**: Can grow if the task is harder than expected, avoiding "too small" architecture bottlenecks.
3.  **Explainability**: You can see *which* layers grew or shrank, indicating where the problem complexity lies.
4.  **Stability**: Unlike standard pruning which can destroy accuracy, SSNC's "Soft Delete -> Wait -> Hard Delete" loop ensures only truly useless neurons are removed.
