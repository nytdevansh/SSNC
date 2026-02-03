# Quick Start Guide

## Prerequisites

Ensure you have Python 3.8+ and PyTorch installed.

```bash
pip install torch numpy matplotlib tqdm
```

## Running the Experiment

The main entry point is `main.py`. This script will:
1.  Download the TREC-6 dataset.
2.  Initialize the SSNC model (~576 neurons).
3.  Train for 50 epochs.
4.  Dynamically prune and grow neurons based on utility.

```bash
python main.py
```

## Understanding the Output

As training progresses, you will see logs from the **Adaptive Observer**:

*   `🧠 Observer: PRUNE...`: The observer has detected useless neurons.
*   `🧠 Observer: STAGNATION...`: The network is not learning, triggering growth.
*   `⚡ Hard-deleted X neurons`: Permanent removal of neurons.
*   `✨ Created X neurons`: New neurons added via neurogenesis.

## Results

After training, look at `results/trec6/trec6_stable_results.png`. You should see:
*   **Accuracy** rising to ~88-92%.
*   **Active Neurons** decreasing (typically by 15-25%) as the network prunes redundancy.
