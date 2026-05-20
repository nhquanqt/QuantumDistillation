# PennyLane VQAs on 8x8 MNIST-like Digits

This project trains variational quantum classifiers on the `sklearn` handwritten digits dataset, which is the standard 8x8 MNIST-like benchmark.

The implementation uses:

- `PennyLane` for quantum circuits
- `PyTorch` for the training loop, parameters, and optimizer
- `scikit-learn` for the 8x8 digits dataset and dataset splitting
- `numpy` for dataset preprocessing

## Project layout

```text
code/
├── pyproject.toml
├── README.md
└── src/
    └── vqa_mnist/
        ├── __init__.py
        ├── compare.py
        ├── dataset.py
        ├── model.py
        └── train.py
```

## Setup

Create an environment and install the package in editable mode:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train one VQA

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --epochs 12 --train-limit 512
```

Useful options:

- `--ansatz`: `basic`, `hardware_efficient`, or `strongly_entangling`
- `--epochs`: training epochs
- `--train-limit`: cap the training set for faster experiments
- `--layers`: number of variational layers
- `--learning-rate`: Adam step size

## Compare several VQAs

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
compare-vqa-mnist --epochs 8 --train-limit 384
```

This runs all supported ansatzes and writes a summary JSON file under `code/outputs/`.

## Notes

- Images are flattened from 8x8 into 64 amplitudes and loaded with amplitude embedding on 6 qubits.
- The quantum circuit produces a 64-dimensional probability vector, followed by a Torch linear readout head for 10-way classification.
- This is meant to be a compact research scaffold that you can extend with new ansatzes, losses, or distillation experiments.
