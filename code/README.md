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
        ├── compare_layers.py
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
train-vqa-mnist --ansatz strongly_entangling --epochs 12
```

Useful options:

- `--ansatz`: `basic`, `hardware_efficient`, or `strongly_entangling`
- `--readout-mode`: `linear` or `probs_only`
- `--num-classes`: `10` or `4`
- `--epochs`: training epochs
- `--train-limit`: optionally cap the training set for faster experiments
- `--val-limit`: optionally cap the validation set
- `--test-limit`: optionally cap the test set
- `--layers`: number of variational layers
- `--learning-rate`: Adam step size
- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--quantum-device`: `cpu` or `cuda`

By default, the project trains, validates, and tests on the full dataset splits.

Each training epoch prints its computing time and also stores it in the output JSON as `epoch_time_seconds`.

If you want to classify with `qml.probs` only and no trainable classical head:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --readout-mode probs_only
```

In `probs_only` mode, the circuit still returns the full 64-dimensional probability vector. The classifier then groups basis-state probabilities into 10 class probabilities using `basis_index mod 10`, and trains directly on those class probabilities without a learnable linear readout layer.

If you want to train on 4 classes only:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --num-classes 4
```

This filters the dataset to digits `0`, `1`, `2`, and `3`, and changes the classifier output from 10 classes to 4 classes.

## Run with CUDA

If your environment supports it, you can enable CUDA for both the PennyLane quantum simulator and the PyTorch readout path:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --epochs 12 --quantum-device cuda --device cuda
```

You can also compare all ansatzes with:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
compare-vqa-mnist --epochs 8 --quantum-device cuda --device cuda
```

Device meaning:

- `--quantum-device cuda` selects PennyLane's `lightning.gpu` simulator backend.
- `--device cuda` places the PyTorch readout layer and loss computation on CUDA.
- `--quantum-device cpu` keeps the quantum simulator on `default.qubit`.
- If the CUDA-backed PennyLane simulator is unavailable, the script raises a clear error.

## Ansatz design

This project uses a hybrid quantum classifier with the following structure:

1. Flatten each 8x8 image into a 64-dimensional vector.
2. Normalize the vector and amplitude-embed it into a 6-qubit quantum state.
3. Apply a trainable variational ansatz.
4. Measure the full 64-dimensional computational-basis probability vector.
5. Feed that vector into a small PyTorch linear layer for 10-class prediction.

The key design choice is that `2^6 = 64`, so 6 qubits are enough to represent the full 8x8 image after flattening.

### Supported ansatzes

#### `basic`

Each layer applies:

- `RY` and `RZ` on every qubit
- a ring of `CNOT` gates connecting qubit `i` to qubit `(i + 1) mod n`

This is the simplest ansatz in the project. It uses 2 trainable parameters per qubit per layer and is a good baseline when you want a smaller, easier-to-interpret circuit.

#### `hardware_efficient`

Each layer applies:

- `RX`, `RY`, and `RZ` on every qubit
- nearest-neighbor `CZ` gates along the qubit line

This is a more expressive layered circuit with 3 trainable parameters per qubit per layer. It is meant to be a stronger generic ansatz while still keeping a simple, hardware-friendly structure.

#### `strongly_entangling`

This uses PennyLane's built-in `StronglyEntanglingLayers` template.

It is the richest ansatz in the project and serves as the most expressive default option. If you want a stronger baseline without hand-designing the entangling pattern yourself, this is usually the best starting point.

### Why the model measures probabilities instead of a few observables

The quantum circuit returns the full probability vector over all 6-qubit basis states instead of only measuring one expectation value per class.

That means the ansatz is being used mainly as a quantum feature transformer. The final class decision is made by the PyTorch readout head, which maps the 64 quantum probabilities to 10 digit classes.

This design makes it easy to compare different ansatz families under the same encoding and readout setup.

## Readout design

The readout is the final classical layer that converts the quantum circuit output into digit-class logits.

For 6 qubits, the circuit returns a 64-dimensional probability vector, one probability for each computational-basis state. This vector is treated as a learned quantum feature representation rather than a final prediction.

The PyTorch readout layer is a linear map from 64 to 10:

$$
z = Wp + b,
$$

where:

- $p$ is the 64-dimensional probability vector from the quantum circuit
- $W$ is a trainable `10 x 64` weight matrix
- $b$ is a trainable 10-dimensional bias
- $z$ is the 10-dimensional logit vector

These logits are passed directly to cross-entropy loss during training. In other words, the model does not manually apply softmax in the code. The classical loss handles the final probability normalization internally.

This design keeps the quantum circuit focused on feature transformation while the classical readout performs the final 10-class decision. It is a simple hybrid architecture that makes ansatz comparisons easier and more stable.

An alternative is `--readout-mode probs_only`, which removes the trainable classical head and uses only `qml.probs` to produce class probabilities. In that mode, the basis-state probabilities are grouped into either 10 or 4 class probabilities depending on `--num-classes`.

## Compare several VQAs

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
compare-vqa-mnist --epochs 8
```

This runs all supported ansatzes and writes a summary JSON file under `code/outputs/`.

## Compare different layer counts

To compare one ansatz across multiple layer depths:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
compare-vqa-layers --ansatz strongly_entangling --epochs 8 --layer-values 1 2 3 4
```

This runs the selected ansatz once for each layer count and writes a summary JSON file under `code/outputs/`.

## Notes

- Images are flattened from 8x8 into 64 amplitudes and loaded with amplitude embedding on 6 qubits.
- The quantum circuit produces a 64-dimensional probability vector, followed by a Torch linear readout head for 10-way classification.
- This is meant to be a compact research scaffold that you can extend with new ansatzes, losses, or distillation experiments.
