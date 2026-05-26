# PennyLane VQAs on 8x8 MNIST

This project trains variational quantum classifiers on the `torchvision` MNIST dataset. The original 28x28 images are resized to 8x8 so they can be amplitude-embedded into 6 qubits.

The implementation uses:

- `PennyLane` for quantum circuits
- `PyTorch` for the training loop, parameters, and optimizer
- `torchvision` for the MNIST dataset
- `numpy` for dataset preprocessing

## Outline

- [Project layout](#project-layout)
- [Setup](#setup)
- [Train one VQA](#train-one-vqa)
- [Run with CUDA](#run-with-cuda)
- [Ansatz design](#ansatz-design)
- [Experiment types](#experiment-types)
- [Readout design](#readout-design)
- [Learnable observable details](#learnable-observable-details)
- [Compare several VQAs](#compare-several-vqas)
- [Compare different layer counts](#compare-different-layer-counts)
- [Notes](#notes)

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

- `--ansatz`: `basic`, `hardware_efficient`, `qcnn`, or `strongly_entangling`
- `--readout-mode`: `linear`, `probs_only`, or `learnable_observable`
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

Each run also prints the number of train, validation, and test samples, and stores those counts in `results.json` under `dataset_sizes`.

The training loop also prints explicit `train_step` and validation step progress for each epoch, and stores the total step counts in `results.json` under `step_counts`.

Test evaluation is run with the best validation checkpoint rather than the final epoch checkpoint. The saved JSON includes `best_checkpoint` metadata, while the actual model weights are stored in the checkpoint files.

Output filenames include the main experiment arguments such as ansatz, readout mode, number of classes, layers, epochs, batch size, learning rate, seed, and device choices.

Each experiment is saved as its own folder. A training run folder includes:

- `results.json`
- `train.log`
- `best_model.pt`
- `final_model.pt`

Comparison commands also create a parent experiment folder with:

- a `summary_*.json` file
- a `compare.log` file
- a `runs/` directory containing one saved folder per individual run

If you want to classify with `qml.probs` only and no trainable classical head:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --readout-mode probs_only
```

In `probs_only` mode, the circuit still returns the full 64-dimensional probability vector. The classifier then groups basis-state probabilities into class probabilities using `basis_index mod num_classes`, and trains directly on those class probabilities without a learnable linear readout layer.

If you want a learnable observable based on [Learning to Program Quantum Measurements for Machine Learning](https://arxiv.org/pdf/2505.13525):

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --readout-mode learnable_observable
```

In `learnable_observable` mode, the circuit returns the quantum state and a small neural controller generates a Hermitian observable for each class on a per-input basis. The class logits are the expectation values of those input-conditioned observables, which follows the paper's idea of programmable quantum measurements.

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

More specifically, this ansatz is called “hardware efficient” because it uses:

- only standard single-qubit rotation gates
- only local nearest-neighbor entangling gates
- a repeated layered structure that is easy to map onto many quantum devices

For this project:

- each qubit receives `RX`, `RY`, and `RZ` in every layer
- the entanglement pattern is a linear chain of `CZ` gates
- the number of trainable parameters per layer is `3 x num_qubits`

With 6 qubits, that means each layer has 18 trainable rotation parameters before adding the next entangling chain.

Why this ansatz is useful here:

- It is richer than the `basic` ansatz because it gives each qubit a full three-angle rotation block.
- It keeps the entangling pattern simple and local, which matches the kind of connectivity many real devices expose.
- It is easier to reason about than `strongly_entangling`, while still being more expressive than the smallest baseline.

Tradeoffs:

- It is more expressive than `basic`, but still more structured and less aggressive than `strongly_entangling`.
- Its nearest-neighbor entanglement may limit how quickly long-range correlations spread across qubits.
- It can be a good middle ground when `basic` feels too weak and `strongly_entangling` feels too unconstrained or too expensive.

In short, `hardware_efficient` is the “balanced middle option” in this repo: more flexible than `basic`, more structured than `strongly_entangling`, and a natural choice when you want a realistic layered circuit with local entanglement.

#### `qcnn`

This option uses a QCNN-inspired multiscale circuit, following the general architectural idea introduced by Cong, Choi, and Lukin in [Quantum convolutional neural networks](https://arxiv.org/abs/1810.03787). Instead of applying the same entangling pattern everywhere, it processes the 6-qubit register through local two-qubit blocks arranged from fine to coarse scales.

For each layer in this project, the block pattern is:

- local pair blocks on `(0,1)`, `(2,3)`, and `(4,5)`
- wider pair blocks on `(1,2)` and `(3,4)`
- one coarse block on `(2,3)`

Each two-qubit block includes:

- single-qubit `RY` and `RZ` rotations on both qubits
- a short entangling sequence built from `CNOT`
- a final pair of trainable rotations after entanglement

This gives the ansatz a convolution-like inductive bias:

- early blocks focus on short-range local structure
- later blocks mix information across wider receptive fields
- repeating `--layers` stacks this hierarchy multiple times

Why this ansatz is useful here:

- It is more structured than a generic hardware-efficient circuit.
- It introduces a hierarchical locality bias that fits image-like data.
- It gives a more “architectural” baseline between simple layered circuits and fully generic expressive templates.

Tradeoffs:

- It has more structure than `strongly_entangling`, which can make it easier to interpret.
- It is less general-purpose than a dense template because its entanglement schedule is intentionally constrained.
- In this implementation it is QCNN-inspired rather than a full qubit-dropping pooling network, since the model still keeps all 6 qubits available for the project’s shared readout pipeline.

In short, `qcnn` is the “multiscale local-structure option” in this repo: use it when you want a circuit with an image-inspired hierarchy rather than a uniform layer pattern everywhere.

#### `strongly_entangling`

This uses PennyLane's built-in `StronglyEntanglingLayers` template, which the PennyLane documentation describes as being inspired by the circuit-centric classifier design of Schuld, Bocharov, Svore, and Wiebe, [Circuit-centric quantum classifiers](https://arxiv.org/abs/1804.00633).

It is the richest ansatz in the project and serves as the most expressive default option. If you want a stronger baseline without hand-designing the entangling pattern yourself, this is usually the best starting point.

More specifically, `StronglyEntanglingLayers` is a layered template that combines:

- multiple trainable single-qubit rotations on every qubit in every layer
- a built-in entangling pattern designed to spread correlations across the register
- a denser parameterization than the simpler hand-written ansatzes in this project

For this project:

- the circuit uses 6 qubits
- the layer count is controlled by `--layers`
- the PennyLane template determines the exact internal rotation-and-entanglement structure

Why this ansatz is useful here:

- It gives a strong off-the-shelf expressive baseline.
- It usually explores a richer part of Hilbert space than the simpler `basic` circuit.
- It avoids hand-designing a custom entanglement schedule while still being more flexible than the nearest-neighbor `hardware_efficient` version.

Tradeoffs:

- It has more trainable freedom, which can help accuracy.
- It can also be slower to train and potentially harder to optimize.
- Because it is a generic template, it is less interpretable than the simpler hand-written ansatzes.

In short, `strongly_entangling` is the “high-capacity default” in this repo: use it when you want the most expressive built-in circuit before moving on to more specialized architectures or measurement designs.

### Why the model measures probabilities instead of a few observables

The quantum circuit returns the full probability vector over all 6-qubit basis states instead of only measuring one expectation value per class.

That means the ansatz is being used mainly as a quantum feature transformer. The final class decision is made by the PyTorch readout head, which maps the 64 quantum probabilities to 10 digit classes.

This design makes it easy to compare different ansatz families under the same encoding and readout setup.

## Experiment types

In this repo, the experiment category is determined by the readout, not by the ansatz.

All three ansatz options:

- `basic`
- `hardware_efficient`
- `qcnn`
- `strongly_entangling`

can be used in either a `quantum` or `hybrid` experiment depending on `--readout-mode`.

Classify the settings as follows:

- `--readout-mode probs_only`: `quantum`
- `--readout-mode linear`: `hybrid`
- `--readout-mode learnable_observable`: `hybrid`

Reasoning:

- `probs_only` uses the quantum circuit output directly and only applies a fixed grouping rule from basis-state probabilities to class probabilities. There is no trainable classical head after the quantum model.
- `linear` is hybrid because the quantum circuit produces features and a trainable PyTorch linear layer performs the final class mapping.
- `learnable_observable` is also hybrid in this implementation because a classical neural controller generates the observable parameters used for the final measurement.

So a few common examples are:

- `basic + probs_only`: `quantum`
- `hardware_efficient + probs_only`: `quantum`
- `qcnn + probs_only`: `quantum`
- `strongly_entangling + probs_only`: `quantum`
- `basic + linear`: `hybrid`
- `hardware_efficient + linear`: `hybrid`
- `qcnn + linear`: `hybrid`
- `strongly_entangling + linear`: `hybrid`
- `basic + learnable_observable`: `hybrid`
- `hardware_efficient + learnable_observable`: `hybrid`
- `qcnn + learnable_observable`: `hybrid`
- `strongly_entangling + learnable_observable`: `hybrid`

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

Another alternative is `--readout-mode learnable_observable`, where a neural controller programs Hermitian observables dynamically and the readout uses expectation values instead of a linear head.

## Learnable observable details

The `learnable_observable` mode is inspired by the paper [Learning to Program Quantum Measurements for Machine Learning](https://arxiv.org/pdf/2505.13525). The main idea is to make the measurement itself trainable and data-conditioned instead of fixing a small set of observables in advance.

### High-level idea

In the standard `linear` mode, the pipeline is:

1. Prepare a quantum state with amplitude embedding and a variational ansatz.
2. Convert that state to basis-state probabilities.
3. Apply a classical linear layer to map those probabilities to class logits.

In `learnable_observable` mode, the pipeline changes to:

1. Prepare a quantum state with amplitude embedding and a variational ansatz.
2. Keep the full quantum state instead of immediately reducing it to probabilities.
3. Use a neural controller to generate one Hermitian observable per class for the current input.
4. Compute the expectation value of each observable on the current quantum state.
5. Use those expectation values as class logits.

So the classifier is no longer asking, “How should a fixed classical head interpret the measurement output?” It is asking, “What measurement should we perform for this specific input?”

### What is learnable in this mode

There are two trainable parts:

- The variational circuit parameters `q_params`, which still control how the input is mapped into a quantum state.
- The observable programmer network, which generates measurement operators from the input features.

The observable programmer is a small MLP:

- Input: the 64-dimensional flattened image vector
- Hidden layer: 128 units with `SiLU`
- Output: enough parameters to build one full Hermitian matrix per class

For 6 qubits, the Hilbert-space dimension is 64, so each observable is a `64 x 64` Hermitian matrix.

### How the Hermitian observable is parameterized

Each class observable is built from:

- A real diagonal
- Real upper-triangular entries
- Imaginary upper-triangular entries

These values are assembled into a complex Hermitian matrix by reflecting the upper triangle onto the lower triangle with complex conjugation. This guarantees that the resulting observable is Hermitian, so its expectation value is real.

### How logits are computed

If the quantum circuit outputs a state vector $|\psi(x)\rangle$ and the programmer generates one observable $O_c(x)$ for class $c$, then the class logit is

$$
z_c(x) = \langle \psi(x) | O_c(x) | \psi(x) \rangle.
$$

The model computes one such value for each class and passes the resulting logit vector into cross-entropy loss.

### Why this is different from `qml.probs` and `linear`

`probs_only`:

- Uses only basis-state probabilities
- Has no trainable classical head
- Groups probabilities into class buckets with a fixed rule

`linear`:

- Uses basis-state probabilities
- Applies a trainable but input-independent classical linear head

`learnable_observable`:

- Uses the full quantum state
- Learns the measurement itself
- Makes the measurement depend on the input

This is strictly richer than a fixed linear interpretation of `qml.probs`, because the model can adapt its measurement operator to the sample being classified.

### Practical implications

Benefits:

- More expressive readout than a fixed linear head
- Closer to the measurement-programming idea in the paper
- Lets the model exploit phase information from the state, not only basis-state probabilities

Costs:

- Much larger classical readout parameterization
- Heavier memory and compute use than `linear` or `probs_only`
- More risk of overfitting on a small dataset

Because the observable is a full `64 x 64` Hermitian matrix per class, this mode is substantially more expensive than the default readouts.

### Recommended use

Use `learnable_observable` when you want to study richer measurement design or compare static readouts against programmable measurements. For quick baselines or fast experiments, `linear` is still the simpler default.

Example:

```bash
cd /Users/hoangquan/Workspaces/QuantumDistillation/code
train-vqa-mnist --ansatz strongly_entangling --readout-mode learnable_observable --num-classes 4
```

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
