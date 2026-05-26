from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml
import torch


SUPPORTED_ANSATZES = (
    "basic",
    "hardware_efficient",
    "qcnn",
    "strongly_entangling",
)


@dataclass(frozen=True)
class ModelSpec:
    ansatz: str
    num_qubits: int = 6
    num_layers: int = 2
    num_classes: int = 10


def resolve_quantum_device_name(quantum_device: str) -> str:
    if quantum_device == "cpu":
        return "default.qubit"
    if quantum_device == "cuda":
        return "lightning.gpu"
    raise ValueError(f"Unsupported quantum device: {quantum_device}")


def parameter_shape(spec: ModelSpec) -> tuple[int, ...]:
    if spec.ansatz == "strongly_entangling":
        return qml.StronglyEntanglingLayers.shape(
            n_layers=spec.num_layers,
            n_wires=spec.num_qubits,
        )
    if spec.ansatz == "basic":
        return (spec.num_layers, spec.num_qubits, 2)
    if spec.ansatz == "hardware_efficient":
        return (spec.num_layers, spec.num_qubits, 3)
    if spec.ansatz == "qcnn":
        return (spec.num_layers, 6, 6)
    raise ValueError(f"Unsupported ansatz: {spec.ansatz}")


def hermitian_parameter_count(matrix_dim: int) -> int:
    return matrix_dim * matrix_dim


def qcnn_block(params: torch.Tensor, wires: tuple[int, int]) -> None:
    left_wire, right_wire = wires
    qml.RY(params[0], wires=left_wire)
    qml.RZ(params[1], wires=left_wire)
    qml.RY(params[2], wires=right_wire)
    qml.RZ(params[3], wires=right_wire)
    qml.CNOT(wires=[left_wire, right_wire])
    qml.RY(params[4], wires=right_wire)
    qml.CNOT(wires=[right_wire, left_wire])
    qml.RZ(params[5], wires=left_wire)


def apply_qcnn_ansatz(params: torch.Tensor, num_layers: int) -> None:
    # A QCNN-inspired hierarchy: three local pair blocks, then two wider blocks,
    # then one coarse block. Repeating the stack increases circuit depth.
    pair_hierarchy = (
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
        (2, 3),
    )
    for layer in range(num_layers):
        for block_index, wires in enumerate(pair_hierarchy):
            qcnn_block(params[layer, block_index], wires)


def build_qnode(spec: ModelSpec, quantum_device: str = "cpu"):
    device_name = resolve_quantum_device_name(quantum_device)
    try:
        dev = qml.device(device_name, wires=spec.num_qubits)
    except Exception as exc:
        if quantum_device == "cuda":
            raise ValueError(
                "quantum_device='cuda' requires PennyLane's lightning.gpu backend. "
                "Install the GPU simulator support and ensure CUDA/cuQuantum are available."
            ) from exc
        raise
    wires = list(range(spec.num_qubits))

    @qml.qnode(dev, interface="torch")
    def circuit(features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        qml.AmplitudeEmbedding(features, wires=wires, normalize=True)

        if spec.ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(params, wires=wires)
        elif spec.ansatz == "basic":
            for layer in range(spec.num_layers):
                for wire in wires:
                    qml.RY(params[layer, wire, 0], wires=wire)
                    qml.RZ(params[layer, wire, 1], wires=wire)
                for wire in wires:
                    qml.CNOT(wires=[wire, (wire + 1) % spec.num_qubits])
        elif spec.ansatz == "hardware_efficient":
            for layer in range(spec.num_layers):
                for wire in wires:
                    qml.RX(params[layer, wire, 0], wires=wire)
                    qml.RY(params[layer, wire, 1], wires=wire)
                    qml.RZ(params[layer, wire, 2], wires=wire)
                for wire in range(spec.num_qubits - 1):
                    qml.CZ(wires=[wire, wire + 1])
        elif spec.ansatz == "qcnn":
            apply_qcnn_ansatz(params, spec.num_layers)
        else:
            raise ValueError(f"Unsupported ansatz: {spec.ansatz}")

        return qml.state()

    return circuit


class HermitianObservableProgrammer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        matrix_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.matrix_dim = matrix_dim
        self.num_classes = num_classes
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            torch.nn.SiLU(),
            torch.nn.Linear(
                hidden_dim,
                num_classes * hermitian_parameter_count(matrix_dim),
                dtype=torch.float64,
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).view(
            self.num_classes,
            hermitian_parameter_count(self.matrix_dim),
        )


class VQADigitsClassifier(torch.nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        seed: int = 123,
        classical_device: torch.device | None = None,
        quantum_device: str = "cpu",
        readout_mode: str = "linear",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.quantum_device = quantum_device
        self.qnode = build_qnode(spec, quantum_device=quantum_device)
        self.classical_device = classical_device or torch.device("cpu")
        self.readout_mode = readout_mode

        generator = torch.Generator()
        generator.manual_seed(seed)

        init_scale = 0.01
        state_dim = 2**spec.num_qubits
        self.q_params = torch.nn.Parameter(
            init_scale
            * torch.randn(parameter_shape(spec), generator=generator, dtype=torch.float64)
        )
        if self.readout_mode == "linear":
            self.readout = torch.nn.Linear(
                state_dim,
                spec.num_classes,
                dtype=torch.float64,
            )
            with torch.no_grad():
                self.readout.weight.copy_(
                    init_scale
                    * torch.randn(
                        self.readout.weight.shape,
                        generator=generator,
                        dtype=torch.float64,
                    )
                )
                self.readout.bias.zero_()
            self.readout.to(self.classical_device)
            self.observable_programmer = None
        elif self.readout_mode == "probs_only":
            self.readout = None
            self.observable_programmer = None
        elif self.readout_mode == "learnable_observable":
            self.readout = None
            self.observable_programmer = HermitianObservableProgrammer(
                input_dim=state_dim,
                matrix_dim=state_dim,
                num_classes=spec.num_classes,
            ).to(self.classical_device)
        else:
            raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

    def _state_to_probs(self, state: torch.Tensor) -> torch.Tensor:
        return torch.abs(state) ** 2

    def _observable_parameters_to_hermitian(
        self,
        observable_params: torch.Tensor,
    ) -> torch.Tensor:
        matrix_dim = 2**self.spec.num_qubits
        num_off_diagonal = matrix_dim * (matrix_dim - 1) // 2
        diagonal = observable_params[:, :matrix_dim]
        upper_real = observable_params[:, matrix_dim : matrix_dim + num_off_diagonal]
        upper_imag = observable_params[:, matrix_dim + num_off_diagonal :]

        observables = torch.zeros(
            (self.spec.num_classes, matrix_dim, matrix_dim),
            dtype=torch.complex128,
            device=observable_params.device,
        )
        diagonal_indices = torch.arange(matrix_dim, device=observable_params.device)
        observables[:, diagonal_indices, diagonal_indices] = diagonal.to(torch.complex128)

        row_indices, col_indices = torch.triu_indices(
            matrix_dim,
            matrix_dim,
            offset=1,
            device=observable_params.device,
        )
        upper_values = upper_real.to(torch.complex128) + 1j * upper_imag.to(
            torch.complex128
        )
        observables[:, row_indices, col_indices] = upper_values
        observables[:, col_indices, row_indices] = torch.conj(upper_values)
        return observables

    def _probs_to_logits(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.readout_mode == "linear":
            return self.readout(probs)

        class_probs = torch.zeros(
            self.spec.num_classes,
            dtype=probs.dtype,
            device=probs.device,
        )
        for basis_index, basis_prob in enumerate(probs):
            class_probs[basis_index % self.spec.num_classes] += basis_prob
        return torch.log(class_probs + 1e-12)

    def _state_to_observable_logits(
        self,
        state: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        observable_params = self.observable_programmer(features.to(self.classical_device))
        observables = self._observable_parameters_to_hermitian(observable_params)
        state = state.to(self.classical_device).to(torch.complex128)
        return torch.einsum("i,cij,j->c", torch.conj(state), observables, state).real

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            state = self.qnode(features.to("cpu"), self.q_params)
            if self.readout_mode == "learnable_observable":
                return self._state_to_observable_logits(state, features)
            probs = self._state_to_probs(state).to(self.classical_device)
            return self._probs_to_logits(probs)

        logits = []
        for sample in features:
            state = self.qnode(sample.to("cpu"), self.q_params)
            if self.readout_mode == "learnable_observable":
                logits.append(self._state_to_observable_logits(state, sample))
                continue
            probs = self._state_to_probs(state).to(self.classical_device)
            logits.append(self._probs_to_logits(probs))
        return torch.stack(logits)


def evaluate_metrics(
    model: VQADigitsClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = float(loss_fn(logits, labels).item())
        accuracy = float((logits.argmax(dim=1) == labels).double().mean().item())
    return {"loss": loss, "accuracy": accuracy}
