from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml
import torch


SUPPORTED_ANSATZES = (
    "basic",
    "hardware_efficient",
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
    raise ValueError(f"Unsupported ansatz: {spec.ansatz}")


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
        else:
            raise ValueError(f"Unsupported ansatz: {spec.ansatz}")

        return qml.probs(wires=wires)

    return circuit


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
        self.q_params = torch.nn.Parameter(
            init_scale
            * torch.randn(parameter_shape(spec), generator=generator, dtype=torch.float64)
        )
        if self.readout_mode == "linear":
            self.readout = torch.nn.Linear(
                2**spec.num_qubits,
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
        elif self.readout_mode == "probs_only":
            self.readout = None
        else:
            raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

    def _probs_to_logits(self, probs: torch.Tensor) -> torch.Tensor:
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            probs = self.qnode(features.to("cpu"), self.q_params)
            probs = probs.to(self.classical_device)
            return self._probs_to_logits(probs)

        logits = []
        for sample in features:
            probs = self.qnode(sample.to("cpu"), self.q_params)
            probs = probs.to(self.classical_device)
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
