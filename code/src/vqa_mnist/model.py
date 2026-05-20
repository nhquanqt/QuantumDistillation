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


def build_qnode(spec: ModelSpec):
    dev = qml.device("default.qubit", wires=spec.num_qubits)
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
    def __init__(self, spec: ModelSpec, seed: int = 123) -> None:
        super().__init__()
        self.spec = spec
        self.qnode = build_qnode(spec)

        generator = torch.Generator()
        generator.manual_seed(seed)

        init_scale = 0.01
        self.q_params = torch.nn.Parameter(
            init_scale
            * torch.randn(parameter_shape(spec), generator=generator, dtype=torch.float64)
        )
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            probs = self.qnode(features, self.q_params)
            return self.readout(probs)

        logits = [self.readout(self.qnode(sample, self.q_params)) for sample in features]
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
