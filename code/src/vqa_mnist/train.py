from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import load_mnist8x8_splits
from model import (
    ModelSpec,
    SUPPORTED_ANSATZES,
    evaluate_metrics,
    VQADigitsClassifier,
)


@dataclass
class TrainingConfig:
    ansatz: str = "strongly_entangling"
    readout_mode: str = "linear"
    num_classes: int = 10
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.05
    layers: int = 2
    seed: int = 123
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    device: str = "auto"
    quantum_device: str = "cpu"


def _as_torch_features(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float64)


def _as_torch_labels(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.argmax(array, axis=1), dtype=torch.long)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def serialize_state_dict(module: torch.nn.Module | None) -> dict[str, list] | None:
    if module is None:
        return None
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in module.state_dict().items()
    }


def _slugify_value(value: object) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def experiment_filename_suffix(config: TrainingConfig) -> str:
    parts = [
        f"ansatz-{config.ansatz}",
        f"readout-{config.readout_mode}",
        f"classes-{config.num_classes}",
        f"layers-{config.layers}",
        f"epochs-{config.epochs}",
        f"batch-{config.batch_size}",
        f"lr-{_slugify_value(config.learning_rate)}",
        f"seed-{config.seed}",
        f"qdev-{config.quantum_device}",
        f"cdev-{config.device}",
    ]
    if config.train_limit is not None:
        parts.append(f"trainlim-{config.train_limit}")
    if config.val_limit is not None:
        parts.append(f"vallim-{config.val_limit}")
    if config.test_limit is not None:
        parts.append(f"testlim-{config.test_limit}")
    return "_".join(parts)


def train_model(config: TrainingConfig) -> dict:
    torch.manual_seed(config.seed)
    classical_device = resolve_device(config.device)
    splits = load_mnist8x8_splits(
        num_classes=config.num_classes,
        seed=config.seed,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        test_limit=config.test_limit,
    )
    spec = ModelSpec(
        ansatz=config.ansatz,
        num_layers=config.layers,
        num_classes=config.num_classes,
    )
    model = VQADigitsClassifier(
        spec,
        seed=config.seed,
        classical_device=classical_device,
        quantum_device=config.quantum_device,
        readout_mode=config.readout_mode,
    )
    train_x = _as_torch_features(splits.train_x)
    train_y = _as_torch_labels(splits.train_y).to(classical_device)
    val_x = _as_torch_features(splits.val_x)
    val_y = _as_torch_labels(splits.val_y).to(classical_device)
    test_x = _as_torch_features(splits.test_x)
    test_y = _as_torch_labels(splits.test_y).to(classical_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []

    print(
        f"quantum_device={config.quantum_device} classical_device={classical_device.type}"
        f" readout_mode={config.readout_mode} num_classes={config.num_classes}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = perf_counter()
        permutation = rng.permutation(len(train_x))
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]

        batch_losses: list[float] = []
        for start in range(0, len(shuffled_x), config.batch_size):
            end = start + config.batch_size
            batch_x = shuffled_x[start:end]
            batch_y = shuffled_y[start:end]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_metrics = evaluate_metrics(
            model,
            train_x,
            train_y,
            loss_fn,
        )
        val_metrics = evaluate_metrics(
            model,
            val_x,
            val_y,
            loss_fn,
        )
        epoch_time_seconds = float(perf_counter() - epoch_start_time)
        epoch_metrics = {
            "epoch": epoch,
            "epoch_time_seconds": epoch_time_seconds,
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} "
            f"time={epoch_metrics['epoch_time_seconds']:.2f}s "
            f"batch_loss={epoch_metrics['batch_loss']:.4f} "
            f"train_acc={epoch_metrics['train_accuracy']:.3f} "
            f"val_acc={epoch_metrics['val_accuracy']:.3f}"
        )

    test_metrics = evaluate_metrics(
        model,
        test_x,
        test_y,
        loss_fn,
    )
    return {
        "config": asdict(config),
        "model_spec": asdict(spec),
        "history": history,
        "test_metrics": test_metrics,
        "final_parameters": {
            "q_params": model.q_params.detach().cpu().numpy().tolist(),
            "readout_weights": (
                model.readout.weight.detach().cpu().numpy().tolist()
                if model.readout is not None
                else None
            ),
            "readout_bias": (
                model.readout.bias.detach().cpu().numpy().tolist()
                if model.readout is not None
                else None
            ),
            "observable_programmer": serialize_state_dict(model.observable_programmer),
        },
    }


def save_run(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = TrainingConfig(**results["config"])
    suffix = experiment_filename_suffix(config)
    path = output_dir / f"train_{suffix}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ansatz", choices=SUPPORTED_ANSATZES, default="strongly_entangling")
    parser.add_argument(
        "--readout-mode",
        choices=("linear", "probs_only", "learnable_observable"),
        default="linear",
    )
    parser.add_argument(
        "--num-classes",
        choices=(4, 10),
        type=int,
        default=10,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--quantum-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = TrainingConfig(
        ansatz=args.ansatz,
        readout_mode=args.readout_mode,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        layers=args.layers,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        device=args.device,
        quantum_device=args.quantum_device,
    )
    results = train_model(config)
    output_path = save_run(results, args.output_dir)
    print(
        f"test_loss={results['test_metrics']['loss']:.4f} "
        f"test_accuracy={results['test_metrics']['accuracy']:.3f}"
    )
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
