from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from dataset import load_mnist8x8_splits
from model import (
    ModelSpec,
    SUPPORTED_ANSATZES,
    experiment_type,
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


def _num_steps(num_samples: int, batch_size: int) -> int:
    return max(1, (num_samples + batch_size - 1) // batch_size)


def _format_eta(seconds: float) -> str:
    remaining = max(0, int(round(seconds)))
    minutes, secs = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


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


def experiment_folder_name(
    prefix: str,
    config: TrainingConfig,
    timestamp: str | None = None,
) -> str:
    suffix = experiment_filename_suffix(config)
    resolved_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{suffix}_{resolved_timestamp}"


def evaluate_metrics_batched(
    model: VQADigitsClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
    *,
    batch_size: int,
    phase: str,
    epoch: int,
    log_lines: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
    num_samples = len(features)
    num_steps = _num_steps(num_samples, batch_size)
    total_loss = 0.0
    total_correct = 0
    phase_start_time = perf_counter()

    with torch.no_grad():
        for step_index, start in enumerate(range(0, num_samples, batch_size), start=1):
            end = start + batch_size
            batch_x = features[start:end]
            batch_y = labels[start:end]
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            batch_count = len(batch_x)
            batch_accuracy = float(
                (logits.argmax(dim=1) == batch_y).double().mean().item()
            )
            total_loss += float(loss.item()) * batch_count
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            elapsed_seconds = perf_counter() - phase_start_time
            average_step_time = elapsed_seconds / step_index
            remaining_steps = num_steps - step_index
            eta_seconds = average_step_time * remaining_steps

            step_log_line = (
                f"epoch={epoch:02d} {phase}_step={step_index}/{num_steps} "
                f"batch_size={batch_count} "
                f"loss={loss.item():.4f} "
                f"acc={batch_accuracy:.3f} "
                f"eta={_format_eta(eta_seconds)}"
            )
            print(step_log_line)
            if log_lines is not None:
                log_lines.append(step_log_line)

    return {
        "loss": total_loss / num_samples,
        "accuracy": total_correct / num_samples,
    }


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
    dataset_sizes = {
        "train": int(len(train_x)),
        "val": int(len(val_x)),
        "test": int(len(test_x)),
    }
    step_counts = {
        "train": _num_steps(dataset_sizes["train"], config.batch_size),
        "val": _num_steps(dataset_sizes["val"], config.batch_size),
        "test": _num_steps(dataset_sizes["test"], config.batch_size),
    }
    run_experiment_type = experiment_type(config.ansatz, config.readout_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_state_dict = _cpu_state_dict(model.state_dict())
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    log_lines: list[str] = []

    run_header = (
        f"experiment_type={run_experiment_type} "
        f"quantum_device={config.quantum_device} classical_device={classical_device.type}"
        f" readout_mode={config.readout_mode} num_classes={config.num_classes}"
    )
    split_header = (
        f"samples_train={dataset_sizes['train']} "
        f"samples_val={dataset_sizes['val']} "
        f"samples_test={dataset_sizes['test']}"
    )
    steps_header = (
        f"steps_train={step_counts['train']} "
        f"steps_val={step_counts['val']} "
        f"steps_test={step_counts['test']}"
    )
    print(run_header)
    print(split_header)
    print(steps_header)
    log_lines.append(run_header)
    log_lines.append(split_header)
    log_lines.append(steps_header)

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = perf_counter()
        permutation = rng.permutation(len(train_x))
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]

        batch_losses: list[float] = []
        train_loss_total = 0.0
        train_correct_total = 0
        train_sample_total = 0
        for step_index, start in enumerate(
            range(0, len(shuffled_x), config.batch_size),
            start=1,
        ):
            end = start + config.batch_size
            batch_x = shuffled_x[start:end]
            batch_y = shuffled_y[start:end]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
            batch_count = len(batch_x)
            batch_accuracy = float(
                (logits.argmax(dim=1) == batch_y).double().mean().item()
            )
            train_loss_total += float(loss.item()) * batch_count
            train_correct_total += int((logits.argmax(dim=1) == batch_y).sum().item())
            train_sample_total += batch_count
            elapsed_seconds = perf_counter() - epoch_start_time
            average_step_time = elapsed_seconds / step_index
            remaining_steps = step_counts["train"] - step_index
            eta_seconds = average_step_time * remaining_steps
            train_step_log_line = (
                f"epoch={epoch:02d} train_step={step_index}/{step_counts['train']} "
                f"loss={loss.item():.4f} "
                f"acc={batch_accuracy:.3f} "
                f"eta={_format_eta(eta_seconds)}"
            )
            print(train_step_log_line)
            log_lines.append(train_step_log_line)

        train_metrics = {
            "loss": train_loss_total / train_sample_total,
            "accuracy": train_correct_total / train_sample_total,
        }
        val_metrics = evaluate_metrics_batched(
            model,
            val_x,
            val_y,
            loss_fn,
            batch_size=config.batch_size,
            phase="validate_val",
            epoch=epoch,
            log_lines=log_lines,
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
        if (
            epoch_metrics["val_accuracy"] > best_val_accuracy
            or (
                epoch_metrics["val_accuracy"] == best_val_accuracy
                and epoch_metrics["val_loss"] < best_val_loss
            )
        ):
            best_epoch = epoch
            best_val_accuracy = epoch_metrics["val_accuracy"]
            best_val_loss = epoch_metrics["val_loss"]
            best_state_dict = _cpu_state_dict(model.state_dict())
        epoch_log_line = (
            f"epoch={epoch:02d} "
            f"time={epoch_metrics['epoch_time_seconds']:.2f}s "
            f"batch_loss={epoch_metrics['batch_loss']:.4f} "
            f"train_acc={epoch_metrics['train_accuracy']:.3f} "
            f"val_acc={epoch_metrics['val_accuracy']:.3f}"
        )
        print(epoch_log_line)
        log_lines.append(epoch_log_line)

    final_state_dict = _cpu_state_dict(model.state_dict())
    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_metrics(
        model,
        test_x,
        test_y,
        loss_fn,
    )
    summary_log_line = (
        f"best_epoch={best_epoch} "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.3f}"
    )
    log_lines.append(summary_log_line)
    return {
        "config": asdict(config),
        "experiment_type": run_experiment_type,
        "model_spec": asdict(spec),
        "dataset_sizes": dataset_sizes,
        "step_counts": step_counts,
        "history": history,
        "best_checkpoint": {
            "epoch": best_epoch,
            "val_accuracy": best_val_accuracy,
            "val_loss": best_val_loss,
        },
        "test_metrics": test_metrics,
        "test_model_epoch": best_epoch,
        "_artifacts": {
            "log_lines": log_lines,
            "best_model_state_dict": best_state_dict,
            "final_model_state_dict": final_state_dict,
        },
    }


def save_run(
    results: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str = "train",
    folder_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = TrainingConfig(**results["config"])
    experiment_dir = output_dir / (
        folder_name or experiment_folder_name(prefix, config, timestamp)
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results.get("_artifacts", {})
    json_payload = {
        key: value
        for key, value in results.items()
        if key != "_artifacts"
    }

    (experiment_dir / "results.json").write_text(json.dumps(json_payload, indent=2))

    log_lines = artifacts.get("log_lines", [])
    if log_lines:
        (experiment_dir / "train.log").write_text("\n".join(log_lines) + "\n")

    best_model_state_dict = artifacts.get("best_model_state_dict")
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, experiment_dir / "best_model.pt")

    final_model_state_dict = artifacts.get("final_model_state_dict")
    if final_model_state_dict is not None:
        torch.save(final_model_state_dict, experiment_dir / "final_model.pt")

    return experiment_dir


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
        f"best_epoch={results['best_checkpoint']['epoch']} "
        f"test_loss={results['test_metrics']['loss']:.4f} "
        f"test_accuracy={results['test_metrics']['accuracy']:.3f}"
    )
    print(f"saved_results_dir={output_path}")


if __name__ == "__main__":
    main()
