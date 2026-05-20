from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

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
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.05
    layers: int = 2
    seed: int = 123
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None


def _as_torch_features(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float64)


def _as_torch_labels(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.argmax(array, axis=1), dtype=torch.long)


def train_model(config: TrainingConfig) -> dict:
    torch.manual_seed(config.seed)
    splits = load_mnist8x8_splits(
        seed=config.seed,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        test_limit=config.test_limit,
    )
    spec = ModelSpec(ansatz=config.ansatz, num_layers=config.layers)
    model = VQADigitsClassifier(spec, seed=config.seed)
    train_x = _as_torch_features(splits.train_x)
    train_y = _as_torch_labels(splits.train_y)
    val_x = _as_torch_features(splits.val_x)
    val_y = _as_torch_labels(splits.val_y)
    test_x = _as_torch_features(splits.test_x)
    test_y = _as_torch_labels(splits.test_y)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
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
        epoch_metrics = {
            "epoch": epoch,
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} "
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
            "readout_weights": model.readout.weight.detach().cpu().numpy().tolist(),
            "readout_bias": model.readout.bias.detach().cpu().numpy().tolist(),
        },
    }


def save_run(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"train_{results['config']['ansatz']}_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ansatz", choices=SUPPORTED_ANSATZES, default="strongly_entangling")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        layers=args.layers,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
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
