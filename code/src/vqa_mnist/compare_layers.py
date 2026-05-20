from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from model import SUPPORTED_ANSATZES
from train import TrainingConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare one ansatz across different layer counts."
    )
    parser.add_argument(
        "--ansatz",
        choices=SUPPORTED_ANSATZES,
        default="strongly_entangling",
    )
    parser.add_argument(
        "--readout-mode",
        choices=("linear", "probs_only"),
        default="linear",
    )
    parser.add_argument(
        "--num-classes",
        choices=(4, 10),
        type=int,
        default=10,
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
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
        "--layer-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of layer counts to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries: list[dict] = []

    for layer_count in args.layer_values:
        print(f"running_ansatz={args.ansatz} layers={layer_count}")
        results = train_model(
            TrainingConfig(
                ansatz=args.ansatz,
                readout_mode=args.readout_mode,
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                layers=layer_count,
                seed=args.seed,
                train_limit=args.train_limit,
                val_limit=args.val_limit,
                test_limit=args.test_limit,
                device=args.device,
                quantum_device=args.quantum_device,
            )
        )
        summaries.append(
            {
                "ansatz": args.ansatz,
                "readout_mode": args.readout_mode,
                "num_classes": args.num_classes,
                "layers": layer_count,
                "test_loss": results["test_metrics"]["loss"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "final_val_accuracy": results["history"][-1]["val_accuracy"],
                "final_epoch_time_seconds": results["history"][-1]["epoch_time_seconds"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = args.output_dir / f"compare_layers_{args.ansatz}_{timestamp}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    for summary in summaries:
        print(
            f"layers={summary['layers']} "
            f"test_acc={summary['test_accuracy']:.3f} "
            f"test_loss={summary['test_loss']:.4f} "
            f"last_epoch_time={summary['final_epoch_time_seconds']:.2f}s"
        )
    print(f"saved_results={summary_path}")


if __name__ == "__main__":
    main()
