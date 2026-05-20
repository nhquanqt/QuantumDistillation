from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from model import SUPPORTED_ANSATZES
from train import TrainingConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare all supported VQA ansatzes."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--readout-mode",
        choices=("linear", "probs_only"),
        default="linear",
    )
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
    args = build_parser().parse_args()
    summaries: list[dict] = []

    for ansatz in SUPPORTED_ANSATZES:
        print(f"running_ansatz={ansatz}")
        results = train_model(
            TrainingConfig(
                ansatz=ansatz,
                readout_mode=args.readout_mode,
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
        )
        summaries.append(
            {
                "ansatz": ansatz,
                "readout_mode": args.readout_mode,
                "test_loss": results["test_metrics"]["loss"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "final_val_accuracy": results["history"][-1]["val_accuracy"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = args.output_dir / f"compare_vqas_{timestamp}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    for summary in summaries:
        print(
            f"ansatz={summary['ansatz']} "
            f"test_acc={summary['test_accuracy']:.3f} "
            f"test_loss={summary['test_loss']:.4f}"
        )
    print(f"saved_results={summary_path}")


if __name__ == "__main__":
    main()
