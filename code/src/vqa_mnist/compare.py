from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from model import SUPPORTED_ANSATZES
from train import (
    TrainingConfig,
    experiment_filename_suffix,
    experiment_folder_name,
    save_run,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare all supported VQA ansatzes."
    )
    parser.add_argument("--epochs", type=int, default=8)
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
    comparison_log_lines: list[str] = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_config = TrainingConfig(
        ansatz="all",
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
    comparison_dir = args.output_dir / experiment_folder_name(
        "compare_vqas",
        summary_config,
        timestamp,
    )
    runs_dir = comparison_dir / "runs"

    for run_index, ansatz in enumerate(SUPPORTED_ANSATZES, start=1):
        run_header = f"running_ansatz={ansatz}"
        print(run_header)
        comparison_log_lines.append(run_header)
        results = train_model(
            TrainingConfig(
                ansatz=ansatz,
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
        )
        save_run(
            results,
            runs_dir,
            folder_name=f"run{run_index:02d}_{ansatz}",
        )
        summaries.append(
            {
                "ansatz": ansatz,
                "readout_mode": args.readout_mode,
                "num_classes": args.num_classes,
                "test_loss": results["test_metrics"]["loss"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "final_val_accuracy": results["history"][-1]["val_accuracy"],
            }
        )

    comparison_dir.mkdir(parents=True, exist_ok=True)
    suffix = experiment_filename_suffix(summary_config)
    summary_path = comparison_dir / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    for summary in summaries:
        summary_log_line = (
            f"ansatz={summary['ansatz']} "
            f"test_acc={summary['test_accuracy']:.3f} "
            f"test_loss={summary['test_loss']:.4f}"
        )
        print(summary_log_line)
        comparison_log_lines.append(summary_log_line)

    (comparison_dir / "compare.log").write_text("\n".join(comparison_log_lines) + "\n")
    print(f"saved_results_dir={comparison_dir}")


if __name__ == "__main__":
    main()
