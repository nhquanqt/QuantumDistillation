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
        description="Train and compare one ansatz across different layer counts."
    )
    parser.add_argument(
        "--ansatz",
        choices=SUPPORTED_ANSATZES,
        default="strongly_entangling",
    )
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
    comparison_log_lines: list[str] = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_config = TrainingConfig(
        ansatz=args.ansatz,
        readout_mode=args.readout_mode,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        layers=args.layer_values[0],
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        device=args.device,
        quantum_device=args.quantum_device,
    )
    comparison_dir = args.output_dir / experiment_folder_name(
        "compare_layers",
        summary_config,
        timestamp,
    )
    runs_dir = comparison_dir / "runs"

    for run_index, layer_count in enumerate(args.layer_values, start=1):
        run_header = f"running_ansatz={args.ansatz} layers={layer_count}"
        print(run_header)
        comparison_log_lines.append(run_header)
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
        save_run(
            results,
            runs_dir,
            folder_name=f"run{run_index:02d}_layers-{layer_count}",
        )
        summaries.append(
            {
                "ansatz": args.ansatz,
                "experiment_type": results["experiment_type"],
                "readout_mode": args.readout_mode,
                "num_classes": args.num_classes,
                "layers": layer_count,
                "test_loss": results["test_metrics"]["loss"],
                "test_accuracy": results["test_metrics"]["accuracy"],
                "final_val_accuracy": results["history"][-1]["val_accuracy"],
                "final_epoch_time_seconds": results["history"][-1]["epoch_time_seconds"],
            }
        )

    comparison_dir.mkdir(parents=True, exist_ok=True)
    layer_tag = "layers-" + "-".join(str(layer) for layer in args.layer_values)
    suffix = experiment_filename_suffix(summary_config)
    summary_path = comparison_dir / f"summary_{layer_tag}_{suffix}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    for summary in summaries:
        summary_log_line = (
            f"layers={summary['layers']} "
            f"test_acc={summary['test_accuracy']:.3f} "
            f"test_loss={summary['test_loss']:.4f} "
            f"last_epoch_time={summary['final_epoch_time_seconds']:.2f}s"
        )
        print(summary_log_line)
        comparison_log_lines.append(summary_log_line)

    (comparison_dir / "compare.log").write_text("\n".join(comparison_log_lines) + "\n")
    print(f"saved_results_dir={comparison_dir}")


if __name__ == "__main__":
    main()
