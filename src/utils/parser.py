"""Command-line argument parser construction for the RecycleNet CLI."""

import argparse
from pathlib import Path

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser with subcommands for each stage.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="recyclenet",
        description="RecycleNet-K8s MLOps Orchestration CLI",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Workflow stage to execute",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Run end-to-end model training, validation, and test evaluation pipeline",
    )
    train_parser.add_argument(
        "--use-best-hpo",
        action="store_true",
        default=False,
        help="Override config with configs/best_hpo_config.yaml if it exists",
    )

    subparsers.add_parser(
        "hpo",
        help="Run distributed hyperparameter optimization with Ray Tune",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export trained PyTorch model to ONNX format and verify numerical parity",
    )
    export_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a local .pt checkpoint file (defaults to MLflow registry)"
        ),
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/recyclenet.onnx"),
        help="Destination path for the exported ONNX model",
    )
    export_parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Number of output classification classes (default: 6)",
    )

    return parser
