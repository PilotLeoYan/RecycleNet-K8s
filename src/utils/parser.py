import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parse with subcommands for each stage.

    Returns:
        Configured ArgumentParser instance.
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
        help="Worflow stage to execute",
        required=True,
    )

    subparsers.add_parser(
        "train",
        help="Run end-to-end model training, validation, and test evaluation pipeline",
    )

    subparsers.add_parser(
        "hpo",
        help="Run distributed hyperparameter optimization with Ray Tune",
    )

    return parser
