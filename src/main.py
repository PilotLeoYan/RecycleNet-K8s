"""Application CLI entrypoint for running RecycleNet training workflows."""

import sys
from pathlib import Path

from src.config.schema import AppConfig
from src.pipeline.export_onnx import ONNXExportPipeline
from src.pipeline.hpo_pipeline import HPOPipeline
from src.pipeline.super_train_pipeline import SuperTrainPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import RecycleNetException, build_parser, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Execute the CLI entrypoint for launching RecycleNet pipeline workflows."""
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Starting RecycleNet")

    try:
        config = AppConfig.from_yaml(args.config)
        logger.debug("Configuration successfully loaded from %s", config.__str__())

        match args.command:
            case "train":
                if getattr(args, "use_best_hpo", False):
                    best_hpo_path = Path(config.hpo.best_config_path)
                    if best_hpo_path.exists():
                        config = AppConfig.from_yaml(best_hpo_path)
                        logger.info(
                            "Overriding configuration with best HPO config from %s",
                            best_hpo_path,
                        )
                    else:
                        logger.warning(
                            "Best HPO config %s not found; proceeding with %s",
                            best_hpo_path,
                            args.config,
                        )
                train_pipeline = TrainPipeline(config)
                train_pipeline.run()

            case "hpo":
                hpo_pipeline = HPOPipeline(config)
                hpo_pipeline.run()

            case "export":
                onnx_pipeline = ONNXExportPipeline(config, output_path=args.output)
                onnx_pipeline.run(
                    num_classes=args.num_classes,
                    checkpoint_path=args.checkpoint,
                )

            case "super-train":
                super_train_pipeline = SuperTrainPipeline(
                    config,
                    output_path=args.output,
                    skip_hpo=args.skip_hpo,
                )
                super_train_pipeline.run(num_classes=args.num_classes)

            case _:
                parser.error(f"Unrecognized command '{args.command}'")

        logger.info("Workflow '%s' completed successfully", args.command)

    except RecycleNetException as e:
        logger.exception("RecycleNet pipeline execution error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Critical error whilst running RecycleNet: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
