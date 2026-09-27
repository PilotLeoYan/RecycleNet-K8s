"""Application CLI entrypoint for running RecycleNet training workflows."""

import sys

from src.config.schema import AppConfig
from src.pipeline.hpo_pipeline import HPOPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import RecycleNetException, build_parser, get_logger

logger = get_logger(__name__)


def main() -> None:
    """CLI entrypoint for launching the RecycleNet training pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Starting RecycleNet")

    try:
        config = AppConfig.from_yaml(args.config)
        logger.debug("Configuration successfully loaded from %s", config.__str__())

        match args.command:
            case "train":
                train_pipeline = TrainPipeline(config)
                train_pipeline.run()

            case "hpo":
                hpo_pipeline = HPOPipeline(config)
                hpo_pipeline.run()

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
