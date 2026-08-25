"""Application CLI entrypoint for running RecycleNet training workflows."""

import sys

from src.pipeline.train_pipeline import TrainingPipelineConfig, TrainPipeline
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """CLI entrypoint for launching the RecycleNet training pipeline."""
    logger.info("Starting RecycleNet")

    try:
        config = TrainingPipelineConfig()
        pipeline = TrainPipeline(config)
        pipeline.run()

        logger.info("Pipeline successfully completed")
    except Exception as e:
        logger.exception("Critical error whilst running the project: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

# python -m src.main
