"""Super-train pipeline: sequential HPO → Train (best-HPO config) → ONNX export."""

from pathlib import Path

from src.config.schema import AppConfig
from src.pipeline.export_onnx import ONNXExportPipeline
from src.pipeline.hpo_pipeline import HPOPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import RecycleNetException, get_logger

logger = get_logger(__name__)


class SuperTrainPipeline:
    """Orchestrate the full HPO → Train → ONNX export pipeline in a single run.

    This pipeline chains three stages sequentially:

    1. **HPO** – Distributed Ray Tune search that saves the best
       hyperparameter config to ``config.hpo.best_config_path``.
    2. **Train** – Full production training that automatically loads the
       best HPO config written in stage 1 (equivalent to ``train
       --use-best-hpo``).
    3. **Export** – Exports the best model from the MLflow registry to
       ONNX format, verifying numerical parity.

    Attributes:
        config: Root application configuration object.
        output_path: Destination path for the exported ONNX model file.
        skip_hpo: When ``True``, stage 1 is skipped and the pipeline
            resumes from training using whatever HPO config already exists
            on disk (useful for re-runs after a successful HPO).
    """

    def __init__(
        self,
        config: AppConfig,
        output_path: Path | None = None,
        *,
        skip_hpo: bool = False,
    ) -> None:
        """Initialize the SuperTrainPipeline.

        Args:
            config: Root application configuration object.
            output_path: Destination path for the ONNX model.
                Defaults to ``config.export.output_path``.
            skip_hpo: If ``True``, skip the HPO stage and use an
                existing ``best_hpo_config.yaml``.  Defaults to ``False``.
        """
        self.config = config
        self.output_path = (
            output_path
            if output_path is not None
            else getattr(config, "export", None)
            and config.export.output_path
            or Path("models/recyclenet.onnx")
        )
        self.skip_hpo = skip_hpo

    def run(
        self,
        num_classes: int = 6,
    ) -> Path:
        """Execute the three-stage super-train pipeline end-to-end.

        Stages
        ------
        1. HPO (skipped when ``skip_hpo=True``)
        2. Train with best HPO config
        3. Export to ONNX

        Args:
            num_classes: Number of output classification classes.
                Defaults to 6 (TrashNet).

        Returns:
            Path: Resolved path to the exported ONNX model file.

        Raises:
            RecycleNetException: If any stage fails.
        """
        try:
            # ------------------------------------------------------------------
            # Stage 1 – Hyperparameter Optimisation
            # ------------------------------------------------------------------
            if self.skip_hpo:
                logger.info(
                    "[super-train] Skip HPO stage (--skip-hpo flag is set). "
                    "Using existing HPO config at %s",
                    self.config.hpo.best_config_path,
                )
            else:
                logger.info("[super-train] === [1/3] Starting HPO stage ===")
                hpo_pipeline = HPOPipeline(self.config)
                hpo_pipeline.run()
                logger.info("[super-train] HPO stage completed.")

            # ------------------------------------------------------------------
            # Stage 2 – Production Training with best HPO config
            # ------------------------------------------------------------------
            logger.info(
                "[super-train] == [2/3] Starting Training stage "
                "(loading best HPO config) =="
            )
            best_hpo_path = Path(self.config.hpo.best_config_path)
            if best_hpo_path.exists():
                train_config = AppConfig.from_yaml(best_hpo_path)
                logger.info(
                    "[super-train] Config overridden with best HPO config: %s",
                    best_hpo_path,
                )
            else:
                train_config = self.config
                logger.warning(
                    "[super-train] Best HPO config not found at %s; "
                    "training with base config %s",
                    best_hpo_path,
                    self.config,
                )

            train_pipeline = TrainPipeline(train_config)
            train_pipeline.run()
            logger.info("[super-train] Training stage completed.")

            # ------------------------------------------------------------------
            # Stage 3 – ONNX Export
            # ------------------------------------------------------------------
            logger.info("[super-train] == [3/3] Starting ONNX Export stage ==")
            export_pipeline = ONNXExportPipeline(
                train_config, output_path=self.output_path
            )
            onnx_path = export_pipeline.run(num_classes=num_classes)
            logger.info(
                " === PIPELINE COMPLETED SUCCESSFULLY. ONNX: %s ==",
                onnx_path,
            )
            return onnx_path

        except RecycleNetException:
            raise
        except Exception as e:
            raise RecycleNetException(
                "Critical failure in super-train pipeline", e
            ) from e
