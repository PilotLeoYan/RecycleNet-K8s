"""Centralized logging utility for the RecycleNet project.

Provides a unified logger configured for either local development (colorized text)
or cloud runtime environments (Google Cloud Logging structured JSON).

Example:
    ```python
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training step", extra={"epoch": 1, "device": "cuda"})
    ```
"""

import logging
import os
import sys

from src.utils.formatter import JSONFormatter, LocalFormatter


def _use_json_logging() -> bool:
    """Determines whether to format logs as structured JSON or colorized console text.

    Checks the `LOG_FORMAT` environment variable first, then probes for common
    Google Cloud runtime environment variables (Cloud Run, Functions, App Engine,
    Vertex AI).

    Returns:
        bool: True if structured JSON logging should be enabled, False otherwise.
    """
    log_format = os.environ.get("LOG_FORMAT", "").lower()
    if log_format == "json":
        return True
    if log_format in ("text", "local", "console"):
        return False

    gcp_signals = (
        "K_SERVICE",  # Cloud Run / Cloud Functions (2nd gen)
        "FUNCTION_TARGET",  # Cloud Functions (1st gen)
        "GAE_ENV",  # App Engine
        "CLOUD_ML_PROJECT_ID",  # Vertex AI / AI Platform Training
        "AIP_MODEL_DIR",  # Vertex AI custom training / prediction
    )
    return any(os.environ.get(var) for var in gcp_signals)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Creates or retrieves a configured logger instance.

    Configures a `StreamHandler` targeting `sys.stdout` with either `JSONFormatter`
    or `LocalFormatter` depending on the runtime environment.

    Args:
        name: The name of the logger, typically `__name__` of the calling module.
        level: Optional logging level. If None, defaults to the `LOG_LEVEL`
            environment variable or `logging.INFO`.

    Returns:
        logging.Logger: Configured logger instance ready for logging.
    """
    logger = logging.getLogger(name)

    # if it already configured
    if logger.handlers:
        return logger

    # set level to INFO is it's been set
    if level is None:
        level = getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        )

    logger.setLevel(level)  # type: ignore

    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if _use_json_logging() else LocalFormatter())
    logger.addHandler(handler)

    return logger
