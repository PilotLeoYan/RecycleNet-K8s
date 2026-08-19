"""
src/utils/logger.py

Example:
    from utils import get_logger

    logger = get_logger(__name__)

    logger.info('Starting project', extra={'framework': 'pytorch', 'device': 'cuda'})
"""

import logging
import os
import sys

from src.utils.formatter import JSONFormatter, LocalFormatter


def _use_json_logging() -> bool:
    """Choose whether use JSON or local console format."""
    log_format = os.environ.get("LOG_FORMAT", "").lower()
    if log_format == "json":
        return True
    if log_format in ("text", "local", "consolse"):
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
    """ """
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
