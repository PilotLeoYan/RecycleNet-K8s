"""Custom logging formatters for Google Cloud Logging and local terminal output."""

import json
import logging
from datetime import UTC, datetime
from typing import override

_LEVEL_TO_GCP_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

_RESERVED_RECORD_KEYS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
)


class JSONFormatter(logging.Formatter):
    """JSON formatter formatted for Google Cloud Logging structured ingestion.

    Serializes log records into single-line JSON objects with severity, timestamp,
    logger name, message, and any additional structured fields passed in `extra`.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formats the specified LogRecord as a single-line JSON payload.

        Args:
            record: The LogRecord instance to format.

        Returns:
            str: JSON-encoded string representation of the log record.
        """
        payload = {
            "severity": _LEVEL_TO_GCP_SEVERITY.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # merge in anything passed via extra={...} as structured field
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in payload:
                continue

            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)

        return json.dumps(payload, default=str)


class LocalFormatter(logging.Formatter):
    """Colorized console formatter for local development with traceback support.

    Provides ANSI colored output based on logging severity level and includes
    formatted tracebacks when exception information is present in the record.

    Attributes:
        COLORS: Mapping of logging severity levels to ANSI color escape codes.
        RESET: ANSI escape code to reset terminal formatting.
    """

    COLORS: dict[int, str] = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET: str = "\033[0m"

    def __init__(self, datefmt: str = "%H:%M:%S") -> None:
        """Initializes the LocalFormatter with per-level ANSI color formatters.

        Args:
            datefmt: Format string for the timestamp in log messages.
        """
        super().__init__(datefmt=datefmt)

        self._formatters = {
            level: logging.Formatter(
                f"{color}%(asctime)s | %(levelname)-8s | "
                f"%(name)s | %(message)s{self.RESET}",
                datefmt=datefmt,
            )
            for level, color in self.COLORS.items()
        }
        self._default = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=datefmt,
        )

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Formats the LogRecord with ANSI color coding and exception details.

        Args:
            record: The LogRecord instance to format.

        Returns:
            str: Colorized and formatted log message string.
        """
        formatter = self._formatters.get(record.levelno, self._default)
        return formatter.format(record)
