import json
import logging
from datetime import UTC, datetime

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
    """One JSON object per line, shapped for Cloud Logging's auto-parser."""

    def format(self, record: logging.LogRecord) -> str:
        """"""
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
    """Colorized formatter for local develop."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """"""
        color = self.COLORS.get(record.levelno, "")
        base = (
            f"{color}%(asctime)s | %(levelname)-8s | %(name)s | %(message)s{self.RESET}"
        )

        return logging.Formatter(base, datefmt="%H:%M:%S").format(record)
