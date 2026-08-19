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
    """Colorized formatter for local development with full traceback support."""

    COLORS: dict[int, str] = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET: str = "\033[0m"

    def __init__(self, datefmt: str = "%H:%M:%S") -> None:
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
        formatter = self._formatters.get(record.levelno, self._default)
        message = formatter.format(record)

        # If the log contains an exception (exc_info), we add the full traceback
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message += f"\n{record.exc_text}"

        color = self.COLORS.get(record.levelno, "")
        base = (
            f"{color}%(asctime)s | %(levelname)-8s | %(name)s | %(message)s{self.RESET}"
        )

        return logging.Formatter(base, datefmt="%H:%M:%S").format(record)
