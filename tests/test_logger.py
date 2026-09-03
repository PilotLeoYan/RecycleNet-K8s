import json
import logging
import sys

from src.utils.formatter import JSONFormatter, LocalFormatter
from src.utils.logger import get_logger


def test_get_logger_returns_logger_instance() -> None:
    """"""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_sets_correct_name() -> None:
    """"""
    logger_name = "logger_of_test"
    logger = get_logger(logger_name)
    assert logger.name == logger_name


def test_get_logger_singleton_behavior() -> None:
    """"""
    logger1 = get_logger("singleton_test")
    logger2 = get_logger("singleton_test")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_local_formatter() -> None:
    formatter = LocalFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "Test message" in formatted
    assert "INFO" in formatted


def test_local_formatter_with_exception() -> None:
    formatter = LocalFormatter()

    try:
        raise ValueError("Simulated error for traceback")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=15,
        msg="An error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    assert "An error occurred" in formatted
    assert "ERROR" in formatted
    assert "ValueError: Simulated error for traceback" in formatted
    assert "Traceback" in formatted


def test_json_formatter() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=20,
        msg="Error message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["message"] == "Error message"
    assert data["severity"] == "ERROR"
