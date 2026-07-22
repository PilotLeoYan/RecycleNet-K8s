import logging

from src.utils.logger import get_logger


def test_get_logger_returns_logger_instance():
    """"""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_sets_correct_name():
    """"""
    logger_name = "logger_of_test"
    logger = get_logger(logger_name)
    assert logger.name == logger_name


def test_get_logger_singleton_behavior():
    """"""
    logger1 = get_logger("singleton_test")
    logger2 = get_logger("singleton_test")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1
