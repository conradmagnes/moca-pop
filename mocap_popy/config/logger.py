import logging

LOGGING_FMT = "%(asctime)s|%(levelname)s|%(module)s.%(funcName)s() -> %(msg)s"


def setup_logger(name: str) -> logging.Logger:
    """!Setup Logger

    @param name
    @return logger
    """
    logging.basicConfig(format=LOGGING_FMT, force=True)
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    return logger


def set_logging_level(logger_names: list, level: int):
    """Set Logging Level

    @param logger_names list of logger names to set
    @param level logging level number
    """

    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def toggle_loggers(state: bool):
    """!Turn Off Logging

    @param which state to toggle them to `True`, `False`, logging on and off respectively
    """

    _level = logging.NOTSET if state else logging.CRITICAL + 1

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(_level)
