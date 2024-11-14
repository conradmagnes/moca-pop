import logging
import os
import subprocess
import sys
import datetime

import mocap_popy.config.directory as sys_dir


TIME_STR_FMT = "%Y-%m-%d %H:%M:%S"
NOW_TIMESTAMP = datetime.datetime.now()
NOW_STRING = NOW_TIMESTAMP.strftime(TIME_STR_FMT)

DEFAULT_LOG_FILENAME = "mocap_popy"
DEFAULT_LOGGING_MODE = "off"
DEFAULT_LOGGING_FMT = "%(asctime)s [%(name)s:%(levelname)s] %(message)s"
DEFAULT_LOGGING_LEVEL = logging.INFO


def set_root_logger(
    name: str = None, mode: str = None, fmt: str = None, level: int = None
):
    """!Set the root logger (useful to reference same file after renaming)"""
    name = name or DEFAULT_LOG_FILENAME
    mode = mode or DEFAULT_LOGGING_MODE
    fmt = fmt or DEFAULT_LOGGING_FMT
    level = level or DEFAULT_LOGGING_LEVEL

    stream_handler = generate_stream_handler()
    file_handler = generate_file_handler(name=name, mode=mode)
    params = {
        "level": level,
        "format": fmt,
        "handlers": [stream_handler, file_handler],
    }

    logging.basicConfig(**params, force=True)


def synchronize_logger(logger_name: str = None):
    """Ensure custom logger inherits root logger's handlers."""
    logger = logging.getLogger(logger_name or "")
    logger.handlers = logging.root.handlers
    logger.setLevel(logging.root.level)


def generate_log_filename(name: str, timestamp: str = None):
    """!Generate a log filename based on a name and timestamp

    @param base_name Base name for the log file
    @param timestamp Timestamp to use in the filename
    """
    timestamp = NOW_STRING if timestamp is None else timestamp
    return f"{name}_{timestamp}.log"


def generate_file_handler(name: str = None, mode: str = None):
    """!Get a file handler for logging

    @param name Base name of the log file (timestamp will be appended)
    @param mode Mode for the file handler. Default is 'a' (append).
            User "off" or "none" to disable logging to file.
    """
    current = get_file_handler()
    if mode is None:
        mode = current.mode if current is not None else DEFAULT_LOGGING_MODE

    if mode is None or mode in ["", "none", "off"]:
        return logging.NullHandler()

    basename = name or DEFAULT_LOG_FILENAME
    filename = generate_log_filename(basename)

    os.makedirs(sys_dir.LOG_DIR, exist_ok=True)
    print(os.path.join(sys_dir.LOG_DIR, filename))
    return logging.FileHandler(os.path.join(sys_dir.LOG_DIR, filename), mode=mode)


def generate_stream_handler():
    """!Get a stream handler for logging"""
    return logging.StreamHandler(sys.stdout)


def get_file_handler():
    """!Get the file handler for logging"""
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler
    return None


def get_stream_handler():
    """!Get the stream handler for logging"""
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            return handler
    return None


def set_logging_mode(mode: str):
    """!Set the LOGGING_MODE for the logging handlers.

    @param mode Log mode (e.g. 'w', 'a', 'r+')
    """
    current = get_file_handler()
    if current is None:
        current = generate_file_handler(mode=mode)
        if current is not None:
            logging.root.addHandler(current)
    else:
        current.mode = mode


def set_global_logging_level(level: int):
    """Set Logging Level globally

    @param level logging level number
    """
    for obj in logging.root.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            obj.setLevel(level)


def toggle_loggers(state: bool, logger_names: list = None):
    """Toggle logging state for specified loggers or all loggers."""
    _level = logging.NOTSET if state else logging.CRITICAL + 1

    loggers = (
        [logging.getLogger(name) for name in logger_names]
        if logger_names
        else logging.root.manager.loggerDict.values()
    )

    for logger in loggers:
        logger.setLevel(_level)
