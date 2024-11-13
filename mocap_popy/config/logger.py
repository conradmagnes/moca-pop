import logging
import os
import subprocess
import sys
import datetime

import mocap_popy.config.directory as sys_dir

SETUP_SUBPROCESS_CALLS = [
    f"mkdir -p {sys_dir.LOG_DIR}",
]

SUBPROCESS_CALL_KWARGS = {"shell": True}

TIME_STR_FMT = "%Y-%m-%d %H:%M:%S"
NOW_TIMESTAMP = datetime.datetime.now()
NOW_STRING = NOW_TIMESTAMP.strftime(TIME_STR_FMT)

LOG_FILENAME = "mocap_popy"

LOGGING_MODE = "a"
LOGGING_FMT = "%(asctime)s [%(name)s:%(levelname)s] %(message)s"
LOGGING_LEVEL = logging.INFO


def set_root_logger():
    """!Set the root logger (useful to reference same file after renaming)"""
    params = {"level": LOGGING_LEVEL, "format": LOGGING_FMT}

    logging.basicConfig(**params, force=True)
    logging.root.addHandler(generate_stream_handler())
    logging.root.addHandler(generate_file_handler())


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
    mode = mode or LOGGING_MODE
    if mode is None or mode in ["", "none", "off"]:
        return logging.NullHandler()

    name = name or LOG_FILENAME
    filename = generate_log_filename(name)

    try:
        return logging.FileHandler(os.path.join(sys_dir.LOG_DIR, filename), mode=mode)
    except FileNotFoundError:
        for ssc in SETUP_SUBPROCESS_CALLS:
            subprocess.call(ssc, **SUBPROCESS_CALL_KWARGS)

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

    NOTE: This should be done prior to naming the logfile and setting the root logger.

    @param mode Log mode (e.g. 'w', 'a', 'r+')
    """
    global LOGGING_MODE
    LOGGING_MODE = mode

    current_file_handler = get_file_handler()
    if current_file_handler is not None:
        logging.root.removeHandler(current_file_handler)
        current_file_handler.close()
        filename = current_file_handler.baseFilename

        new_file_handler = generate_file_handler(name=filename, mode=mode)
        logging.root.addHandler(new_file_handler)


def set_logging_filename(name: str):
    """!Set the logging filename in the logging configuration options

    This function enables the usage of this configuration file across
    multiple test scripts, while generating personalized test log files.

    @param fn Target logfile name (timestamp will be appended)
    """
    global LOG_FILENAME

    filename = generate_log_filename(name)
    LOG_FILENAME = filename

    current_file_handler = get_file_handler()
    if current_file_handler is not None:
        logging.root.removeHandler(current_file_handler)
        current_file_handler.close()

        new_file_handler = generate_file_handler(name=filename)
        logging.root.addHandler(new_file_handler)


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
