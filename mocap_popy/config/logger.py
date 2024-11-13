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

DEFAULT_LOGGING_MODE = "w"
DEFAULT_LOGFILE_BASENAME = "mocap_popy"
DEFAULT_LOGFILE_NAME = f"{DEFAULT_LOGFILE_BASENAME}_{NOW_STRING}.log"

LOGGING_FMT = "%(asctime)s [%(name)s:%(levelname)s] %(message)s"

try:
    LOGGING_STREAMS = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(sys_dir.LOG_DIR, DEFAULT_LOGFILE_NAME),
            mode=DEFAULT_LOGGING_MODE,
        ),
    ]
except FileNotFoundError:
    for ssc in SETUP_SUBPROCESS_CALLS:
        subprocess.call(ssc, **SUBPROCESS_CALL_KWARGS)
    LOGGING_STREAMS = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(sys_dir.LOG_DIR, DEFAULT_LOGFILE_NAME),
            mode=DEFAULT_LOGGING_MODE,
        ),
    ]

LOGGING_OPTIONS = {
    "level": logging.INFO,
    "format": LOGGING_FMT,
    "handlers": LOGGING_STREAMS,
}


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


def set_logging_filename(base_name: str):
    """!Set the logging filename in the logging configuration options

    This function enables the usage of this configuration file across
    multiple test scripts, while generating personalized test log files.

    NOTE: This should be done prior to setting the root logger.

    @param fn Target logfile name
    """
    global LOGGING_STREAMS, LOGGING_OPTIONS

    log_filename = f"{base_name}_{NOW_STRING}.log"
    LOGGING_STREAMS = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(sys_dir.LOG_DIR, log_filename), mode=DEFAULT_LOGGING_MODE
        ),
    ]

    LOGGING_OPTIONS = {
        "level": logging.INFO,
        "format": LOGGING_FMT,
        "handlers": LOGGING_STREAMS,
    }
