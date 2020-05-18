"""Simultaneous logging to console and files.
"""
import logging
import os
import sys

from src.utils import misc


def get_logger(name, config_name, timestamp):
    """Initialize logger for given name, directory name and timestamp.

    :param name: Loger instance name.
        If file path is used, e.g. `__file__`, only base name without extension
        will be used.
    :type name: str
    :param config_name: Configuration name
    :type config_name: str
    :param timestamp: Unix timestamp.
    :type timestamp: float
    :return: New instance of `Logger`
    :rtype: Logger
    """
    log_name = os.path.splitext(os.path.basename(name))[0]
    log_dir = misc.get_report_subdir(config_name, timestamp, 'logs')

    if not log_dir.is_dir():
        log_dir.mkdir(parents=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Add console handler.
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]:[%(name)s] %(message)s')
    )
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    # Add file handler.
    fileHandler = logging.FileHandler(log_dir / f'{log_name}.log')
    fileHandler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)

    return logger
