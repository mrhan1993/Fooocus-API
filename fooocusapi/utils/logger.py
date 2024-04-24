# -*- coding: utf-8 -*-

""" A simply logger.

This module is used to log the program.

@file: logger.py
@author: mrhan1993
@update: 2024-03-22
"""
import logging
import os
import sys

try:
    from colorlog import ColoredFormatter
except ImportError:
    from fooocusapi.utils.tools import run_pip
    run_pip(
        command="install colorlog",
        desc="Install colorlog for logger.",
        live=True
    )
finally:
    from colorlog import ColoredFormatter


own_path = os.path.dirname(os.path.abspath(__file__))
log_dir = "logs"
default_log_path = os.path.join(own_path, '../../', log_dir)

std_formatter = ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

file_formatter = ColoredFormatter(
    fmt="[%(asctime)s] %(levelname)-8s%(reset)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    no_color=True,
    style='%'
)


class ConfigLogger:
    """
    Configure logger.
    :param log_path: log file path, better absolute path
    :param std_format: stdout log format
    :param file_format: file log format
    """
    def __init__(self,
                 log_path: str = default_log_path,
                 std_format: ColoredFormatter = std_formatter,
                 file_format: ColoredFormatter = file_formatter) -> None:
        self.log_path = log_path
        self.std_format = std_format
        self.file_format = file_format


class Logger:
    """
    A simple logger.
    :param log_name: log name
    :param config: config logger
    """
    def __init__(self, log_name, config: ConfigLogger = ConfigLogger()):
        log_path = config.log_path
        err_log_path = os.path.join(str(log_path), f"{log_name}_error.log")
        info_log_path = os.path.join(str(log_path), f"{log_name}_info.log")
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self._file_logger = logging.getLogger(log_name)
        self._file_logger.setLevel("INFO")

        self._std_logger = logging.getLogger()
        self._std_logger.setLevel("INFO")

        # 创建一个ERROR级别的handler，将日志记录到error.log文件中
        error_handler = logging.FileHandler(err_log_path, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)

        # 创建一个INFO级别的handler，将日志记录到info.log文件中
        info_handler = logging.FileHandler(info_log_path, encoding='utf-8')
        info_handler.setLevel(logging.INFO)

        # 创建一个 stream handler
        stream_handler = logging.StreamHandler(sys.stdout)

        error_handler.setFormatter(config.file_format)
        info_handler.setFormatter(config.file_format)
        stream_handler.setFormatter(config.std_format)

        # 将handler添加到logger中
        self._file_logger.addHandler(error_handler)
        self._file_logger.addHandler(info_handler)
        self._std_logger.addHandler(stream_handler)

    def file_error(self, message):
        """file error log"""
        self._file_logger.error(message)

    def file_info(self, message):
        """file info log"""
        self._file_logger.info(message)

    def std_info(self, message):
        """std info log"""
        self._std_logger.info(message)

    def std_warn(self, message):
        """std warn log"""
        self._std_logger.warning(message)

    def std_error(self, message):
        """std error log"""
        self._std_logger.error(message)


logger = Logger(log_name="fooocus_api")
