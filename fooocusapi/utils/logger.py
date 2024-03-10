"""a simple logger"""
import logging
import os
import sys

from colorlog import ColoredFormatter

log_dir = "logs"

proj_path = os.path.dirname(os.path.abspath(__file__))

formatter = ColoredFormatter(
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


class Logger:
    def __init__(self, log_name, log_format: ColoredFormatter = formatter):
        log_path = os.path.join(proj_path, '../../', log_dir)
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

        error_handler.setFormatter(file_formatter)
        info_handler.setFormatter(file_formatter)
        stream_handler.setFormatter(log_format)

        # 将handler添加到logger中
        self._file_logger.addHandler(error_handler)
        self._file_logger.addHandler(info_handler)
        self._std_logger.addHandler(stream_handler)

    def file_error(self, message):
        self._file_logger.error(message)

    def file_info(self, message):
        self._file_logger.info(message)

    def std_info(self, message):
        self._std_logger.info(message)

    def std_warn(self, message):
        self._std_logger.warning(message)

    def std_error(self, message):
        self._std_logger.error(message)


default_logger = Logger(log_name="fooocus_api")
