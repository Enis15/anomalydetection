import os
import json

import logging.handlers
import logging

from datetime import datetime

def logger(name: str = __name__, log_folder_name: str = './', log_file_name: str = 'status.log'):
    if not log_folder_name.endswith('/'):
        log_folder_name += '/'
    _logger = logging.getLogger(name)
    level= logging.DEBUG
    _logger.setLevel(level)
    sh = logging.StreamHandler()
    _logger.addHandler(sh)

    date = datetime.today()
    folder = date.strftime('%Y-%m-%d')

    if 'logs' not in os.listdir(log_folder_name):
        os.mkdir(log_folder_name + 'logs')
    if folder not in os.listdir(log_folder_name + 'logs'):
        os.mkdir(log_folder_name + 'logs/' + folder)

    filename = f'logs/{folder}/{log_file_name}'

    logger_file_handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=1024*1024,
        backupCount=1,
        encoding='utf-8',
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_file_handler.setFormatter(formatter)
    _logger.addHandler(logger_file_handler)

    return _logger

