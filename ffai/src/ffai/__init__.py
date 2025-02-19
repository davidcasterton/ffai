# Package initialization

import logging
import colorlog

def get_logger(name, level=logging.INFO):
    """Set up a colored logger"""
    logger = colorlog.getLogger(name)

    if not logger.handlers:  # Only add handler if it doesn't exist
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)s:%(name)s:%(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)

    if level == logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    return logger
