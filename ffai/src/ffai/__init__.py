# Package initialization

import logging
import colorlog

def get_logger(name, level=None):
    """Set up a colored logger"""
    root = colorlog.getLogger()

    if not root.handlers:  # Only add handler if it doesn't exist
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
        root.addHandler(handler)

    # Get logger for module
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    return logger
