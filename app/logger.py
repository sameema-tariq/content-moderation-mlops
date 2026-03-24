"""Shared logger for the content moderation pipeline."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
