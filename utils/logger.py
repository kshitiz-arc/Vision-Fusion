"""
VisionFusion AI — Logging Utility
==================================
Centralized, color-coded logging with file rotation support.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """ANSI color-coded log formatter for terminal output."""

    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color   = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{self.BOLD}[{record.levelname}]{self.RESET} {message}"


def get_logger(name: str = "visionfusion", level: str = "INFO",
               log_file: str | None = None) -> logging.Logger:
    """
    Build and return a named logger.

    Parameters
    ----------
    name : str
        Logger namespace (typically the module ``__name__``).
    level : str
        Minimum severity level string (DEBUG / INFO / WARNING / ERROR).
    log_file : str | None
        Optional path for file-based logging (rotated daily).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger                 # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # — terminal handler -------------------------------------------------------
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        ColorFormatter("%(asctime)s  %(name)s  %(message)s",
                       datefmt="%H:%M:%S")
    )
    logger.addHandler(stream_handler)

    # — optional file handler --------------------------------------------------
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger
