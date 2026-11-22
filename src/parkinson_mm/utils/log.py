"""Journalisation homogène : console + fichier reports/logs/<étape>.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # évite de doubler les messages si on rappelle la fonction
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
        logger.addHandler(console)
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(FORMAT))
            logger.addHandler(file_handler)
    return logger
