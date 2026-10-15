"""
Logging utilities for experiments.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger for an experiment.

    Args:
        experiment_name: Name of the experiment (used in log filename)
        log_dir: Directory to save log files (default: "logs")
        log_level: Logging level (default: logging.INFO)
        console_output: Whether to also output to console (default: True)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler - create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = log_path / log_filename

    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_filepath}")

    return logger


def get_experiment_logger(experiment_name: str | None = None) -> logging.Logger:
    """
    Get an existing logger or create a default one.

    Args:
        experiment_name: Name of the experiment. If None, uses default name.

    Returns:
        Logger instance
    """
    if experiment_name is None:
        experiment_name = "experiment"

    logger = logging.getLogger(experiment_name)

    # If logger has no handlers, set up default
    if not logger.handlers:
        logger = setup_experiment_logger(experiment_name)

    return logger
