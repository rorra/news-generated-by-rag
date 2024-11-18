import logging
import sys
from datetime import datetime


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    # Create file handler with formatting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f"logs/{name}_{timestamp}.log")
    file_handler.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
