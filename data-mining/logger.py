# logger.py

import logging
import os


def setup_logger(debug=False):
    # Create a logger
    logger = logging.getLogger('web_scraper')

    # Set the logging level
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


# Get debug mode from environment variable (default to False)
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Create and configure the logger
logger = setup_logger(DEBUG)
