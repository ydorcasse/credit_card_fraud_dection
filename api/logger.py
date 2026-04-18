import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def setup_logger():
    """Configure application logger with daily file rotation and console output."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("fraud_api")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    # Daily rotating file handler — creates logs/api_YYYY-MM-DD.log
    log_file = os.path.join(LOG_DIR, "api.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=90,
        utc=True,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = lambda name: name.replace(".log.", "_") + ".log"

    # Structured log format: timestamp | level | IP | message
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Console handler for Docker / stdout visibility
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
