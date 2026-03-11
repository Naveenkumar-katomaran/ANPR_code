import logging
from logging.handlers import RotatingFileHandler
import os
from app.config import *

def setup_logging(level="INFO"):
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )



    file_handler = RotatingFileHandler(
        f"logs/{LOG_FILE}", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console)

    # Visual separator on every startup
    logging.info("=" * 60)
    logging.info("  LPR SERVICE STARTING UP")
    logging.info("=" * 60)
