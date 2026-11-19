# src/utils/logger.py
import logging
import sys
from .config import load_config

config = load_config()
log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format=config["logging"]["format"],
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)
