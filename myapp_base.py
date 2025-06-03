import os
from loguru import logger
from fastapi import FastAPI

def setup_loguru(logfile="logs/app.log"):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger.remove()
    logger.add(
        logfile,
        rotation="500MB",
        retention="7 days",
        level="INFO",
        format="{time} {level} {message}"
    )
    return logger

def create_app():
    return FastAPI()
