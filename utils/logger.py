import sys

from loguru import logger

from .paths import get_exp_dir


# Clean up default logger
logger.remove()


def setup_logger(save_dir=get_exp_dir() / "logs", level="INFO", file_level=None):
    """Configure the logger."""
    console_msg = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
        " | <level>{level: <8}</level>"
        " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>: "
        "<level>{message}</level>"
        # " - \n<level>{message}</level>"
    )
    logfile_msg = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8}"
        " | {name}:{function}:{line}: {message}"
        # " | {name}:{function}:{line}: \n{message}"
    )

    # Console logging
    logger.add(
        sys.stderr,
        format=console_msg,
        level=level,
        colorize=True
    )
    # File logging
    logger.add(
        f"{save_dir}/{{time}}.log",
        format=logfile_msg,
        level="DEBUG" if file_level is None else file_level,
        rotation="10 MB",
        retention="1 month"
    )

    return logger


logger = setup_logger()
