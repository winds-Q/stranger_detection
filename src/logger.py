import logging
import os
import sys
from datetime import datetime

_setup_done = False


def setup_logger(log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    global _setup_done
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if _setup_done:
        return root_logger
    _setup_done = True

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("stranger_detection_%Y%m%d.log")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_filename), encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger
