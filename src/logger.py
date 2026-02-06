import logging
from pathlib import Path
from typing import Optional

def setup_logger(log_dir: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("rtod")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / (log_file or "detection.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger() -> logging.Logger:
    return logging.getLogger("rtod")
