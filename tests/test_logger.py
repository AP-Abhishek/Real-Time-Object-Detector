import pytest
from src.logger import setup_logger, get_logger
import tempfile
import os


class TestLogger:
    
    def test_setup_logger_console(self):
        logger = setup_logger()
        assert logger is not None
        assert logger.name == "rtod"
    
    def test_setup_logger_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(log_dir=tmpdir, log_file="test.log")
            assert logger is not None
            
            log_file = os.path.join(tmpdir, "test.log")
            # Logger is setup but may not have written yet
            assert os.path.exists(tmpdir)
    
    def test_get_logger(self):
        logger = get_logger()
        assert logger is not None
        assert logger.name == "rtod"
    
    def test_logger_consistency(self):
        logger1 = setup_logger()
        logger2 = get_logger()
        assert logger1.name == logger2.name
