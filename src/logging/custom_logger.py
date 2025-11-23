import logging
import os
from datetime import datetime
from pathlib import Path


class CustomLogger:
    """Production-grade logger with timestamp-based log files."""
    
    _loggers = {}
    
    @staticmethod
    def get_logger(name: str = "bank_marketing") -> logging.Logger:
        """Get or create a logger instance with file and console handlers."""
        
        if name in CustomLogger._loggers:
            return CustomLogger._loggers[name]
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{timestamp}.log"
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Cache logger
        CustomLogger._loggers[name] = logger
        
        logger.info(f"Logger initialized. Log file: {log_file}")
        
        return logger


# Convenience function
def get_logger(name: str = "bank_marketing") -> logging.Logger:
    """Get logger instance."""
    return CustomLogger.get_logger(name)
