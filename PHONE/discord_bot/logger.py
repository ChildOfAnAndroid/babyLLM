# v1.9
"""
Centralized logging system for babyLLM Discord bot
Replaces scattered print statements with structured logging
"""
import logging
from datetime import datetime
from typing import Optional

class BabyLogger:
    def __init__(self, name: str = "BABYBOT"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with custom format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, tag: str, message: str, **kwargs):
        """Standard info logging"""
        self.logger.info(f"[{tag}] {message}", **kwargs)
    
    def debug(self, tag: str, message: str, **kwargs):
        """Debug level logging"""
        self.logger.debug(f"[{tag}] {message}", **kwargs)
    
    def warn(self, tag: str, message: str, **kwargs):
        """Warning level logging"""
        self.logger.warning(f"[{tag}] {message}", **kwargs)
    
    def error(self, tag: str, message: str, **kwargs):
        """Error level logging"""
        self.logger.error(f"[{tag}] {message}", **kwargs)
    
    def emergency(self, tag: str, message: str, **kwargs):
        """Critical emergency logging"""
        self.logger.critical(f"[{tag}] 🚨 EMERGENCY: {message}", **kwargs)

# Global logger instance
logger = BabyLogger()
