"""
Common utility functions for the Bank Marketing prediction system.
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from src.logging import get_logger
from src.exception import BankMarketingException

logger = get_logger(__name__)


def create_directories(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to directory
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    except Exception as e:
        raise BankMarketingException(f"Error creating directory {path}: {str(e)}", sys)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON to: {file_path}")
    except Exception as e:
        raise BankMarketingException(f"Error saving JSON to {file_path}: {str(e)}", sys)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from: {file_path}")
        return data
    except Exception as e:
        raise BankMarketingException(f"Error loading JSON from {file_path}: {str(e)}", sys)


def get_timestamp() -> str:
    """
    Get current timestamp string for artifact naming.
    
    Returns:
        Timestamp string in format: YYYY_MM_DD_HH_MM_SS
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
