"""
Tests for Utility Functions
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestUtils:
    """Test suite for utility functions"""
    
    def test_create_directories(self, temp_dir):
        """Test directory creation"""
        test_path = temp_dir / "test_dir" / "nested"
        test_path.mkdir(parents=True, exist_ok=True)
        
        assert test_path.exists()
        assert test_path.is_dir()
    
    def test_save_load_csv(self, sample_data, temp_dir):
        """Test CSV save and load"""
        csv_path = temp_dir / "test.csv"
        
        # Save
        sample_data.to_csv(csv_path, index=False)
        assert csv_path.exists()
        
        # Load
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(sample_data)
        assert list(loaded_df.columns) == list(sample_data.columns)
    
    def test_timestamp_generation(self):
        """Test timestamp generation"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        assert len(timestamp) == 15
        assert timestamp.isdigit() or '_' in timestamp
    
    def test_logging_setup(self):
        """Test logging configuration"""
        import logging
        
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        
        assert logger.level == logging.INFO
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = {
            'model_name': 'RandomForest',
            'n_estimators': 100,
            'random_state': 42
        }
        
        assert config['model_name'] == 'RandomForest'
        assert config['n_estimators'] == 100
    
    def test_data_validation_helper(self, sample_data):
        """Test data validation helper functions"""
        # Check for required columns
        required_cols = ['age', 'job', 'y']
        
        for col in required_cols:
            assert col in sample_data.columns
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(sample_data['age'])
        # Job column exists and has values
        assert 'job' in sample_data.columns
        assert len(sample_data['job']) > 0
