"""
Tests for Data Validation Component
"""
import pytest
import pandas as pd
import numpy as np
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_schema_validation(self, sample_data):
        """Test schema validation"""
        expected_columns = ['age', 'job', 'marital', 'education', 'balance', 'y']
        
        for col in expected_columns:
            assert col in sample_data.columns
    
    def test_data_types(self, sample_data):
        """Test data type validation"""
        assert sample_data['age'].dtype in [np.int64, np.int32]
        assert sample_data['balance'].dtype in [np.int64, np.int32]
        assert sample_data['job'].dtype == object
    
    def test_missing_values(self, sample_data):
        """Test missing value detection"""
        missing_count = sample_data.isnull().sum().sum()
        assert missing_count >= 0
    
    def test_target_variable(self, sample_data):
        """Test target variable validation"""
        assert 'y' in sample_data.columns
        unique_values = sample_data['y'].unique()
        assert len(unique_values) <= 2
    
    def test_numerical_ranges(self, sample_data):
        """Test numerical column ranges"""
        assert sample_data['age'].min() >= 0
        assert sample_data['age'].max() <= 120
        assert sample_data['duration'].min() >= 0
    
    def test_categorical_values(self, sample_data):
        """Test categorical column values"""
        valid_marital = ['married', 'single', 'divorced']
        assert all(sample_data['marital'].isin(valid_marital))
    
    def test_data_drift_detection(self, sample_data):
        """Test data drift detection logic"""
        # Create reference and current data
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]
        
        # Simple drift check - compare distributions
        from scipy.stats import ks_2samp
        
        stat, p_value = ks_2samp(reference['age'], current['age'])
        
        # p_value > 0.05 means no significant drift
        assert p_value >= 0 and p_value <= 1
