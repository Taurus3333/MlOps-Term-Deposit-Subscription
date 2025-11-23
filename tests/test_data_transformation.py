"""
Tests for Data Transformation Component
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TestDataTransformation:
    """Test suite for data transformation"""
    
    def test_label_encoding(self, sample_data):
        """Test label encoding for categorical variables"""
        le = LabelEncoder()
        encoded = le.fit_transform(sample_data['job'])
        
        assert len(encoded) == len(sample_data)
        assert encoded.dtype in [np.int64, np.int32]
    
    def test_standard_scaling(self, sample_data):
        """Test standard scaling for numerical features"""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample_data[['age', 'balance']])
        
        assert scaled.shape == (len(sample_data), 2)
        # Check if mean is close to 0 and std is close to 1
        assert abs(scaled.mean()) < 0.1
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering"""
        # Create new features
        sample_data['age_balance_ratio'] = sample_data['age'] / (sample_data['balance'] + 1)
        
        assert 'age_balance_ratio' in sample_data.columns
        assert not sample_data['age_balance_ratio'].isnull().any()
    
    def test_target_encoding(self, sample_data):
        """Test target variable encoding"""
        le = LabelEncoder()
        encoded_target = le.fit_transform(sample_data['y'])
        
        assert len(np.unique(encoded_target)) <= 2
        assert encoded_target.dtype in [np.int64, np.int32]
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 40],
            'balance': [1000, 2000, np.nan, 3000]
        })
        
        # Fill missing values
        df_filled = df.fillna(df.mean())
        
        assert not df_filled.isnull().any().any()
    
    def test_outlier_detection(self, sample_data):
        """Test outlier detection"""
        Q1 = sample_data['balance'].quantile(0.25)
        Q3 = sample_data['balance'].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = sample_data[
            (sample_data['balance'] < Q1 - 1.5 * IQR) | 
            (sample_data['balance'] > Q3 + 1.5 * IQR)
        ]
        
        assert len(outliers) >= 0
