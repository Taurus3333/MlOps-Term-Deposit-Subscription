"""
Tests for Data Ingestion Component
"""
import pytest
import pandas as pd
from pathlib import Path
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


class TestDataIngestion:
    """Test suite for data ingestion"""
    
    def test_data_ingestion_config(self, temp_dir):
        """Test DataIngestion configuration"""
        config = DataIngestionConfig(
            raw_data_path=temp_dir / "source.csv",
            cleaned_data_path=temp_dir / "destination",
            artifact_dir=temp_dir / "artifacts"
        )
        
        assert config.raw_data_path == temp_dir / "source.csv"
        assert config.cleaned_data_path == temp_dir / "destination"
    
    def test_read_csv_data(self, sample_csv_file):
        """Test reading CSV data"""
        df = pd.read_csv(sample_csv_file)
        
        assert df is not None
        assert len(df) > 0
        assert 'y' in df.columns
    
    def test_data_split(self, sample_data):
        """Test train-test split functionality"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(sample_data, test_size=0.2, random_state=42)
        
        assert len(train_df) + len(test_df) == len(sample_data)
        assert len(train_df) > len(test_df)
    
    def test_missing_source_file(self, temp_dir):
        """Test handling of missing source file"""
        config = DataIngestionConfig(
            raw_data_path=temp_dir / "nonexistent.csv",
            cleaned_data_path=temp_dir / "destination",
            artifact_dir=temp_dir / "artifacts"
        )
        
        with pytest.raises(FileNotFoundError):
            pd.read_csv(config.raw_data_path)
    
    def test_data_columns(self, sample_data):
        """Test that required columns are present"""
        required_columns = ['age', 'job', 'marital', 'education', 'balance', 'y']
        
        for col in required_columns:
            assert col in sample_data.columns
