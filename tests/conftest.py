"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def sample_data():
    """Create sample bank marketing data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'job': np.random.choice(['admin.', 'technician', 'services', 'management'], n_samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], n_samples),
        'default': np.random.choice(['yes', 'no'], n_samples),
        'balance': np.random.randint(-5000, 50000, n_samples),
        'housing': np.random.choice(['yes', 'no'], n_samples),
        'loan': np.random.choice(['yes', 'no'], n_samples),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_samples),
        'day': np.random.randint(1, 31, n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n_samples),
        'duration': np.random.randint(0, 3000, n_samples),
        'campaign': np.random.randint(1, 50, n_samples),
        'pdays': np.random.randint(-1, 500, n_samples),
        'previous': np.random.randint(0, 40, n_samples),
        'poutcome': np.random.choice(['unknown', 'failure', 'success', 'other'], n_samples),
        'y': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test artifacts"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_csv_file(sample_data, temp_dir):
    """Create a temporary CSV file with sample data"""
    csv_path = temp_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_config(temp_dir):
    """Create mock configuration for testing"""
    return {
        'data_ingestion': {
            'source_path': str(temp_dir / "source"),
            'destination_path': str(temp_dir / "destination")
        },
        'data_validation': {
            'schema_path': str(temp_dir / "schema.yaml"),
            'drift_threshold': 0.05
        },
        'model_trainer': {
            'model_path': str(temp_dir / "models"),
            'experiment_name': "test_experiment"
        }
    }
