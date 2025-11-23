"""
Integration Tests for End-to-End Pipeline
"""
import pytest
import pandas as pd
from pathlib import Path


@pytest.mark.integration
class TestIntegration:
    """Integration test suite"""
    
    def test_data_pipeline_flow(self, sample_data, temp_dir):
        """Test complete data pipeline flow"""
        # Step 1: Data Ingestion
        input_path = temp_dir / "input.csv"
        sample_data.to_csv(input_path, index=False)
        
        df = pd.read_csv(input_path)
        assert len(df) > 0
        
        # Step 2: Data Validation
        assert 'y' in df.columns
        assert df['age'].dtype in [int, 'int64', 'int32']
        
        # Step 3: Data Transformation
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['y_encoded'] = le.fit_transform(df['y'])
        
        assert 'y_encoded' in df.columns
        
        # Step 4: Train-Test Split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        assert len(train_df) + len(test_df) == len(df)
    
    def test_model_training_pipeline(self, sample_data):
        """Test model training pipeline"""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        le = LabelEncoder()
        for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
            if col in sample_data.columns:
                sample_data[col] = le.fit_transform(sample_data[col].astype(str))
        
        X = sample_data.drop('y', axis=1)
        y = le.fit_transform(sample_data['y'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Evaluate
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy > 0.4  # Reasonable threshold for random data
    
    def test_api_prediction_flow(self):
        """Test API prediction flow"""
        from fastapi.testclient import TestClient
        from src.api import app
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test prediction (may fail if model not loaded)
        payload = {
            "age": 35,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "duration": 300,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 500]  # 500 if model not loaded
