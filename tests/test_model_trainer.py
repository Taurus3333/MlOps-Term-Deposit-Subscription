"""
Tests for Model Training Component
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


class TestModelTrainer:
    """Test suite for model training"""
    
    @pytest.fixture
    def train_test_data(self, sample_data):
        """Prepare train-test split"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
            if col in sample_data.columns:
                sample_data[col] = le.fit_transform(sample_data[col].astype(str))
        
        X = sample_data.drop('y', axis=1)
        y = le.fit_transform(sample_data['y'])
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_random_forest_training(self, train_test_data):
        """Test Random Forest model training"""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy > 0.5  # Should be better than random
        assert len(predictions) == len(y_test)
    
    def test_logistic_regression_training(self, train_test_data):
        """Test Logistic Regression model training"""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy > 0.3  # Lower threshold for random test data
        assert len(predictions) == len(y_test)
    
    def test_model_predictions(self, train_test_data):
        """Test model prediction functionality"""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test single prediction
        single_pred = model.predict(X_test.iloc[[0]])
        assert len(single_pred) == 1
        assert single_pred[0] in [0, 1]
    
    def test_model_metrics(self, train_test_data):
        """Test model evaluation metrics"""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, predictions, average='binary', zero_division=0)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_feature_importance(self, train_test_data):
        """Test feature importance extraction"""
        X_train, X_test, y_train, y_test = train_test_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        
        assert len(importances) == X_train.shape[1]
        assert all(imp >= 0 for imp in importances)
        assert abs(sum(importances) - 1.0) < 0.01  # Should sum to 1
