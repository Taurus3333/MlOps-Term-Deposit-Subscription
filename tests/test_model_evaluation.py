"""
Tests for Model Evaluation Component
"""
import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        
        return y_true, y_pred, y_proba
    
    def test_accuracy_calculation(self, sample_predictions):
        """Test accuracy metric calculation"""
        y_true, y_pred, _ = sample_predictions
        
        accuracy = accuracy_score(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, (float, np.floating))
    
    def test_precision_calculation(self, sample_predictions):
        """Test precision metric calculation"""
        y_true, y_pred, _ = sample_predictions
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        assert 0 <= precision <= 1
    
    def test_recall_calculation(self, sample_predictions):
        """Test recall metric calculation"""
        y_true, y_pred, _ = sample_predictions
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        assert 0 <= recall <= 1
    
    def test_f1_score_calculation(self, sample_predictions):
        """Test F1 score calculation"""
        y_true, y_pred, _ = sample_predictions
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        assert 0 <= f1 <= 1
    
    def test_roc_auc_calculation(self, sample_predictions):
        """Test ROC AUC score calculation"""
        y_true, _, y_proba = sample_predictions
        
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            assert 0 <= roc_auc <= 1
        except ValueError:
            # Handle case where only one class is present
            pass
    
    def test_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation"""
        y_true, y_pred, _ = sample_predictions
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
        assert all(cm.flatten() >= 0)
    
    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        assert accuracy == 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
