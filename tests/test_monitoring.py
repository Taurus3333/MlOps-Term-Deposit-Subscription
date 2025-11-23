"""
Tests for Monitoring Component
"""
import pytest
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


class TestMonitoring:
    """Test suite for monitoring functionality"""
    
    def test_data_drift_detection(self, sample_data):
        """Test data drift detection using KS test"""
        # Split data into reference and current
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]
        
        # Perform KS test on numerical column
        stat, p_value = ks_2samp(reference['age'], current['age'])
        
        assert 0 <= p_value <= 1
        assert stat >= 0
    
    def test_drift_threshold(self, sample_data):
        """Test drift threshold logic"""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]
        
        threshold = 0.05
        
        stat, p_value = ks_2samp(reference['balance'], current['balance'])
        
        drift_detected = p_value < threshold
        
        assert isinstance(drift_detected, (bool, np.bool_))
    
    def test_multiple_feature_drift(self, sample_data):
        """Test drift detection across multiple features"""
        reference = sample_data.iloc[:50]
        current = sample_data.iloc[50:]
        
        numerical_cols = ['age', 'balance', 'duration', 'campaign']
        drift_results = {}
        
        for col in numerical_cols:
            if col in reference.columns:
                stat, p_value = ks_2samp(reference[col], current[col])
                drift_results[col] = p_value
        
        assert len(drift_results) > 0
        assert all(0 <= p <= 1 for p in drift_results.values())
    
    def test_prediction_monitoring(self):
        """Test prediction distribution monitoring"""
        # Simulate predictions
        predictions = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        
        # Calculate prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        assert sum(distribution.values()) == 100
        assert all(count >= 0 for count in distribution.values())
    
    def test_model_performance_tracking(self):
        """Test model performance metrics tracking"""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.78,
            'f1_score': 0.80,
            'timestamp': pd.Timestamp.now()
        }
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_alert_generation(self):
        """Test alert generation logic"""
        current_accuracy = 0.75
        baseline_accuracy = 0.85
        threshold = 0.05
        
        performance_drop = baseline_accuracy - current_accuracy
        alert_triggered = performance_drop > threshold
        
        assert isinstance(alert_triggered, bool)
        assert alert_triggered == True  # Should trigger alert
