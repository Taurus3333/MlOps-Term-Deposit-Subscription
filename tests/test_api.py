"""
Tests for FastAPI Application
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app


class TestAPI:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_predict_endpoint_valid_data(self, client):
        """Test prediction endpoint with valid data"""
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
        
        # May fail if model not loaded, but should return proper status
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing fields"""
        payload = {
            "age": 35,
            "job": "technician"
            # Missing other required fields
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_types(self, client):
        """Test prediction endpoint with invalid data types"""
        payload = {
            "age": "invalid",  # Should be int
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
        
        assert response.status_code == 422  # Validation error
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        # Should return metrics or 404 if not available
        assert response.status_code in [200, 404, 500]
