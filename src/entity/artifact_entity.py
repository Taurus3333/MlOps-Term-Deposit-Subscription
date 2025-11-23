"""Artifact entities for pipeline component outputs."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class DataIngestionArtifact:
    """Output artifact from data ingestion component."""
    cleaned_data_path: Path
    row_count: int
    column_count: int
    metadata_path: Path
    timestamp: str = datetime.now().isoformat()


@dataclass
class DataValidationArtifact:
    """Output artifact from data validation component."""
    validation_status: str  # PASS, FAIL, WARNING
    validation_report_path: Path
    is_valid: bool
    timestamp: str = datetime.now().isoformat()


@dataclass
class DataTransformationArtifact:
    """Output artifact from data transformation component."""
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    feature_names: list
    target_name: str
    train_shape: tuple
    test_shape: tuple
    timestamp: str = datetime.now().isoformat()


@dataclass
class ModelTrainerArtifact:
    """Output artifact from model training component."""
    model_path: Path
    model_name: str
    train_metrics: Dict[str, float]
    cv_scores: Dict[str, float]
    best_params: Dict[str, Any]
    timestamp: str = datetime.now().isoformat()


@dataclass
class ModelEvaluationArtifact:
    """Output artifact from model evaluation component."""
    test_metrics: Dict[str, float]
    confusion_matrix_path: Path
    roc_curve_path: Path
    classification_report_path: Path
    is_model_accepted: bool
    timestamp: str = datetime.now().isoformat()


@dataclass
class ModelPusherArtifact:
    """Output artifact from model pusher component."""
    pushed_model_path: Path
    model_version: str
    is_pushed: bool
    registry_path: Path
    timestamp: str = datetime.now().isoformat()
