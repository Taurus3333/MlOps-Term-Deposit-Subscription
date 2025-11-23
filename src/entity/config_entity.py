"""Configuration entities for pipeline components."""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from src.constants import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    SCHEMA_FILE,
    RESULTS_FILE,
    CURRENT_ARTIFACT_DIR
)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    raw_data_path: Path = RAW_DATA_FILE
    cleaned_data_path: Path = CLEANED_DATA_FILE
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "data_ingestion"


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    data_path: Path = CLEANED_DATA_FILE
    schema_path: Path = SCHEMA_FILE
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "data_validation"


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation component."""
    data_path: Path = CLEANED_DATA_FILE
    results_path: Path = RESULTS_FILE
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "data_transformation"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelTrainerConfig:
    """Configuration for model training component."""
    results_path: Path = RESULTS_FILE
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "model_trainer"
    random_state: int = 42
    cv_folds: int = 5


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component."""
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "model_evaluation"
    test_size: float = 0.2
    threshold: float = 0.5
    metrics: list = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']


@dataclass
class ModelPusherConfig:
    """Configuration for model pusher component."""
    artifact_dir: Path = CURRENT_ARTIFACT_DIR / "model_pusher"
    model_registry_dir: Path = Path("model_registry")
    model_name: str = "bank_marketing_model"
