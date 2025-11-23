"""MLflow utilities for experiment tracking and model registry."""
import os
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.logging.custom_logger import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Manage MLflow tracking and registry operations."""
    
    def __init__(self, experiment_name="bank_marketing_prediction"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set tracking URI from environment or use default
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {str(e)}")
            raise
    
    def start_run(self, run_name=None):
        """Start MLflow run."""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        """Log parameters."""
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log params: {str(e)}")
    
    def log_metrics(self, metrics, step=None):
        """Log metrics."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_artifact(self, artifact_path):
        """Log artifact file."""
        try:
            mlflow.log_artifact(str(artifact_path))
            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
    
    def log_artifacts(self, artifact_dir):
        """Log artifact directory."""
        try:
            mlflow.log_artifacts(str(artifact_dir))
            logger.info(f"Logged artifacts from: {artifact_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")
    
    def log_model(self, model, artifact_path="model"):
        """Log sklearn model."""
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
    
    def set_tags(self, tags):
        """Set tags for the run."""
        try:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            logger.info(f"Set {len(tags)} tags in MLflow")
        except Exception as e:
            logger.error(f"Failed to set tags: {str(e)}")
    
    def register_model(self, model_uri, model_name):
        """Register model to MLflow Model Registry."""
        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name}, Version: {result.version}")
            return result
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def transition_model_stage(self, model_name, version, stage):
        """Transition model to a different stage (Staging/Production/Archived)."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {str(e)}")
            raise
    
    def get_latest_model_version(self, model_name, stage=None):
        """Get latest model version from registry."""
        try:
            client = mlflow.tracking.MlflowClient()
            if stage:
                versions = client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = client.get_latest_versions(model_name)
            
            if versions:
                latest = versions[0]
                logger.info(f"Latest model: {model_name} v{latest.version} ({latest.current_stage})")
                return latest
            else:
                logger.warning(f"No model versions found for: {model_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to get latest model: {str(e)}")
            return None
    
    def end_run(self):
        """End MLflow run."""
        mlflow.end_run()
