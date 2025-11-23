"""Model pusher component - push model to registry."""
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact
)

logger = get_logger(__name__)


class ModelPusher:
    """Push accepted model to registry."""
    
    def __init__(
        self,
        config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ):
        self.config = config
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_registry_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_model_version(self):
        """Generate model version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        f1_score = self.model_evaluation_artifact.test_metrics['f1_score']
        version = f"v_{timestamp}_f1_{f1_score:.4f}"
        return version
    
    def push_model_to_registry(self):
        """Push model and artifacts to registry."""
        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logger.warning("Model not accepted. Skipping push to registry.")
                return None, False
            
            logger.info("Model accepted. Pushing to registry...")
            
            # Generate version
            version = self.generate_model_version()
            logger.info(f"Model version: {version}")
            
            # Create version directory
            version_dir = self.config.model_registry_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model
            model_dest = version_dir / "model.pkl"
            shutil.copy2(self.model_trainer_artifact.model_path, model_dest)
            logger.info(f"Model copied to: {model_dest}")
            
            # Copy preprocessor
            preprocessor_dest = version_dir / "preprocessor.pkl"
            shutil.copy2(self.data_transformation_artifact.preprocessor_path, preprocessor_dest)
            logger.info(f"Preprocessor copied to: {preprocessor_dest}")
            
            # Save metadata
            metadata = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_trainer_artifact.model_name,
                "test_metrics": self.model_evaluation_artifact.test_metrics,
                "train_metrics": self.model_trainer_artifact.train_metrics,
                "cv_scores": self.model_trainer_artifact.cv_scores,
                "parameters": self.model_trainer_artifact.best_params,
                "feature_names": self.data_transformation_artifact.feature_names,
                "target_name": self.data_transformation_artifact.target_name,
                "train_shape": self.data_transformation_artifact.train_shape,
                "test_shape": self.data_transformation_artifact.test_shape
            }
            
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Update latest symlink/marker
            latest_path = self.config.model_registry_dir / "latest.txt"
            with open(latest_path, 'w') as f:
                f.write(version)
            logger.info(f"Latest version marker updated: {latest_path}")
            
            return version_dir, True
            
        except Exception as e:
            raise CustomException(f"Failed to push model: {str(e)}", sys.exc_info())
    
    def save_push_report(self, registry_path, is_pushed, version):
        """Save model push report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "is_pushed": is_pushed,
                "model_version": version if version else "N/A",
                "registry_path": str(registry_path) if registry_path else "N/A",
                "model_accepted": self.model_evaluation_artifact.is_model_accepted,
                "test_f1_score": self.model_evaluation_artifact.test_metrics['f1_score'],
                "test_roc_auc": self.model_evaluation_artifact.test_metrics['roc_auc']
            }
            
            report_path = self.config.artifact_dir / "push_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Push report saved: {report_path}")
            
        except Exception as e:
            raise CustomException(f"Failed to save push report: {str(e)}", sys.exc_info())
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Execute model pusher pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Pusher")
            logger.info("=" * 60)
            
            # Push model to registry
            registry_path, is_pushed = self.push_model_to_registry()
            
            # Generate version
            version = self.generate_model_version() if is_pushed else "not_pushed"
            
            # Save report
            self.save_push_report(registry_path, is_pushed, version)
            
            # Create artifact
            artifact = ModelPusherArtifact(
                pushed_model_path=registry_path if registry_path else Path("N/A"),
                model_version=version,
                is_pushed=is_pushed,
                registry_path=self.config.model_registry_dir
            )
            
            if is_pushed:
                logger.info(f"[SUCCESS] Model pushed to registry: {registry_path}")
            else:
                logger.info("[SKIPPED] Model not pushed (not accepted)")
            
            logger.info("=" * 60)
            logger.info("Model Pusher Completed Successfully")
            logger.info("=" * 60)
            
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model pusher failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    # Run full pipeline
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.components.model_evaluation import ModelEvaluation
    from src.entity.config_entity import (
        DataTransformationConfig,
        ModelTrainerConfig,
        ModelEvaluationConfig
    )
    
    logger.info("Running data transformation...")
    transform_config = DataTransformationConfig()
    transformation = DataTransformation(transform_config)
    data_artifact = transformation.initiate_data_transformation()
    
    logger.info("\nRunning model training...")
    trainer_config = ModelTrainerConfig()
    trainer = ModelTrainer(trainer_config, data_artifact)
    model_artifact = trainer.initiate_model_training()
    
    logger.info("\nRunning model evaluation...")
    eval_config = ModelEvaluationConfig()
    evaluator = ModelEvaluation(eval_config, model_artifact, data_artifact)
    eval_artifact = evaluator.initiate_model_evaluation()
    
    logger.info("\nRunning model pusher...")
    pusher_config = ModelPusherConfig()
    pusher = ModelPusher(pusher_config, eval_artifact, model_artifact, data_artifact)
    pusher_artifact = pusher.initiate_model_pusher()
    
    print(f"\n{'='*60}")
    print("MODEL PUSHER SUMMARY")
    print(f"{'='*60}")
    print(f"Model Pushed: {'YES' if pusher_artifact.is_pushed else 'NO'}")
    print(f"Version: {pusher_artifact.model_version}")
    print(f"Registry Path: {pusher_artifact.registry_path}")
    if pusher_artifact.is_pushed:
        print(f"Model Location: {pusher_artifact.pushed_model_path}")
    print(f"{'='*60}")
