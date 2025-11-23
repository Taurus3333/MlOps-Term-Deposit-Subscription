"""
Master Pipeline Orchestrator for Bank Marketing ML System.

This is the main entry point that orchestrates all 6 components:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Pusher

Usage:
    python -m src.run_pipeline
"""
import sys
from datetime import datetime
from pathlib import Path

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException

# Import all components
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

# Import all configs
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

logger = get_logger(__name__)


class TrainingPipeline:
    """Master training pipeline orchestrator."""
    
    def __init__(self):
        self.start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("BANK MARKETING ML PIPELINE - TRAINING MODE")
        logger.info("=" * 80)
        logger.info(f"Pipeline started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_data_ingestion(self):
        """Step 1: Data Ingestion."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1/6: DATA INGESTION")
            logger.info("=" * 80)
            
            config = DataIngestionConfig()
            ingestion = DataIngestion()
            artifact = ingestion.run_etl()
            
            logger.info("[PASS] Data Ingestion completed successfully")
            return artifact
            
        except Exception as e:
            raise CustomException(f"Data Ingestion failed: {str(e)}", sys.exc_info())
    
    def run_data_validation(self):
        """Step 2: Data Validation."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2/6: DATA VALIDATION")
            logger.info("=" * 80)
            
            config = DataValidationConfig()
            validation = DataValidation()
            status = validation.run_validation()
            
            if status == "FAIL":
                logger.error("[FAIL] Data Validation failed. Check validation report.")
                raise Exception("Data validation failed. Cannot proceed with training.")
            
            logger.info(f"[PASS] Data Validation completed with status: {status}")
            return status
            
        except Exception as e:
            raise CustomException(f"Data Validation failed: {str(e)}", sys.exc_info())
    
    def run_data_transformation(self):
        """Step 3: Data Transformation."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3/6: DATA TRANSFORMATION")
            logger.info("=" * 80)
            
            config = DataTransformationConfig()
            transformation = DataTransformation(config)
            artifact = transformation.initiate_data_transformation()
            
            logger.info("[PASS] Data Transformation completed successfully")
            return artifact
            
        except Exception as e:
            raise CustomException(f"Data Transformation failed: {str(e)}", sys.exc_info())
    
    def run_model_training(self, data_transformation_artifact):
        """Step 4: Model Training."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4/6: MODEL TRAINING")
            logger.info("=" * 80)
            
            config = ModelTrainerConfig()
            trainer = ModelTrainer(config, data_transformation_artifact)
            artifact = trainer.initiate_model_training()
            
            logger.info("[PASS] Model Training completed successfully")
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model Training failed: {str(e)}", sys.exc_info())
    
    def run_model_evaluation(self, model_trainer_artifact, data_transformation_artifact):
        """Step 5: Model Evaluation."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5/6: MODEL EVALUATION")
            logger.info("=" * 80)
            
            config = ModelEvaluationConfig()
            evaluator = ModelEvaluation(config, model_trainer_artifact, data_transformation_artifact)
            artifact = evaluator.initiate_model_evaluation()
            
            logger.info("[PASS] Model Evaluation completed successfully")
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model Evaluation failed: {str(e)}", sys.exc_info())
    
    def run_model_pusher(self, model_evaluation_artifact, model_trainer_artifact, data_transformation_artifact):
        """Step 6: Model Pusher."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6/6: MODEL PUSHER")
            logger.info("=" * 80)
            
            config = ModelPusherConfig()
            pusher = ModelPusher(config, model_evaluation_artifact, model_trainer_artifact, data_transformation_artifact)
            artifact = pusher.initiate_model_pusher()
            
            logger.info("[PASS] Model Pusher completed successfully")
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model Pusher failed: {str(e)}", sys.exc_info())
    
    def run_pipeline(self):
        """Execute complete training pipeline."""
        try:
            # Step 1: Data Ingestion (Skip - already have cleaned data)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1/6: DATA INGESTION")
            logger.info("=" * 80)
            logger.info("[SKIP] Data already ingested and cleaned. Using existing cleaned data.")
            
            # Step 2: Data Validation
            validation_status = self.run_data_validation()
            
            # Step 3: Data Transformation
            transformation_artifact = self.run_data_transformation()
            
            # Step 4: Model Training
            trainer_artifact = self.run_model_training(transformation_artifact)
            
            # Step 5: Model Evaluation
            evaluation_artifact = self.run_model_evaluation(trainer_artifact, transformation_artifact)
            
            # Step 6: Model Pusher
            pusher_artifact = self.run_model_pusher(evaluation_artifact, trainer_artifact, transformation_artifact)
            
            # Pipeline summary
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            logger.info("")
            logger.info("Component Status:")
            logger.info("  [SKIP] 1. Data Ingestion (using existing cleaned data)")
            logger.info(f"  [PASS] 2. Data Validation ({validation_status})")
            logger.info("  [PASS] 3. Data Transformation")
            logger.info("  [PASS] 4. Model Training")
            logger.info("  [PASS] 5. Model Evaluation")
            logger.info(f"  [{'PASS' if pusher_artifact.is_pushed else 'SKIP'}] 6. Model Pusher")
            logger.info("")
            logger.info("Model Performance:")
            logger.info(f"  Model: {trainer_artifact.model_name}")
            logger.info(f"  Train F1: {trainer_artifact.train_metrics['f1_score']:.4f}")
            logger.info(f"  Test F1: {evaluation_artifact.test_metrics['f1_score']:.4f}")
            logger.info(f"  Test ROC-AUC: {evaluation_artifact.test_metrics['roc_auc']:.4f}")
            logger.info(f"  Model Accepted: {'YES' if evaluation_artifact.is_model_accepted else 'NO'}")
            logger.info(f"  Model Pushed: {'YES' if pusher_artifact.is_pushed else 'NO'}")
            if pusher_artifact.is_pushed:
                logger.info(f"  Model Version: {pusher_artifact.model_version}")
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return {
                "status": "SUCCESS",
                "duration_seconds": duration,
                "model_name": trainer_artifact.model_name,
                "test_f1_score": evaluation_artifact.test_metrics['f1_score'],
                "test_roc_auc": evaluation_artifact.test_metrics['roc_auc'],
                "model_accepted": evaluation_artifact.is_model_accepted,
                "model_pushed": pusher_artifact.is_pushed,
                "model_version": pusher_artifact.model_version if pusher_artifact.is_pushed else None
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            logger.error("\n" + "=" * 80)
            logger.error("PIPELINE FAILED!")
            logger.error("=" * 80)
            logger.error(f"Duration before failure: {duration:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 80)
            
            raise CustomException(f"Pipeline execution failed: {str(e)}", sys.exc_info())


def main():
    """Main entry point."""
    try:
        pipeline = TrainingPipeline()
        result = pipeline.run_pipeline()
        
        print("\n" + "=" * 80)
        print("PIPELINE RESULT")
        print("=" * 80)
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']:.2f}s")
        print(f"Model: {result['model_name']}")
        print(f"Test F1: {result['test_f1_score']:.4f}")
        print(f"Test ROC-AUC: {result['test_roc_auc']:.4f}")
        print(f"Model Pushed: {'YES' if result['model_pushed'] else 'NO'}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
