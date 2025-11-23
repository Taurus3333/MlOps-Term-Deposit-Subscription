"""Model evaluation component."""
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact
)

logger = get_logger(__name__)


class ModelEvaluation:
    """Evaluate trained model on test set."""
    
    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ):
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load trained model."""
        try:
            logger.info(f"Loading model from: {self.model_trainer_artifact.model_path}")
            
            with open(self.model_trainer_artifact.model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            raise CustomException(f"Failed to load model: {str(e)}", sys.exc_info())
    
    def load_test_data(self):
        """Load test data."""
        try:
            logger.info(f"Loading test data from: {self.data_transformation_artifact.test_data_path}")
            
            test_df = pd.read_csv(self.data_transformation_artifact.test_data_path)
            X_test = test_df.drop(columns=['target'])
            y_test = test_df['target']
            
            logger.info(f"Test data loaded: {X_test.shape}")
            return X_test, y_test
            
        except Exception as e:
            raise CustomException(f"Failed to load test data: {str(e)}", sys.exc_info())
    
    def evaluate_metrics(self, model, X_test, y_test):
        """Calculate evaluation metrics."""
        try:
            logger.info("Calculating evaluation metrics...")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            logger.info("Test Set Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics, y_pred, y_pred_proba
            
        except Exception as e:
            raise CustomException(f"Metric calculation failed: {str(e)}", sys.exc_info())
    
    def generate_confusion_matrix(self, y_test, y_pred):
        """Generate and save confusion matrix."""
        try:
            logger.info("Generating confusion matrix...")
            
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title('Confusion Matrix - Test Set', fontweight='bold', fontsize=14)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Add percentages
            total = cm.sum()
            for i in range(2):
                for j in range(2):
                    plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/total*100:.1f}%)',
                            ha='center', va='center', fontsize=10, color='gray')
            
            plt.tight_layout()
            cm_path = self.config.artifact_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved: {cm_path}")
            return cm_path
            
        except Exception as e:
            raise CustomException(f"Confusion matrix generation failed: {str(e)}", sys.exc_info())
    
    def generate_roc_curve(self, y_test, y_pred_proba, roc_auc):
        """Generate and save ROC curve."""
        try:
            logger.info("Generating ROC curve...")
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve - Test Set', fontweight='bold', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            roc_path = self.config.artifact_dir / 'roc_curve.png'
            plt.savefig(roc_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved: {roc_path}")
            return roc_path
            
        except Exception as e:
            raise CustomException(f"ROC curve generation failed: {str(e)}", sys.exc_info())
    
    def generate_classification_report(self, y_test, y_pred):
        """Generate and save classification report."""
        try:
            logger.info("Generating classification report...")
            
            report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
            
            report_path = self.config.artifact_dir / 'classification_report.txt'
            with open(report_path, 'w') as f:
                f.write("Classification Report - Test Set\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)
            
            logger.info(f"Classification report saved: {report_path}")
            logger.info(f"\n{report}")
            
            return report_path
            
        except Exception as e:
            raise CustomException(f"Classification report generation failed: {str(e)}", sys.exc_info())
    
    def check_model_acceptance(self, metrics):
        """Check if model meets acceptance criteria."""
        try:
            logger.info("Checking model acceptance criteria...")
            
            # Acceptance criteria
            min_f1 = 0.30  # Minimum F1 score
            min_roc_auc = 0.70  # Minimum ROC-AUC
            
            is_accepted = (
                metrics['f1_score'] >= min_f1 and
                metrics['roc_auc'] >= min_roc_auc
            )
            
            if is_accepted:
                logger.info(f"[PASS] Model ACCEPTED - F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            else:
                logger.warning(f"[FAIL] Model REJECTED - F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.warning(f"Criteria: F1 >= {min_f1}, ROC-AUC >= {min_roc_auc}")
            
            return is_accepted
            
        except Exception as e:
            raise CustomException(f"Model acceptance check failed: {str(e)}", sys.exc_info())
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Execute model evaluation pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Evaluation")
            logger.info("=" * 60)
            
            # Load model and test data
            model = self.load_model()
            X_test, y_test = self.load_test_data()
            
            # Evaluate metrics
            test_metrics, y_pred, y_pred_proba = self.evaluate_metrics(model, X_test, y_test)
            
            # Generate visualizations and reports
            cm_path = self.generate_confusion_matrix(y_test, y_pred)
            roc_path = self.generate_roc_curve(y_test, y_pred_proba, test_metrics['roc_auc'])
            report_path = self.generate_classification_report(y_test, y_pred)
            
            # Check acceptance
            is_accepted = self.check_model_acceptance(test_metrics)
            
            # Create artifact
            artifact = ModelEvaluationArtifact(
                test_metrics=test_metrics,
                confusion_matrix_path=cm_path,
                roc_curve_path=roc_path,
                classification_report_path=report_path,
                is_model_accepted=is_accepted
            )
            
            logger.info("=" * 60)
            logger.info("Model Evaluation Completed Successfully")
            logger.info("=" * 60)
            
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model evaluation failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    # Run full pipeline
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
    
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
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"F1 Score: {eval_artifact.test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {eval_artifact.test_metrics['roc_auc']:.4f}")
    print(f"Precision: {eval_artifact.test_metrics['precision']:.4f}")
    print(f"Recall: {eval_artifact.test_metrics['recall']:.4f}")
    print(f"Model Accepted: {'YES' if eval_artifact.is_model_accepted else 'NO'}")
    print(f"{'='*60}")
