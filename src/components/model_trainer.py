"""Model training component using experiment results."""
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.constants import RESULTS_FILE
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.utils.mlflow_utils import MLflowManager

logger = get_logger(__name__)


class ModelTrainer:
    """Train model using experiment results."""
    
    def __init__(self, config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_manager = MLflowManager()
    
    def load_data(self):
        """Load transformed train and test data."""
        try:
            logger.info("Loading transformed data...")
            
            train_df = pd.read_csv(self.data_transformation_artifact.train_data_path)
            test_df = pd.read_csv(self.data_transformation_artifact.test_data_path)
            
            X_train = train_df.drop(columns=['target'])
            y_train = train_df['target']
            X_test = test_df.drop(columns=['target'])
            y_test = test_df['target']
            
            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise CustomException(f"Failed to load data: {str(e)}", sys.exc_info())
    
    def load_experiment_results(self):
        """Load experiment results."""
        try:
            logger.info(f"Loading experiment results from: {self.config.results_path}")
            
            with open(self.config.results_path, 'r') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            raise CustomException(f"Failed to load results: {str(e)}", sys.exc_info())
    
    def get_model_from_results(self, results):
        """Initialize model from experiment results."""
        try:
            model_name = results['best_model']['name']
            params = results['best_model']['parameters']
            
            logger.info(f"Initializing model: {model_name}")
            logger.info(f"Parameters: {params}")
            
            # Convert string parameters back to appropriate types
            model_params = {}
            for key, value in params.items():
                if value == 'None':
                    model_params[key] = None
                elif value.isdigit():
                    model_params[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    model_params[key] = float(value)
                else:
                    model_params[key] = value
            
            # Initialize model based on name
            if 'Random Forest' in model_name:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=self.config.random_state, **model_params)
            
            elif 'LightGBM' in model_name:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(random_state=self.config.random_state, verbose=-1, **model_params)
            
            elif 'XGBoost' in model_name:
                import xgboost as xgb
                model = xgb.XGBClassifier(random_state=self.config.random_state, eval_metric='logloss', **model_params)
            
            elif 'Gradient Boosting' in model_name:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(random_state=self.config.random_state, **model_params)
            
            elif 'Logistic Regression' in model_name:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=self.config.random_state, max_iter=1000, **model_params)
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            return model, model_name
            
        except Exception as e:
            raise CustomException(f"Failed to initialize model: {str(e)}", sys.exc_info())
    
    def train_model(self, model, X_train, y_train):
        """Train the model."""
        try:
            logger.info("Training model...")
            
            model.fit(X_train, y_train)
            
            logger.info("Model training completed")
            return model
            
        except Exception as e:
            raise CustomException(f"Model training failed: {str(e)}", sys.exc_info())
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model performance."""
        try:
            logger.info("Evaluating model...")
            
            # Train predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            
            train_metrics = {
                'accuracy': float(accuracy_score(y_train, y_train_pred)),
                'precision': float(precision_score(y_train, y_train_pred, zero_division=0)),
                'recall': float(recall_score(y_train, y_train_pred, zero_division=0)),
                'f1_score': float(f1_score(y_train, y_train_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_train, y_train_proba))
            }
            
            # Test predictions
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            test_metrics = {
                'accuracy': float(accuracy_score(y_test, y_test_pred)),
                'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_test_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_test_proba))
            }
            
            logger.info(f"Train Metrics - F1: {train_metrics['f1_score']:.4f}, ROC-AUC: {train_metrics['roc_auc']:.4f}")
            logger.info(f"Test Metrics - F1: {test_metrics['f1_score']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")
            
            return train_metrics, test_metrics
            
        except Exception as e:
            raise CustomException(f"Model evaluation failed: {str(e)}", sys.exc_info())
    
    def cross_validate_model(self, model, X_train, y_train):
        """Perform cross-validation."""
        try:
            logger.info(f"Performing {self.config.cv_folds}-fold cross-validation...")
            
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='f1',
                n_jobs=-1
            )
            
            cv_results = {
                'f1_mean': float(cv_scores.mean()),
                'f1_std': float(cv_scores.std()),
                'f1_scores': [float(score) for score in cv_scores]
            }
            
            logger.info(f"CV F1 Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            
            return cv_results
            
        except Exception as e:
            raise CustomException(f"Cross-validation failed: {str(e)}", sys.exc_info())
    
    def save_model(self, model, model_name):
        """Save trained model."""
        try:
            logger.info("Saving model...")
            
            model_path = self.config.artifact_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved: {model_path}")
            return model_path
            
        except Exception as e:
            raise CustomException(f"Failed to save model: {str(e)}", sys.exc_info())
    
    def initiate_model_training(self) -> ModelTrainerArtifact:
        """Execute model training pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Training with MLflow")
            logger.info("=" * 60)
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Load experiment results
            results = self.load_experiment_results()
            
            # Get model from results
            model, model_name = self.get_model_from_results(results)
            
            # Start MLflow run
            with self.mlflow_manager.start_run(run_name=f"{model_name}_training"):
                
                # Log parameters
                self.mlflow_manager.log_params(results['best_model']['parameters'])
                self.mlflow_manager.log_params({
                    'sampling_method': results['best_model']['sampling_method'],
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
                
                # Set tags
                self.mlflow_manager.set_tags({
                    'model_name': model_name,
                    'sampling_method': results['best_model']['sampling_method'],
                    'stage': 'training'
                })
                
                # Train model
                trained_model = self.train_model(model, X_train, y_train)
                
                # Evaluate model
                train_metrics, test_metrics = self.evaluate_model(
                    trained_model, X_train, y_train, X_test, y_test
                )
                
                # Log train metrics
                self.mlflow_manager.log_metrics({
                    f'train_{k}': v for k, v in train_metrics.items()
                })
                
                # Log test metrics
                self.mlflow_manager.log_metrics({
                    f'test_{k}': v for k, v in test_metrics.items()
                })
                
                # Cross-validation
                cv_scores = self.cross_validate_model(trained_model, X_train, y_train)
                self.mlflow_manager.log_metrics({
                    'cv_f1_mean': cv_scores['f1_mean'],
                    'cv_f1_std': cv_scores['f1_std']
                })
                
                # Save model locally
                model_path = self.save_model(trained_model, model_name)
                
                # Log model to MLflow
                self.mlflow_manager.log_model(trained_model, "model")
                
                # Log artifacts
                self.mlflow_manager.log_artifact(model_path)
                
                logger.info("Model and metrics logged to MLflow")
            
            # Create artifact
            artifact = ModelTrainerArtifact(
                model_path=model_path,
                model_name=model_name,
                train_metrics=train_metrics,
                cv_scores=cv_scores,
                best_params=results['best_model']['parameters']
            )
            
            logger.info("=" * 60)
            logger.info("Model Training Completed Successfully")
            logger.info("=" * 60)
            
            return artifact
            
        except Exception as e:
            raise CustomException(f"Model training failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    # First run data transformation to get artifact
    from src.components.data_transformation import DataTransformation
    from src.entity.config_entity import DataTransformationConfig
    
    logger.info("Running data transformation first...")
    transform_config = DataTransformationConfig()
    transformation = DataTransformation(transform_config)
    data_artifact = transformation.initiate_data_transformation()
    
    # Now run model training with the artifact
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config, data_artifact)
    artifact = trainer.initiate_model_training()
    
    print(f"\n{'='*60}")
    print("MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {artifact.model_name}")
    print(f"Model Path: {artifact.model_path}")
    print(f"Train F1: {artifact.train_metrics['f1_score']:.4f}")
    print(f"CV F1: {artifact.cv_scores['f1_mean']:.4f} ± {artifact.cv_scores['f1_std']:.4f}")
    print(f"{'='*60}")
