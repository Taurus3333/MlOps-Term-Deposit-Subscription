"""
Airflow DAG for Continuous Training - Bank Marketing ML Pipeline
Orchestrates all 6 components for automated model retraining
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from src.logging import logger


# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@bank.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def run_data_ingestion(**context):
    """Component 1: Data Ingestion using PySpark"""
    logger.info("=" * 80)
    logger.info("STARTING DATA INGESTION")
    logger.info("=" * 80)
    
    try:
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)
        artifact = ingestion.initiate_data_ingestion()
        
        # Push artifact path to XCom for downstream tasks
        context['ti'].xcom_push(key='cleaned_data_path', value=str(artifact.cleaned_data_path))
        
        logger.info(f"✓ Data Ingestion completed: {artifact.cleaned_data_path}")
        return str(artifact.cleaned_data_path)
    
    except Exception as e:
        logger.error(f"✗ Data Ingestion failed: {str(e)}")
        raise


def run_data_validation(**context):
    """Component 2: Data Validation with Schema and Drift Checks"""
    logger.info("=" * 80)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 80)
    
    try:
        # Pull artifact from previous task
        cleaned_data_path = context['ti'].xcom_pull(key='cleaned_data_path', task_ids='data_ingestion')
        logger.info(f"Using data from: {cleaned_data_path}")
        
        config = DataValidationConfig()
        validation = DataValidation(config)
        artifact = validation.initiate_data_validation()
        
        # Check validation status
        if artifact.validation_status == "FAIL":
            logger.error("✗ Data validation FAILED! Check validation report.")
            raise ValueError("Data validation failed! Cannot proceed with training.")
        
        context['ti'].xcom_push(key='validation_status', value=artifact.validation_status)
        
        logger.info(f"✓ Data Validation completed: {artifact.validation_status}")
        return artifact.validation_status
    
    except Exception as e:
        logger.error(f"✗ Data Validation failed: {str(e)}")
        raise


def run_data_transformation(**context):
    """Component 3: Feature Engineering and Transformation"""
    logger.info("=" * 80)
    logger.info("STARTING DATA TRANSFORMATION")
    logger.info("=" * 80)
    
    try:
        # Check validation passed
        validation_status = context['ti'].xcom_pull(key='validation_status', task_ids='data_validation')
        if validation_status != "PASS":
            raise ValueError(f"Cannot proceed. Validation status: {validation_status}")
        
        config = DataTransformationConfig()
        transformation = DataTransformation(config)
        artifact = transformation.initiate_data_transformation()
        
        context['ti'].xcom_push(key='transformed_data_path', value=str(artifact.transformed_data_path))
        
        logger.info(f"✓ Data Transformation completed: {artifact.transformed_data_path}")
        return str(artifact.transformed_data_path)
    
    except Exception as e:
        logger.error(f"✗ Data Transformation failed: {str(e)}")
        raise


def run_model_training(**context):
    """Component 4: Model Training with MLflow Tracking"""
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 80)
    
    try:
        # Pull transformation artifact
        transformed_data_path = context['ti'].xcom_pull(key='transformed_data_path', task_ids='data_transformation')
        logger.info(f"Using transformed data from: {transformed_data_path}")
        
        config = ModelTrainerConfig()
        trainer = ModelTrainer(config)
        artifact = trainer.initiate_model_training()
        
        context['ti'].xcom_push(key='trained_model_path', value=str(artifact.trained_model_path))
        context['ti'].xcom_push(key='model_metrics', value=artifact.metrics)
        
        logger.info(f"✓ Model Training completed: {artifact.trained_model_path}")
        logger.info(f"Training Metrics: {artifact.metrics}")
        return str(artifact.trained_model_path)
    
    except Exception as e:
        logger.error(f"✗ Model Training failed: {str(e)}")
        raise


def run_model_evaluation(**context):
    """Component 5: Model Evaluation and Comparison"""
    logger.info("=" * 80)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 80)
    
    try:
        # Pull training artifacts
        trained_model_path = context['ti'].xcom_pull(key='trained_model_path', task_ids='model_training')
        training_metrics = context['ti'].xcom_pull(key='model_metrics', task_ids='model_training')
        
        logger.info(f"Evaluating model: {trained_model_path}")
        logger.info(f"Training metrics: {training_metrics}")
        
        config = ModelEvaluationConfig()
        evaluation = ModelEvaluation(config)
        artifact = evaluation.initiate_model_evaluation()
        
        # Check if model is accepted
        if not artifact.is_model_accepted:
            logger.warning("⚠ Model NOT accepted! Current production model performs better.")
            logger.warning("Pipeline will continue but model will NOT be promoted.")
        
        context['ti'].xcom_push(key='is_model_accepted', value=artifact.is_model_accepted)
        context['ti'].xcom_push(key='evaluation_metrics', value=artifact.metrics)
        
        status = "ACCEPTED" if artifact.is_model_accepted else "REJECTED"
        logger.info(f"✓ Model Evaluation completed: {status}")
        logger.info(f"Evaluation Metrics: {artifact.metrics}")
        return status
    
    except Exception as e:
        logger.error(f"✗ Model Evaluation failed: {str(e)}")
        raise


def run_model_pusher(**context):
    """Component 6: Push Model to Registry (if accepted)"""
    logger.info("=" * 80)
    logger.info("STARTING MODEL PUSHER")
    logger.info("=" * 80)
    
    try:
        # Check evaluation status
        is_model_accepted = context['ti'].xcom_pull(key='is_model_accepted', task_ids='model_evaluation')
        
        if not is_model_accepted:
            logger.warning("⚠ Model was REJECTED. Skipping model push.")
            logger.info("Current production model remains unchanged.")
            return "SKIPPED"
        
        config = ModelPusherConfig()
        pusher = ModelPusher(config)
        artifact = pusher.initiate_model_pusher()
        
        logger.info(f"✓ Model Pusher completed: {artifact.model_registry_path}")
        logger.info(f"Model version: {artifact.model_version}")
        logger.info(f"Model is now in PRODUCTION!")
        return str(artifact.model_registry_path)
    
    except Exception as e:
        logger.error(f"✗ Model Pusher failed: {str(e)}")
        raise


def send_success_notification(**context):
    """Send Success Notification and Summary"""
    logger.info("=" * 80)
    logger.info("CONTINUOUS TRAINING PIPELINE COMPLETED!")
    logger.info("=" * 80)
    
    # Get all artifacts from XCom
    cleaned_data_path = context['ti'].xcom_pull(key='cleaned_data_path', task_ids='data_ingestion')
    validation_status = context['ti'].xcom_pull(key='validation_status', task_ids='data_validation')
    transformed_data_path = context['ti'].xcom_pull(key='transformed_data_path', task_ids='data_transformation')
    trained_model_path = context['ti'].xcom_pull(key='trained_model_path', task_ids='model_training')
    is_model_accepted = context['ti'].xcom_pull(key='is_model_accepted', task_ids='model_evaluation')
    evaluation_metrics = context['ti'].xcom_pull(key='evaluation_metrics', task_ids='model_evaluation')
    
    logger.info(f"Data Ingestion: {cleaned_data_path}")
    logger.info(f"Validation Status: {validation_status}")
    logger.info(f"Transformation: {transformed_data_path}")
    logger.info(f"Trained Model: {trained_model_path}")
    logger.info(f"Model Accepted: {is_model_accepted}")
    logger.info(f"Evaluation Metrics: {evaluation_metrics}")
    
    if is_model_accepted:
        logger.info("✓ NEW MODEL DEPLOYED TO PRODUCTION!")
    else:
        logger.info("⚠ Model not promoted. Production model unchanged.")
    
    logger.info("=" * 80)
    
    return "SUCCESS"


# Define the DAG
dag = DAG(
    'bank_marketing_continuous_training',
    default_args=default_args,
    description='Continuous training pipeline for bank marketing ML system',
    schedule_interval='@daily',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'bank-marketing', 'continuous-training', 'production'],
)


# Task 1: Data Ingestion
task_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    provide_context=True,
    dag=dag,
)

# Task 2: Data Validation
task_validation = PythonOperator(
    task_id='data_validation',
    python_callable=run_data_validation,
    provide_context=True,
    dag=dag,
)

# Task 3: Data Transformation
task_transformation = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    provide_context=True,
    dag=dag,
)

# Task 4: Model Training
task_training = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    provide_context=True,
    dag=dag,
)

# Task 5: Model Evaluation
task_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    provide_context=True,
    dag=dag,
)

# Task 6: Model Pusher
task_pusher = PythonOperator(
    task_id='model_pusher',
    python_callable=run_model_pusher,
    provide_context=True,
    dag=dag,
)

# Task 7: Success Notification
task_notification = PythonOperator(
    task_id='success_notification',
    python_callable=send_success_notification,
    provide_context=True,
    dag=dag,
)

# Define task dependencies (DAG structure)
task_ingestion >> task_validation >> task_transformation >> task_training >> task_evaluation >> task_pusher >> task_notification
