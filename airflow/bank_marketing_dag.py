"""
Airflow DAG for Bank Marketing ML Pipeline
Orchestrates all 6 components: Ingestion -> Validation -> Transformation -> Training -> Evaluation -> Pusher
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
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
    """Task 1: Data Ingestion using PySpark"""
    logger.info("Starting Data Ingestion task...")
    
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()
    
    # Push artifact path to XCom for downstream tasks
    context['ti'].xcom_push(key='ingestion_artifact', value=str(artifact.cleaned_data_path))
    
    logger.info(f"Data Ingestion completed. Artifact: {artifact.cleaned_data_path}")
    return str(artifact.cleaned_data_path)


def run_data_validation(**context):
    """Task 2: Data Validation with Schema and Drift Checks"""
    logger.info("Starting Data Validation task...")
    
    # Pull artifact from previous task
    ingestion_artifact = context['ti'].xcom_pull(key='ingestion_artifact', task_ids='data_ingestion')
    logger.info(f"Using data from: {ingestion_artifact}")
    
    config = DataValidationConfig()
    validation = DataValidation(config)
    artifact = validation.initiate_data_validation()
    
    # Check validation status
    if artifact.validation_status == "FAIL":
        raise ValueError("Data validation failed! Check validation report.")
    
    context['ti'].xcom_push(key='validation_status', value=artifact.validation_status)
    
    logger.info(f"Data Validation completed. Status: {artifact.validation_status}")
    return artifact.validation_status


def run_data_transformation(**context):
    """Task 3: Feature Engineering and Transformation"""
    logger.info("Starting Data Transformation task...")
    
    # Check validation passed
    validation_status = context['ti'].xcom_pull(key='validation_status', task_ids='data_validation')
    if validation_status != "PASS":
        raise ValueError(f"Cannot proceed with transformation. Validation status: {validation_status}")
    
    config = DataTransformationConfig()
    transformation = DataTransformation(config)
    artifact = transformation.initiate_data_transformation()
    
    context['ti'].xcom_push(key='transformation_artifact', value=str(artifact.transformed_data_path))
    
    logger.info(f"Data Transformation completed. Artifact: {artifact.transformed_data_path}")
    return str(artifact.transformed_data_path)


def run_model_training(**context):
    """Task 4: Model Training with MLflow Tracking"""
    logger.info("Starting Model Training task...")
    
    # Pull transformation artifact
    transformation_artifact = context['ti'].xcom_pull(key='transformation_artifact', task_ids='data_transformation')
    logger.info(f"Using transformed data from: {transformation_artifact}")
    
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config)
    artifact = trainer.initiate_model_training()
    
    context['ti'].xcom_push(key='trained_model_path', value=str(artifact.trained_model_path))
    context['ti'].xcom_push(key='model_metrics', value=artifact.metrics)
    
    logger.info(f"Model Training completed. Model: {artifact.trained_model_path}")
    logger.info(f"Training Metrics: {artifact.metrics}")
    return str(artifact.trained_model_path)


def run_model_evaluation(**context):
    """Task 5: Model Evaluation and Comparison"""
    logger.info("Starting Model Evaluation task...")
    
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
        logger.warning("Model not accepted! Current production model performs better.")
        raise ValueError("Model evaluation failed. New model does not beat baseline.")
    
    context['ti'].xcom_push(key='evaluation_status', value='ACCEPTED')
    context['ti'].xcom_push(key='evaluation_metrics', value=artifact.metrics)
    
    logger.info(f"Model Evaluation completed. Status: ACCEPTED")
    logger.info(f"Evaluation Metrics: {artifact.metrics}")
    return 'ACCEPTED'


def run_model_pusher(**context):
    """Task 6: Push Model to Registry"""
    logger.info("Starting Model Pusher task...")
    
    # Check evaluation status
    evaluation_status = context['ti'].xcom_pull(key='evaluation_status', task_ids='model_evaluation')
    if evaluation_status != 'ACCEPTED':
        raise ValueError("Cannot push model. Evaluation status is not ACCEPTED.")
    
    config = ModelPusherConfig()
    pusher = ModelPusher(config)
    artifact = pusher.initiate_model_pusher()
    
    logger.info(f"Model Pusher completed. Model pushed to: {artifact.model_registry_path}")
    logger.info(f"Model version: {artifact.model_version}")
    return str(artifact.model_registry_path)


def send_success_notification(**context):
    """Task 7: Send Success Notification"""
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUCCESSFUL!")
    logger.info("=" * 80)
    
    # Get all artifacts from XCom
    ingestion_artifact = context['ti'].xcom_pull(key='ingestion_artifact', task_ids='data_ingestion')
    validation_status = context['ti'].xcom_pull(key='validation_status', task_ids='data_validation')
    transformation_artifact = context['ti'].xcom_pull(key='transformation_artifact', task_ids='data_transformation')
    trained_model_path = context['ti'].xcom_pull(key='trained_model_path', task_ids='model_training')
    evaluation_metrics = context['ti'].xcom_pull(key='evaluation_metrics', task_ids='model_evaluation')
    
    logger.info(f"Data Ingestion: {ingestion_artifact}")
    logger.info(f"Validation Status: {validation_status}")
    logger.info(f"Transformation: {transformation_artifact}")
    logger.info(f"Trained Model: {trained_model_path}")
    logger.info(f"Evaluation Metrics: {evaluation_metrics}")
    logger.info("=" * 80)
    
    return "SUCCESS"


# Define the DAG
dag = DAG(
    'bank_marketing_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for bank marketing prediction',
    schedule_interval='@daily',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'bank-marketing', 'production'],
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
