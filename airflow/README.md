# Airflow Continuous Training Pipeline

Production-ready Airflow DAG for automated model retraining.

## Overview

This DAG orchestrates all 6 ML pipeline components for continuous training:

```
data_ingestion → data_validation → data_transformation → model_training → model_evaluation → model_pusher → notification
```

## Features

✅ **Automated Retraining** - Runs daily at midnight  
✅ **Quality Gates** - Stops if validation fails  
✅ **Model Comparison** - Only promotes if better than baseline  
✅ **XCom Communication** - Tasks share artifacts  
✅ **Automatic Retries** - 2 attempts with 5-minute delay  
✅ **Email Alerts** - Notifications on failure  
✅ **Full Logging** - Complete audit trail  

## Quick Start

### Option 1: Standalone Airflow

```bash
# Install Airflow
pip install apache-airflow==2.7.0

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow  # Mac/Linux
$env:AIRFLOW_HOME = "$PWD\airflow"  # Windows PowerShell

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Copy DAG
mkdir -p airflow/dags
cp airflow/dags/bank_marketing_training_dag.py airflow/dags/

# Start Airflow (2 terminals)
# Terminal 1:
airflow webserver --port 8080

# Terminal 2:
airflow scheduler
```

### Option 2: Docker Compose

```bash
cd airflow
docker-compose up -d
```

## Access Airflow UI

Open browser: http://localhost:8080

Login:
- Username: `admin`
- Password: `admin`

## DAG Configuration

**Schedule:** Daily at midnight (`@daily`)  
**Retries:** 2 attempts  
**Retry Delay:** 5 minutes  
**Timeout:** 2 hours per task  
**Email Alerts:** Enabled on failure  

## Task Details

### 1. data_ingestion
- Runs PySpark ETL
- Cleans raw CSV data
- Outputs Parquet files
- Pushes cleaned data path to XCom

### 2. data_validation
- Checks schema compliance
- Detects data drift (KS tests)
- Enforces quality gates
- Fails pipeline if validation fails

### 3. data_transformation
- Engineers features
- Handles missing values
- Scales numerical features
- Encodes categorical variables

### 4. model_training
- Trains ML model
- Logs to MLflow
- Tracks metrics and parameters
- Saves model artifacts

### 5. model_evaluation
- Evaluates on test set
- Compares with production baseline
- Decides accept/reject
- Logs evaluation metrics

### 6. model_pusher
- Pushes model to registry (if accepted)
- Versions the model
- Updates production model
- Skips if model rejected

### 7. success_notification
- Logs pipeline summary
- Reports final status
- Sends notifications

## Monitoring

### View DAG Status

In Airflow UI:
1. Click on DAG name
2. See task status (green = success, red = failed)
3. Click tasks to view logs

### View Task Logs

1. Click on task
2. Click "Log" button
3. View execution details

### Check XCom Values

1. Click on task
2. Click "XCom" button
3. See shared data between tasks

## Triggering the DAG

### Manual Trigger

1. In Airflow UI, find DAG
2. Toggle to "On"
3. Click "Trigger DAG" button

### Via CLI

```bash
airflow dags trigger bank_marketing_continuous_training
```

### Scheduled Execution

DAG runs automatically daily at midnight (configurable in DAG file).

## Customization

### Change Schedule

Edit `bank_marketing_training_dag.py`:

```python
schedule_interval='@daily',  # Options: @hourly, @weekly, '0 9 * * *'
```

### Change Email Alerts

Edit `default_args`:

```python
'email': ['your-email@company.com'],
'email_on_failure': True,
```

### Adjust Retries

```python
'retries': 2,
'retry_delay': timedelta(minutes=5),
```

## Production Deployment

### AWS MWAA (Managed Airflow)

1. Upload DAG to S3 bucket
2. MWAA automatically picks it up
3. Configure environment variables
4. Set up IAM roles

### Kubernetes (Airflow on K8s)

1. Deploy Airflow Helm chart
2. Mount DAGs via ConfigMap or PVC
3. Configure resource limits
4. Set up monitoring

## Continuous Training Workflow

**Daily at Midnight:**
1. New data arrives in data directory
2. Airflow triggers DAG automatically
3. Pipeline runs all 6 components
4. If new model is better, it's promoted
5. Production model is updated
6. Team receives notification

**If Validation Fails:**
- Pipeline stops immediately
- No training happens
- Alert sent to team
- Manual intervention required

**If Model is Rejected:**
- Training completes
- Model is NOT promoted
- Production model unchanged
- Logged for analysis

## Benefits

✅ **Automated** - No manual intervention  
✅ **Reliable** - Automatic retries and error handling  
✅ **Auditable** - Full logging and lineage  
✅ **Safe** - Quality gates prevent bad models  
✅ **Scalable** - Can run on distributed infrastructure  
✅ **Observable** - Visual monitoring in UI  

## Interview Talking Points

> "I implemented Apache Airflow for continuous training. The DAG orchestrates all 6 pipeline components with proper dependencies and XCom communication. It runs daily, automatically retrains the model, and only promotes it if it beats the production baseline. This ensures the model stays current with changing customer behavior without manual intervention."

## Next Steps

1. Configure email SMTP settings
2. Set up monitoring dashboards
3. Implement data quality checks
4. Add parallel task execution
5. Integrate with data catalog

---

**You now have production-grade continuous training!** 🚀
