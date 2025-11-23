# Bank Marketing ML System

Production MLOps system for term deposit prediction with PySpark, MLflow, FastAPI, Docker, AWS deployment, and Airflow orchestration.

---

## Prerequisites

- Python 3.10+
- Java 11 or 17 (for PySpark)
- Docker (optional)
- AWS Account (for deployment)

---

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd bank-marketing-prediction
python -m venv .venv

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Java Setup (Required)

**Windows:**
```cmd
# Download JDK 11 from https://adoptium.net/
# Install and set environment variables:
setx JAVA_HOME "C:\Program Files\Java\jdk-11"
setx PATH "%PATH%;%JAVA_HOME%\bin"
```

**Mac/Linux:**
```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk

# Mac
brew install openjdk@11

# Verify
java -version
```

---

## Run ML Pipeline

```bash
python -m src.run_pipeline
```

**What happens:**
1. Data ingestion (PySpark ETL)
2. Data validation
3. Feature transformation
4. Model training (MLflow)
5. Model evaluation
6. Model registry

---

## Run API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Access: http://localhost:8000

---

## Docker

### Build and Run

```bash
# Build
docker build -t bank-marketing-api .

# Run
docker run -p 8000:8000 bank-marketing-api
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop
docker-compose down
```

**Services:**
- API: http://localhost:8000
- MLflow: http://localhost:5000

---

## Airflow (Continuous Training)

### Option 1: Docker (Easiest)

```bash
cd airflow
docker-compose up -d
```

Access: http://localhost:8080 (admin/admin)

### Option 2: Standalone

```bash
# Install
pip install apache-airflow==2.7.0

# Setup
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAG
mkdir -p airflow/dags
cp airflow/dags/bank_marketing_training_dag.py airflow/dags/

# Start (2 terminals)
airflow webserver --port 8080  # Terminal 1
airflow scheduler              # Terminal 2
```

**Trigger DAG:**
1. Open http://localhost:8080
2. Toggle DAG to "On"
3. Click "Trigger DAG"

---

## AWS Deployment

See **CICD.md** for complete guide.

**Quick steps:**
1. Create AWS account + IAM user
2. Install AWS CLI: `aws configure`
3. Create ECR repository
4. Create ECS cluster + service
5. Add GitHub secrets
6. Push to main → auto-deploy

**Live endpoint:** http://44.210.237.18:8000

---

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# View report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

---

## Project Structure

```
├── src/
│   ├── components/      # 6 pipeline components
│   ├── api.py          # FastAPI app
│   └── run_pipeline.py # Orchestrator
├── airflow/
│   └── dags/           # Airflow DAG
├── tests/              # 51 tests
├── data/               # Raw data
├── mlruns/             # MLflow experiments
├── Dockerfile          # Container
└── docker-compose.yml  # Multi-service
```

---

## Quick Commands

```bash
# Run pipeline
python -m src.run_pipeline

# Run API
uvicorn src.api:app --reload

# Run tests
pytest tests/ -v

# Docker
docker-compose up -d

# Airflow
cd airflow && docker-compose up -d
```

---

## MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Access: http://localhost:5000

---

## Troubleshooting

**PySpark Error:**
- Set JAVA_HOME correctly
- Use Java 11 or 17
- Restart terminal

**Docker Error:**
- Start Docker Desktop
- Check ports 8000, 5000, 8080

**Import Error:**
- Activate virtual environment
- Run `pip install -r requirements.txt`

---

## What You Get

✅ PySpark ETL pipeline  
✅ MLflow experiment tracking  
✅ FastAPI REST API + Web UI  
✅ Docker containerization  
✅ AWS ECS deployment  
✅ GitHub Actions CI/CD  
✅ Airflow continuous training  
✅ 51 automated tests  

---

## Tech Stack

**ML:** Scikit-learn, XGBoost, LightGBM, Imbalanced-learn  
**Data:** PySpark, Pandas, Parquet  
**MLOps:** MLflow, Airflow  
**API:** FastAPI, Pydantic  
**Cloud:** AWS (ECS, ECR, CloudWatch)  
**DevOps:** Docker, GitHub Actions  

---

