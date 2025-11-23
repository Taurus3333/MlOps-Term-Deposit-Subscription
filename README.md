# Bank Marketing Term Deposit Prediction

Production-grade MLOps system for predicting term deposit subscriptions with full CI/CD and AWS deployment.

## Overview

End-to-end machine learning pipeline that predicts whether a bank customer will subscribe to a term deposit. Built with enterprise MLOps best practices: PySpark ETL, MLflow tracking, FastAPI serving, automated monitoring, and CI/CD deployment to AWS.

## Prerequisites

- Python 3.10+
- Java JDK 11 or 17 (for PySpark)
- Docker (optional)
- AWS Account (for deployment)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd bank-marketing-prediction

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Java Setup (Required for PySpark)

**Windows:**
1. Download JDK 11 from https://adoptium.net/
2. Install to `C:\Program Files\Java\jdk-11`
3. Set environment variables:
```cmd
setx JAVA_HOME "C:\Program Files\Java\jdk-11"
setx PATH "%PATH%;%JAVA_HOME%\bin"
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk

# Mac
brew install openjdk@11

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Verify: `java -version`

### 3. Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

## Usage

### Run Full ML Pipeline

```bash
python -m src.run_pipeline
```

**Pipeline executes:**
1. Data Ingestion (PySpark ETL)
2. Data Validation (schema + drift detection)
3. Feature Transformation
4. Model Training (MLflow tracking)
5. Model Evaluation
6. Model Registry (promotion if better than baseline)

### Run API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Access UI: http://localhost:8000

### Run with Docker

```bash
# Build and start
docker-compose up -d

# Stop
docker-compose down
```

**Services:**
- API: http://localhost:8000
- MLflow: http://localhost:5000

## Project Structure

```
├── src/
│   ├── components/          # 6 pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── entity/              # Config & artifact entities
│   ├── utils/               # Utility functions
│   ├── logging/             # Custom logger
│   ├── exception/           # Custom exceptions
│   ├── monitoring/          # Drift detection
│   ├── api.py              # FastAPI application
│   └── run_pipeline.py     # Pipeline orchestration
├── tests/                   # Test suite (51 tests)
├── data/                    # Raw data
├── cleaned_data/            # Processed Parquet files
├── artifacts/               # Pipeline outputs
├── mlruns/                  # MLflow experiments
├── model_registry/          # Production models
├── static/                  # Web UI assets
├── templates/               # HTML templates
├── .github/workflows/       # CI/CD pipelines
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service setup
├── requirements.txt         # Dependencies
├── CICD.md                  # AWS deployment guide
└── README.md               # This file
```

## Key Features

### Data Engineering
- **PySpark ETL** - Scalable data processing
- **Parquet Storage** - Columnar, compressed format
- **Schema Validation** - Automated quality gates
- **Drift Detection** - KS tests for distribution shifts

### Machine Learning
- **4 Algorithms** - Logistic Regression, Random Forest, XGBoost, LightGBM
- **Imbalanced Data** - SMOTE + Tomek Links
- **F1-Score Optimization** - Business-aligned metrics
- **MLflow Tracking** - Experiment management

### Deployment
- **FastAPI** - High-performance REST API
- **Web UI** - Banking-style interface for business users
- **Batch Inference** - Campaign scoring
- **Real-time Predictions** - Instant results

### MLOps
- **6 Modular Components** - Ingestion → Validation → Transformation → Training → Evaluation → Pusher
- **Model Registry** - Automated promotion
- **Monitoring** - Drift detection + retraining triggers
- **Artifact Versioning** - Full lineage tracking

### CI/CD & Cloud
- **GitHub Actions** - Automated pipeline
- **Docker** - Containerization
- **AWS ECS Fargate** - Serverless deployment
- **Amazon ECR** - Docker registry
- **CloudWatch** - Centralized logging
- **Zero Downtime** - Rolling updates

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

**Test Coverage:**
- 51 automated tests
- Unit + Integration tests
- API endpoint tests
- ~75% code coverage

## Monitoring

### MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Access at: http://localhost:5000

### View Logs

```bash
# Local logs
tail -f logs/*.log

# Docker logs
docker-compose logs -f api
```

## CI/CD Pipeline

**Automated on every push to main:**

1. **Quality Checks**
   - Linting (flake8)
   - Code formatting (black)
   - 51 automated tests
   - Coverage reports

2. **Build & Push**
   - Docker image build
   - Vulnerability scanning
   - Push to Amazon ECR

3. **Deploy**
   - Update ECS task definition
   - Rolling deployment to Fargate
   - Health checks

4. **Notifications**
   - Deployment status alerts

## AWS Deployment

See **CICD.md** for complete step-by-step AWS deployment guide.

**Quick summary:**
1. Set up AWS account + IAM user
2. Create ECR repository
3. Create ECS cluster + service
4. Configure GitHub secrets
5. Push to main → automatic deployment

## API Endpoints

- `GET /` - Web UI
- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Model metrics

**Example prediction request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

## Makefile Commands

```bash
make setup          # Setup environment
make run-pipeline   # Run ML pipeline
make run-api        # Start API server
make test           # Run tests
make test-coverage  # Tests with coverage
make docker-build   # Build Docker image
make docker-up      # Start containers
make docker-down    # Stop containers
make clean          # Clean artifacts
```

## Technology Stack

**Languages & Frameworks:**
- Python 3.10
- PySpark
- FastAPI
- HTML/CSS/JavaScript

**ML & Data Science:**
- Scikit-learn
- XGBoost
- LightGBM
- Pandas, NumPy
- Imbalanced-learn

**MLOps:**
- MLflow (tracking + registry)
- Evidently (drift detection)

**DevOps & Cloud:**
- Docker & Docker Compose
- GitHub Actions
- AWS (ECS, ECR, CloudWatch)

**Testing:**
- Pytest
- Coverage.py

## Troubleshooting

**PySpark Java Error:**
- Ensure JAVA_HOME is set correctly
- Use Java 11 or 17 (not Java 8 or 21)
- Restart terminal after setting environment variables

**Docker Issues:**
- Ensure Docker Desktop is running
- Check ports 8000 and 5000 are available

**Import Errors:**
- Activate virtual environment
- Run `pip install -r requirements.txt`

**AWS Deployment Issues:**
- Check IAM permissions
- Verify GitHub secrets are set
- Review CloudWatch logs

## Project Highlights

✅ **Production-Grade** - Enterprise MLOps practices  
✅ **Scalable** - PySpark for big data processing  
✅ **Automated** - Full CI/CD pipeline  
✅ **Monitored** - Drift detection + retraining  
✅ **Tested** - 51 automated tests  
✅ **Deployed** - AWS ECS Fargate  
✅ **Documented** - Comprehensive guides  

## Business Impact

- **Improved Targeting** - 25-40% conversion vs 12% baseline
- **Cost Reduction** - Eliminate low-probability customers
- **Faster Iteration** - Hours instead of weeks
- **Compliance** - Full audit trail
- **Reliability** - Automated, repeatable process

## License

MIT

## Documentation

- **CICD.md** - Complete AWS deployment guide
- **TESTING.md** - Testing documentation
- **profile/** - Interview materials (pitch deck, narration, resume)

---

**Built with production MLOps best practices for enterprise de