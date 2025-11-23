# Bank Marketing Term Deposit Prediction

Production MLOps system for predicting term deposit subscriptions.

## Prerequisites

- Python 3.10+
- Java JDK 11 or 17 (for PySpark)
- Docker (optional)

## Installation

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

## Java Setup (Required for PySpark)

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

## Configuration

Copy `.env.example` to `.env` and update if needed:
```bash
cp .env.example .env
```

## Usage

### Run Full ML Pipeline

**Option 1: Simple Orchestrator**
```bash
python -m src.run_pipeline
```

**Option 2: Airflow Orchestration (Optional)**
```bash
# See airflow/QUICKSTART.md for setup
airflow dags trigger bank_marketing_ml_pipeline
```

This executes:
1. Data ingestion (PySpark ETL)
2. Data validation
3. Feature transformation
4. Model training (MLflow tracking)
5. Model evaluation
6. Model deployment

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

Services:
- API: http://localhost:8000
- MLflow: http://localhost:5000

## Makefile Commands

```bash
make setup          # Setup environment
make run-pipeline   # Run ML pipeline
make run-api        # Start API server
make test           # Run tests
make docker-build   # Build Docker image
make docker-up      # Start containers
make clean          # Clean artifacts
```

## Project Structure

```
├── src/
│   ├── components/      # Pipeline components
│   ├── api.py          # FastAPI application
│   └── run_pipeline.py # Pipeline orchestration
├── airflow/            # Airflow DAG (optional)
│   ├── bank_marketing_dag.py
│   └── QUICKSTART.md
├── data/               # Raw data
├── cleaned_data/       # Processed data
├── artifacts/          # Pipeline outputs
├── tests/              # Test suite
├── static/             # Web UI assets
├── templates/          # HTML templates
└── requirements.txt    # Dependencies
```

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

## Monitoring

View MLflow experiments:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

Access at: http://localhost:5000

## Troubleshooting

**PySpark Java Error:**
- Ensure JAVA_HOME is set correctly
- Use Java 11 or 17 (not Java 8 or 21)
- Restart terminal after setting environment variables

**Docker Issues:**
- Ensure Docker Desktop is running
- Check port 8000 and 5000 are available

**Import Errors:**
- Activate virtual environment
- Run `pip install -r requirements.txt`

## API Endpoints

- `GET /` - Web UI
- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Model metrics

