# Bank Marketing ML System - Interview Narrative

## Opening Context

I built a production-grade MLOps system that predicts whether a bank customer will subscribe to a term deposit. This isn't just an academic exercise — banks spend millions on marketing campaigns, and knowing which customers to target directly impacts ROI. A bad prediction doesn't just waste money; it damages customer relationships through irrelevant outreach.

## How I Built It - Step by Step

### Starting with Data Engineering, Not Models

I didn't jump straight into modeling. Real enterprise systems fail because of bad data pipelines, not bad algorithms. So I started with ETL using PySpark — not pandas. Why? Because banks deal with millions of records, and PySpark scales horizontally. I set up a Spark session, handled malformed CSVs with permissive mode, and wrote the cleaned data to Parquet format. Parquet is columnar, compressed, and optimized for analytics — exactly what downstream ML pipelines need.

The ETL wasn't a one-time script. I designed it to be repeatable and production-grade, with proper error handling and logging. Every run creates timestamped artifacts so we can trace back to any version of the data.

### Data Validation - The Safety Net

Before any model sees the data, it goes through automated validation. I created a schema.yaml that defines expected columns, data types, and value ranges. The validation component checks for missing values, schema compliance, and data drift using Kolmogorov-Smirnov tests. If the data doesn't pass validation, the pipeline stops. This protects us from training on corrupted data and prevents silent failures that cost weeks to debug.

### Modular MLOps Architecture

I broke the system into six modular components: ingestion, validation, transformation, training, evaluation, and model pusher. Each component produces an artifact that the next one consumes. This isn't just clean code — it's operational intelligence. When something breaks, I know exactly which stage failed. When auditors ask how a prediction was made, I can trace the entire lineage from raw data to model output.

Each component has its own configuration entity and artifact entity. This separation means I can swap out implementations without breaking the pipeline. Want to change the validation logic? Just update that component. Need a different model? Only the trainer changes.

### Experimentation Phase - Finding What Works

I ran experiments in a separate notebook using the cleaned Parquet data. I tested multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM. I also handled class imbalance using SMOTE and Tomek Links because the dataset is heavily skewed — most customers don't subscribe.

Here's the critical decision: I didn't optimize for accuracy. Accuracy is misleading when classes are imbalanced. Instead, I optimized for F1-score because the cost of false positives and false negatives is different. Missing a potential subscriber (false negative) means lost revenue. Annoying a customer with irrelevant offers (false positive) damages trust. F1-score balances both concerns.

I saved the best experiment results to results.json. This makes the pipeline deterministic — the training component knows exactly which model and hyperparameters to use. No guesswork, no manual intervention.

### MLflow Integration - Memory and Governance

Every training run logs to MLflow. I track metrics (accuracy, precision, recall, F1, ROC-AUC), parameters (hyperparameters, data versions), and artifacts (models, plots, feature importance). MLflow gives us experiment history and model registry. Only models that beat the current production baseline get promoted. This prevents regression — we never deploy a worse model.

The registry also provides versioning and stage transitions (staging, production, archived). This is how real ML teams operate. You don't just save a pickle file and hope for the best.

### FastAPI Serving + Business UI

The model is served through FastAPI. I built two interfaces: a REST API for programmatic access and a professional banking-style web UI for business users. Marketing analysts don't write code — they enter customer profiles into a form and get instant predictions with probability scores.

The API includes health checks, input validation using Pydantic, and proper error handling. It loads the production model from the registry on startup. If the model file is missing, it fails gracefully with clear error messages.

### Batch Inference - Real-World Workflow

Not all predictions happen in real-time. Banks often run nightly batch scoring for upcoming campaigns. I simulated this by building a batch inference component that processes CSV files and outputs predictions with confidence scores. This is how marketing teams actually consume ML — they upload a customer list and get back a scored list.

### Monitoring + Retraining Signals

Models decay. Customer behavior changes. Economic conditions shift. I built a monitoring system that detects data drift and performance degradation. It compares new data distributions against reference data using statistical tests. If drift exceeds a threshold or model performance drops below baseline, it triggers a retraining signal.

This prevents silent model decay — the most dangerous failure mode in production ML. You don't want to discover your model stopped working six months after it happened.

### CI/CD + Docker + GitHub Actions

The entire system is containerized with Docker. This eliminates "works on my machine" problems. The Dockerfile uses multi-stage builds for smaller images, runs as a non-root user for security, and includes health checks.

I set up a GitHub Actions pipeline with four stages: quality checks (linting, testing, SonarQube), build and push to ECR, deploy to ECS, and notifications. Every push to main triggers the pipeline. Tests must pass before deployment. This is how you ship reliable ML systems — with automation and guardrails.

### Orchestration - Airflow DAG for Production Workflow

I built two orchestration approaches. First, a simple Python orchestrator in `run_pipeline.py` that runs all components sequentially — good for development and testing. But for production, I created an Apache Airflow DAG.

The Airflow DAG defines the complete workflow as a directed acyclic graph with seven tasks: data ingestion, validation, transformation, training, evaluation, model pusher, and success notification. Each task is a PythonOperator that calls the corresponding component. Tasks communicate through XCom — Airflow's cross-communication system. For example, the ingestion task pushes the cleaned data path to XCom, and the validation task pulls it.

The DAG includes automatic retries, email alerts on failure, execution timeouts, and scheduling. I configured it to run daily at midnight, but you can adjust the schedule. The Airflow UI gives you visual monitoring — you see which tasks succeeded, which failed, and how long each took. You can view logs, re-run failed tasks, and backfill historical runs.

This is how real data teams operate. You don't manually run scripts — you define workflows as code, and Airflow handles scheduling, monitoring, and alerting. If a task fails at 3 AM, you get an email. If the validation fails, the pipeline stops before wasting compute on training. Everything is logged, versioned, and reproducible.

## Real Business Value

This system delivers tangible value:

**Lower Marketing Waste** - Target only high-probability customers, reducing cost per acquisition.

**Better Targeting** - Personalized outreach based on predicted likelihood, improving conversion rates.

**Faster Iteration** - Automated pipelines mean new models can be tested and deployed in hours, not weeks.

**Regulatory Compliance** - Full audit trail from data to prediction. Every decision is traceable and reproducible.

**Repeatability** - No manual steps. No tribal knowledge. The system runs the same way every time.

## Closing

So overall, the system I built isn't just a model — it's a full lifecycle machine learning platform that can adapt, scale, and operate like a real production environment. It handles data engineering, experimentation, training, deployment, monitoring, and retraining. It's designed the way FAANG companies and top financial institutions build ML systems — with automation, observability, and reliability baked in from day one.
