# Bank Marketing ML System - Technical Pitch

## 1️⃣ CRUX OF THE PROBLEM

Banks spend millions on marketing campaigns to sell term deposits. The challenge isn't just predicting who will subscribe — it's making economically optimal decisions under uncertainty.

This is not a symmetric classification problem. False positives waste marketing budget and annoy customers with irrelevant offers. False negatives miss revenue opportunities from customers who would have subscribed. The cost structure is asymmetric, which means accuracy is the wrong metric. We need a system that balances precision and recall based on business economics, not just statistical correctness.

Additionally, customer behavior shifts over time. A model trained on 2023 data may fail silently in 2024. We need continuous monitoring, drift detection, and automated retraining — not a one-time model deployment.

## 2️⃣ SOLUTION PROPOSED

I built a production-grade MLOps system with the full ML lifecycle:

**Data Engineering:**
- PySpark-based ETL for scalable data processing
- Automated data validation with schema enforcement
- Drift detection using statistical tests

**Model Development:**
- Experimentation with multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Imbalanced data handling (SMOTE, Tomek Links)
- MLflow experiment tracking and model registry

**Deployment:**
- FastAPI REST API for programmatic access
- Professional web UI for business users
- Batch inference for campaign scoring

**Operations:**
- Continuous monitoring with retraining triggers
- CI/CD pipeline with GitHub Actions
- Docker containerization for consistent environments

**Metric Justification:**
I optimized for F1-score, not accuracy. Why? Because the dataset is imbalanced (88% no, 12% yes), and the cost of misclassification is uneven. Accuracy would give us 88% by predicting "no" for everyone — useless for business. F1-score balances precision (don't waste money on unlikely customers) and recall (don't miss high-probability customers). This aligns model optimization with business objectives.

## 3️⃣ ARCHITECTURE OVERVIEW

**Data Pipeline:**
- Raw CSV → PySpark ETL → Parquet (columnar, compressed, analytics-optimized)
- Schema validation → Drift detection → Quality gates
- Automated artifact versioning with timestamps
- Apache Airflow DAG orchestration with task dependencies

**ML Pipeline:**
- Feature engineering → Train/test split → Model training
- MLflow tracking (metrics, parameters, artifacts)
- Model evaluation → Registry promotion (only if better than baseline)

**Serving Layer:**
- FastAPI application with Pydantic validation
- REST API for real-time predictions
- Web UI for business users (no code required)
- Batch inference for campaign scoring

**Monitoring & Retraining:**
- Statistical drift detection (KS tests on feature distributions)
- Performance monitoring (F1-score tracking)
- Automated retraining triggers when drift or degradation detected

**CI/CD & Infrastructure:**
- GitHub Actions pipeline (lint → test → build → deploy)
- Docker multi-stage builds (security, efficiency)
- Automated testing with 51 unit/integration tests
- AWS ECS Fargate deployment (serverless containers)
- Amazon ECR for Docker registry
- CloudWatch for centralized logging
- IAM roles and security groups for access control
- Zero-downtime rolling updates
- Live production endpoint

**Technology Stack:**
- PySpark for scalable ETL
- Apache Airflow for workflow orchestration
- MLflow for experiment tracking and model registry
- FastAPI for high-performance API serving
- Docker Compose for local development
- GitHub Actions for CI/CD automation

## 4️⃣ DESIGN CHOICES + JUSTIFICATION

**PySpark instead of pandas:**
Banks have millions of customer records. Pandas loads everything into memory and fails at scale. PySpark distributes computation across nodes and handles datasets larger than RAM. It's the difference between a system that works on 100K rows and one that works on 100M rows.

**Parquet instead of CSV:**
CSV is human-readable but inefficient. Parquet is columnar (read only needed columns), compressed (10x smaller), and schema-enforced (prevents type errors). Analytics queries run 10-100x faster. This matters when you're processing data daily.

**MLflow instead of spreadsheets:**
Tracking experiments in spreadsheets doesn't scale. MLflow provides versioned experiment history, artifact storage, and model registry with stage transitions. It answers: "Which model is in production? What were its hyperparameters? What data was it trained on?" This is required for reproducibility and compliance.

**FastAPI instead of Flask:**
FastAPI has automatic API documentation (OpenAPI/Swagger), built-in request validation (Pydantic), and async support for high concurrency. It's faster and safer than Flask. When you're serving predictions to thousands of users, performance and type safety matter.

**CI/CD instead of manual deployment:**
Manual deployments fail. Someone forgets a step, pushes broken code, or deploys the wrong version. CI/CD automates testing, building, and deployment. Every change is validated before production. This reduces downtime and increases velocity.

**Docker instead of "works on my machine":**
Environment inconsistencies cause 50% of production bugs. Docker packages the application with all dependencies. The same container runs on my laptop, staging, and production. No surprises, no dependency conflicts.

## 5️⃣ EXAMPLES OF WORKFLOWS

**Workflow 1: Scheduled Daily Pipeline (Airflow)**
1. Airflow scheduler triggers DAG at midnight
2. Data ingestion task runs PySpark ETL, writes to Parquet
3. Validation task checks schema and drift, pushes status to XCom
4. If validation passes, transformation task engineers features
5. Training task trains model with MLflow tracking
6. Evaluation task compares against production baseline
7. If new model is better, pusher task promotes to registry
8. Success notification task logs completion and sends alerts
9. Airflow UI shows visual status of all tasks with logs

**Workflow 2: Business User Makes Prediction**
1. Marketing analyst opens web UI
2. Enters customer profile (age, job, balance, previous campaign results)
3. Clicks "Predict"
4. API validates input, loads production model from registry
5. Model returns prediction (yes/no) with probability score
6. UI displays result with confidence level
7. Prediction is logged for monitoring and audit trail

**Workflow 3: Drift Detection Triggers Retraining**
1. Monitoring system runs daily, comparing new data to reference distribution
2. KS test detects significant drift in "balance" and "duration" features
3. Performance metrics show F1-score dropped from 0.82 to 0.74
4. Retraining signal is triggered
5. Pipeline automatically runs full training cycle with new data
6. New model is evaluated and promoted if it improves performance
7. Alert is sent to ML team with drift report and retraining results

## 6️⃣ ANALOGY

This system works like a fully trained operations team at a bank:

**PySpark is the data analyst** — cleans messy records, handles scale, prepares reports.

**MLflow is the institutional memory** — remembers every experiment, every model version, every decision.

**FastAPI is the customer service desk** — takes requests, validates them, delivers answers instantly.

**Monitoring is the risk management team** — watches for problems, raises alerts before disasters happen.

**CI/CD is the quality assurance department** — ensures nothing broken reaches production, enforces standards automatically.

No single person needs to remember everything. The system has checks, balances, and automation at every step.

## 7️⃣ REAL BUSINESS IMPACT

**Increased Conversion Accuracy:**
Targeting high-probability customers improves conversion rates. Instead of 12% baseline, targeted campaigns can achieve 25-40% conversion by focusing on the right segments.

**Reduced Marketing Cost Per Lead:**
Eliminating low-probability customers from campaigns reduces wasted spend. If a campaign costs $5 per contact and reaches 100K people, improving targeting saves hundreds of thousands in wasted outreach.

**Automated Compliance & Audit Trails:**
Financial institutions face regulatory scrutiny. This system provides full lineage: which data, which model version, which hyperparameters produced each prediction. Auditors can reproduce any decision.

**Faster Retraining Cycles:**
Traditional ML teams take weeks to retrain models. This system detects drift and retrains automatically. Faster adaptation means the model stays relevant as customer behavior changes.

**Reliable Decision Pipeline:**
Manual processes break. Automated pipelines with testing and monitoring don't. This system runs the same way every time, reducing operational risk and freeing ML engineers to focus on improvements, not firefighting.

## 8️⃣ INDUSTRY VALIDATION

This architecture matches how leading tech and financial companies build ML systems in 2024-2025:

**Databricks Lakehouse:**
Combines data engineering (Spark) with ML (MLflow) in a unified platform. My system follows the same pattern: PySpark for ETL, MLflow for lifecycle management.

**AWS SageMaker:**
Provides managed infrastructure for training, deployment, and monitoring. My system implements the same workflow: modular pipelines, model registry, automated retraining.

**Google Vertex AI:**
Emphasizes ML pipelines with artifact tracking and continuous monitoring. My architecture mirrors this: component-based design, versioned artifacts, drift detection.

**Azure ML:**
Focuses on MLOps with CI/CD integration and model governance. My system includes the same elements: GitHub Actions pipelines, model registry with stage transitions, audit trails.

**Netflix & Uber (Michelangelo):**
Built internal ML platforms with automated pipelines, feature stores, and monitoring. My system demonstrates the same principles: automation, observability, reproducibility.

**Industry Direction (2024-2028):**
The ML industry is converging on:
- Automated ML lifecycle (not manual notebooks)
- Continuous monitoring (not deploy-and-forget)
- Scalable data engineering (Spark, not pandas)
- ML observability (drift detection, performance tracking)
- CI/CD for models (treat models like software)

My system implements all of these. It's not experimental — it's how production ML is done today.

## 9️⃣ KEY TAKEAWAYS

**What problem it solves:**
Banks waste millions on untargeted marketing. This system predicts who will subscribe, optimizing spend and improving conversion rates.

**Why it matters today:**
Customer behavior changes constantly. Static models fail silently. This system adapts automatically through monitoring and retraining.

**Why this architecture is durable:**
It's built on proven patterns from FAANG and top ML platforms. Modular design means components can be upgraded without rewriting everything.

**Why enterprises want this:**
It reduces operational risk, provides audit trails, scales with data growth, and frees ML teams from manual toil. It's reliable, repeatable, and maintainable.

**Why I'm uniquely positioned:**
I understand both the ML (algorithms, metrics, evaluation) and the engineering (pipelines, deployment, monitoring). I don't just train models — I build systems that ship, scale, and operate in production. I think like a senior engineer who's shipped real ML products, not an academic who stops at Jupyter notebooks.

## 🔟 TONE & POSITIONING

This isn't a toy project. It's a production-grade system designed with the same principles used at:
- FAANG companies building recommendation engines
- Financial institutions deploying credit risk models
- Tech unicorns scaling ML platforms

I built it to demonstrate that I can:
- Design end-to-end ML systems, not just train models
- Make engineering decisions based on business requirements
- Implement MLOps best practices from day one
- Ship reliable, maintainable, scalable ML products

This is the kind of system that gets deployed, not the kind that stays in a notebook.
