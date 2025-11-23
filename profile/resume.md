# Resume Entry - Bank Marketing ML System

## Project Headline

**Production-Grade MLOps Pipeline for Bank Term Deposit Prediction with PySpark ETL, MLflow Registry, FastAPI Serving, and Automated CI/CD**

---

## Bullet Points

• Architected end-to-end MLOps pipeline with 6 modular components (ingestion, validation, transformation, training, evaluation, deployment) orchestrated via Apache Airflow DAG, using PySpark for scalable ETL, processing imbalanced datasets and outputting Parquet for downstream analytics

• Implemented automated data validation with schema enforcement and drift detection using Kolmogorov-Smirnov tests, preventing silent model degradation and ensuring data quality gates before training

• Built ML experimentation framework with MLflow tracking and model registry, evaluating 4 algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) with SMOTE/Tomek Links for class imbalance, optimizing F1-score over accuracy to align with asymmetric business costs

• Deployed dual-interface serving layer using FastAPI with REST API for programmatic access and banking-style web UI for business users, enabling both real-time predictions and batch inference for campaign scoring

• Established CI/CD pipeline with GitHub Actions integrating automated testing (51 unit/integration tests), SonarQube code quality checks, Docker multi-stage builds, and containerized deployment with health checks

• Designed monitoring system with statistical drift detection and performance tracking, triggering automated retraining when F1-score degradation or distribution shifts exceed defined thresholds

• Delivered production-ready system with full artifact lineage, reproducible experiments, and audit trail compliance, reducing manual ML operations overhead while enabling rapid model iteration and deployment cycles

---

## Impact Statement

**Outcome:** Built enterprise-grade ML platform demonstrating FAANG-level MLOps practices—Airflow-orchestrated workflows, automated lifecycle management, continuous monitoring, scalable data engineering, and deployment reliability—reducing time-to-production for ML models while ensuring regulatory compliance and operational stability.

---

## Alternative Compact Format (for space-constrained resumes)

**Production MLOps Pipeline - Bank Marketing Prediction System**

• Designed 6-component MLOps pipeline orchestrated with Apache Airflow DAG, PySpark ETL, MLflow registry, and automated validation using KS drift detection  
• Trained and evaluated 4 ML algorithms with imbalanced data handling (SMOTE/Tomek), optimizing F1-score for business cost alignment  
• Deployed FastAPI serving layer with REST API and web UI, supporting real-time and batch inference workflows  
• Implemented CI/CD with GitHub Actions, Docker containerization, automated testing (51 tests), and SonarQube integration  
• Built monitoring system with drift detection and automated retraining triggers, ensuring model reliability and compliance

---

## Technical Keywords (for ATS optimization)

MLOps | Apache Airflow | PySpark | MLflow | FastAPI | Docker | CI/CD | GitHub Actions | Data Pipeline | Workflow Orchestration | DAG | Model Registry | Drift Detection | Automated Testing | REST API | Batch Inference | Feature Engineering | Class Imbalance | SMOTE | Model Monitoring | Parquet | SonarQube | Python | Scikit-learn | XGBoost | LightGBM | Pydantic | Artifact Management | Reproducibility | Production ML | Task Scheduling | XCom
