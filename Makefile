.PHONY: help setup install clean test lint format run-pipeline run-api docker-build docker-up docker-down

help:
	@echo "Bank Marketing Term Deposit Prediction - Makefile Commands"
	@echo "==========================================================="
	@echo "setup          - Create virtual environment and install dependencies"
	@echo "install        - Install dependencies only"
	@echo "clean          - Remove artifacts, logs, and cache files"
	@echo "test           - Run tests with pytest"
	@echo "lint           - Run flake8 linting"
	@echo "format         - Format code with black"
	@echo "run-pipeline   - Execute full ML pipeline"
	@echo "run-api        - Start FastAPI server locally"
	@echo "docker-build   - Build Docker image"
	@echo "docker-up      - Start Docker Compose services"
	@echo "docker-down    - Stop Docker Compose services"

setup:
	python -m venv .venv
	.venv\Scripts\activate && pip install --upgrade pip
	.venv\Scripts\activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

clean:
	if exist artifacts rmdir /s /q artifacts
	if exist logs\*.log del /q logs\*.log
	if exist cleaned_data\*.parquet del /q cleaned_data\*.parquet
	if exist results.json del /q results.json
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"

test:
	python run_tests.py

test-quick:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

test-integration:
	pytest tests/ -v -m integration

test-unit:
	pytest tests/ -v -m "not integration"

lint:
	flake8 src/ --max-line-length=120 --exclude=__pycache__

format:
	black src/ --line-length=120

run-pipeline:
	python -m src.pipeline

run-api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t bank-marketing-api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
