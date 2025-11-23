"""Project-wide constants."""
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
SCHEMA_DIR = PROJECT_ROOT / "data_schema"

# Timestamp for artifact versioning
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_ARTIFACT_DIR = ARTIFACTS_DIR / TIMESTAMP

# Data files
RAW_DATA_FILE = DATA_DIR / "bank-full.csv"
CLEANED_DATA_FILE = CLEANED_DATA_DIR / "cleaned-bank.parquet"
SCHEMA_FILE = SCHEMA_DIR / "schema.yaml"
RESULTS_FILE = PROJECT_ROOT / "results.json"

# Column names (based on UCI Bank Marketing dataset)
COLUMN_NAMES = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]

# Target column
TARGET_COLUMN = "y"

# Categorical columns
CATEGORICAL_COLUMNS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome", "y"
]

# Numerical columns
NUMERICAL_COLUMNS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
