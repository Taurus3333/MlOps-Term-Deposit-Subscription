"""Data validation component using pandas and schema.yaml."""
import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd
from scipy.stats import ks_2samp

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.constants import (
    CLEANED_DATA_FILE,
    SCHEMA_FILE,
    CURRENT_ARTIFACT_DIR,
    NUMERICAL_COLUMNS
)

logger = get_logger(__name__)


class DataValidation:
    """Validate cleaned data against schema definition."""
    
    def __init__(self):
        self.artifact_dir = CURRENT_ARTIFACT_DIR / "data_validation"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.schema = None
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "data_file": str(CLEANED_DATA_FILE),
            "schema_file": str(SCHEMA_FILE),
            "checks": [],
            "overall_status": "PENDING"
        }
    
    def load_schema(self):
        """Load schema definition from YAML."""
        try:
            logger.info(f"Loading schema from: {SCHEMA_FILE}")
            
            with open(SCHEMA_FILE, 'r') as f:
                self.schema = yaml.safe_load(f)
            
            logger.info(f"Schema loaded: {self.schema['dataset_name']} v{self.schema['version']}")
            return self.schema
            
        except Exception as e:
            raise CustomException(f"Failed to load schema: {str(e)}", sys.exc_info())
    
    def load_data(self):
        """Load cleaned data using pandas."""
        try:
            # Check if CSV exists (OneDrive workaround)
            csv_file = CLEANED_DATA_FILE.with_suffix('.csv')
            
            if csv_file.exists():
                logger.info(f"Loading data from CSV: {csv_file}")
                df = pd.read_csv(csv_file)
            elif CLEANED_DATA_FILE.exists():
                logger.info(f"Loading data from Parquet: {CLEANED_DATA_FILE}")
                df = pd.read_parquet(CLEANED_DATA_FILE)
            else:
                raise FileNotFoundError(f"No data file found at {CLEANED_DATA_FILE}")
            
            logger.info(f"Data loaded. Rows: {len(df)}, Columns: {len(df.columns)}")
            return df
            
        except Exception as e:
            raise CustomException(f"Failed to load data: {str(e)}", sys.exc_info())
    
    def add_check(self, check_name, status, message, details=None):
        """Add validation check result."""
        check = {
            "check": check_name,
            "status": status,
            "message": message
        }
        if details:
            check["details"] = details
        
        self.validation_report["checks"].append(check)
        
        status_icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
        logger.info(f"{status_icon} {check_name}: {message}")
    
    def validate_columns(self, df):
        """Validate column presence."""
        logger.info("Validating column presence...")
        
        required_cols = self.schema['required_columns']
        actual_cols = df.columns.tolist()
        
        missing_cols = set(required_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(required_cols)
        
        if missing_cols:
            self.add_check(
                "Column Presence",
                "FAIL",
                f"Missing columns: {missing_cols}",
                {"missing": list(missing_cols)}
            )
            return False
        
        if extra_cols:
            self.add_check(
                "Column Presence",
                "WARNING",
                f"Extra columns found: {extra_cols}",
                {"extra": list(extra_cols)}
            )
        else:
            self.add_check(
                "Column Presence",
                "PASS",
                f"All {len(required_cols)} required columns present"
            )
        
        return True
    
    def validate_data_types(self, df):
        """Validate data types."""
        logger.info("Validating data types...")
        
        type_issues = []
        
        for col_name, col_spec in self.schema['columns'].items():
            if col_name not in df.columns:
                continue
            
            expected_type = col_spec['type']
            actual_dtype = df[col_name].dtype
            
            # Map schema types to pandas types
            type_valid = False
            if expected_type == 'integer' and pd.api.types.is_integer_dtype(actual_dtype):
                type_valid = True
            elif expected_type == 'string' and pd.api.types.is_object_dtype(actual_dtype):
                type_valid = True
            
            if not type_valid:
                type_issues.append({
                    "column": col_name,
                    "expected": expected_type,
                    "actual": str(actual_dtype)
                })
        
        if type_issues:
            self.add_check(
                "Data Types",
                "FAIL",
                f"{len(type_issues)} columns have incorrect types",
                {"issues": type_issues}
            )
            return False
        else:
            self.add_check(
                "Data Types",
                "PASS",
                "All columns have correct data types"
            )
            return True
    
    def validate_value_ranges(self, df):
        """Validate numerical value ranges."""
        logger.info("Validating value ranges...")
        
        range_violations = []
        
        for col_name, col_spec in self.schema['columns'].items():
            if col_name not in df.columns or col_spec['type'] != 'integer':
                continue
            
            if 'min_value' in col_spec:
                min_val = col_spec['min_value']
                violations = (df[col_name] < min_val).sum()
                if violations > 0:
                    range_violations.append({
                        "column": col_name,
                        "issue": f"{violations} values below minimum {min_val}"
                    })
            
            if 'max_value' in col_spec:
                max_val = col_spec['max_value']
                violations = (df[col_name] > max_val).sum()
                if violations > 0:
                    range_violations.append({
                        "column": col_name,
                        "issue": f"{violations} values above maximum {max_val}"
                    })
        
        if range_violations:
            self.add_check(
                "Value Ranges",
                "FAIL",
                f"{len(range_violations)} range violations found",
                {"violations": range_violations}
            )
            return False
        else:
            self.add_check(
                "Value Ranges",
                "PASS",
                "All numerical values within expected ranges"
            )
            return True
    
    def validate_categorical_values(self, df):
        """Validate categorical column values."""
        logger.info("Validating categorical values...")
        
        category_violations = []
        
        for col_name, col_spec in self.schema['columns'].items():
            if col_name not in df.columns or 'allowed_values' not in col_spec:
                continue
            
            allowed = set(col_spec['allowed_values'])
            actual = set(df[col_name].unique())
            
            invalid = actual - allowed
            
            if invalid:
                category_violations.append({
                    "column": col_name,
                    "invalid_values": list(invalid),
                    "count": len(invalid)
                })
        
        if category_violations:
            self.add_check(
                "Categorical Values",
                "FAIL",
                f"{len(category_violations)} columns have invalid categories",
                {"violations": category_violations}
            )
            return False
        else:
            self.add_check(
                "Categorical Values",
                "PASS",
                "All categorical values are valid"
            )
            return True
    
    def validate_row_count(self, df):
        """Validate row count is within expected range."""
        logger.info("Validating row count...")
        
        row_count = len(df)
        min_rows = self.schema['expected_row_count']['min']
        max_rows = self.schema['expected_row_count']['max']
        
        if row_count < min_rows:
            self.add_check(
                "Row Count",
                "FAIL",
                f"Row count {row_count} below minimum {min_rows}"
            )
            return False
        elif row_count > max_rows:
            self.add_check(
                "Row Count",
                "WARNING",
                f"Row count {row_count} above maximum {max_rows}"
            )
        else:
            self.add_check(
                "Row Count",
                "PASS",
                f"Row count {row_count} within expected range"
            )
        
        return True
    
    def validate_data_drift(self, df):
        """Validate data distribution using Kolmogorov-Smirnov test."""
        logger.info("Validating data drift using KS test...")
        
        # Check if reference data exists for comparison
        reference_file = CLEANED_DATA_FILE.parent / "reference_data.csv"
        
        if not reference_file.exists():
            # First run - save current data as reference
            logger.info("No reference data found. Saving current data as reference...")
            df.to_csv(reference_file, index=False)
            self.add_check(
                "Data Drift (KS Test)",
                "PASS",
                "Reference data created. No drift detected (first run)"
            )
            return True
        
        # Load reference data
        df_reference = pd.read_csv(reference_file)
        
        # Perform KS test on numerical columns
        drift_detected = []
        ks_results = {}
        threshold = 0.05  # p-value threshold
        
        for col in NUMERICAL_COLUMNS:
            if col in df.columns and col in df_reference.columns:
                # Perform KS test
                statistic, p_value = ks_2samp(df[col], df_reference[col])
                ks_results[col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": bool(p_value < threshold)
                }
                
                if p_value < threshold:
                    drift_detected.append({
                        "column": col,
                        "p_value": float(p_value),
                        "statistic": float(statistic)
                    })
        
        if drift_detected:
            self.add_check(
                "Data Drift (KS Test)",
                "WARNING",
                f"Distribution drift detected in {len(drift_detected)} columns",
                {
                    "drifted_columns": drift_detected,
                    "threshold": threshold,
                    "recommendation": "Consider retraining model or investigating data source changes"
                }
            )
        else:
            self.add_check(
                "Data Drift (KS Test)",
                "PASS",
                f"No significant drift detected in {len(ks_results)} numerical columns",
                {"ks_results": ks_results}
            )
        
        return True
    
    def save_validation_report(self):
        """Save validation report to artifacts."""
        try:
            # Determine overall status
            statuses = [check['status'] for check in self.validation_report['checks']]
            if 'FAIL' in statuses:
                self.validation_report['overall_status'] = 'FAIL'
            elif 'WARNING' in statuses:
                self.validation_report['overall_status'] = 'WARNING'
            else:
                self.validation_report['overall_status'] = 'PASS'
            
            report_file = self.artifact_dir / "validation_report.json"
            with open(report_file, 'w') as f:
                json.dump(self.validation_report, f, indent=2)
            
            logger.info(f"Validation report saved to: {report_file}")
            logger.info(f"Overall validation status: {self.validation_report['overall_status']}")
            
        except Exception as e:
            raise CustomException(f"Failed to save validation report: {str(e)}", sys.exc_info())
    
    def run_validation(self):
        """Execute full validation pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Validation Pipeline")
            logger.info("=" * 60)
            
            # Load schema
            self.load_schema()
            
            # Load data
            df = self.load_data()
            
            # Run validation checks
            self.validate_columns(df)
            self.validate_data_types(df)
            self.validate_value_ranges(df)
            self.validate_categorical_values(df)
            self.validate_row_count(df)
            self.validate_data_drift(df)
            
            # Save report
            self.save_validation_report()
            
            logger.info("=" * 60)
            logger.info("Data Validation Pipeline Completed")
            logger.info("=" * 60)
            
            return self.validation_report['overall_status']
            
        except Exception as e:
            raise CustomException(f"Validation pipeline failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    validation = DataValidation()
    status = validation.run_validation()
    print(f"\nValidation Status: {status}")
