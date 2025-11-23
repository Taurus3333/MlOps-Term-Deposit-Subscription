"""Data ingestion component using PySpark for production-grade ETL."""
import sys
from pathlib import Path
from datetime import datetime
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace
from pyspark.sql.types import IntegerType, StringType, DoubleType

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.constants import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    CURRENT_ARTIFACT_DIR,
    COLUMN_NAMES,
    NUMERICAL_COLUMNS
)

logger = get_logger(__name__)


class DataIngestion:
    """Handle data ingestion and ETL using PySpark."""
    
    def __init__(self):
        self.artifact_dir = CURRENT_ARTIFACT_DIR / "data_ingestion"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.spark = None
        
    def initialize_spark(self):
        """Initialize Spark session."""
        try:
            logger.info("Initializing Spark session...")
            
            # Set Java home for PySpark (Windows compatibility)
            import os
            
            # Normalize JAVA_HOME path for Windows (convert forward slashes to backslashes)
            if os.environ.get('JAVA_HOME'):
                java_home = os.environ['JAVA_HOME'].replace('/', '\\')
                os.environ['JAVA_HOME'] = java_home
                logger.info(f"Normalized JAVA_HOME to: {java_home}")
            else:
                # Try common Java installation paths
                possible_java_homes = [
                    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot",
                    r"C:\Program Files\Java\jdk-17",
                    r"C:\Program Files\Java\jdk-11",
                ]
                for java_home in possible_java_homes:
                    if os.path.exists(java_home):
                        os.environ['JAVA_HOME'] = java_home
                        logger.info(f"Set JAVA_HOME to: {java_home}")
                        break
            
            # Also set PYSPARK_SUBMIT_ARGS to avoid path issues
            os.environ['PYSPARK_PYTHON'] = sys.executable
            os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
            
            self.spark = SparkSession.builder \
                .appName("BankMarketingETL") \
                .config("spark.driver.memory", "4g") \
                .config("spark.sql.shuffle.partitions", "4") \
                .config("spark.driver.extraJavaOptions", "-Djava.io.tmpdir=C:\\temp") \
                .getOrCreate()
            
            logger.info(f"Spark session initialized. Version: {self.spark.version}")
            return self.spark
            
        except Exception as e:
            raise CustomException(f"Failed to initialize Spark: {str(e)}", sys.exc_info())
    
    def read_malformed_csv(self):
        """Read the malformed CSV with quoted semicolon-delimited data."""
        try:
            logger.info(f"Reading raw data from: {RAW_DATA_FILE}")
            
            # Read as single column first
            df_raw = self.spark.read.text(str(RAW_DATA_FILE))
            
            logger.info(f"Raw data loaded. Row count: {df_raw.count()}")
            
            # Skip header row
            header = df_raw.first()[0]
            df_raw = df_raw.filter(col("value") != header)
            
            # Split by semicolon and remove quotes
            df_split = df_raw.selectExpr(
                *[f"split(value, ';')[{i}] as {col_name}" 
                  for i, col_name in enumerate(COLUMN_NAMES)]
            )
            
            # Clean quotes from all columns
            for col_name in COLUMN_NAMES:
                df_split = df_split.withColumn(
                    col_name,
                    regexp_replace(trim(col(col_name)), '"', '')
                )
            
            logger.info("Successfully parsed malformed CSV structure")
            return df_split
            
        except Exception as e:
            raise CustomException(f"Failed to read CSV: {str(e)}", sys.exc_info())
    
    def apply_type_casting(self, df):
        """Apply proper data types to columns."""
        try:
            logger.info("Applying type casting...")
            
            # Cast numerical columns
            for col_name in NUMERICAL_COLUMNS:
                df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
            
            # Ensure categorical columns are strings
            categorical_cols = [c for c in COLUMN_NAMES if c not in NUMERICAL_COLUMNS]
            for col_name in categorical_cols:
                df = df.withColumn(col_name, col(col_name).cast(StringType()))
            
            logger.info("Type casting completed")
            return df
            
        except Exception as e:
            raise CustomException(f"Type casting failed: {str(e)}", sys.exc_info())
    
    def handle_nulls(self, df):
        """Handle null values."""
        try:
            logger.info("Handling null values...")
            
            null_counts = {col_name: df.filter(col(col_name).isNull()).count() 
                          for col_name in df.columns}
            
            logger.info(f"Null counts before handling: {null_counts}")
            
            # Drop rows with nulls in critical columns
            df_cleaned = df.dropna(subset=["age", "y"])
            
            rows_before = df.count()
            rows_after = df_cleaned.count()
            rows_dropped = rows_before - rows_after
            
            logger.info(f"Rows dropped due to nulls: {rows_dropped}")
            
            return df_cleaned
            
        except Exception as e:
            raise CustomException(f"Null handling failed: {str(e)}", sys.exc_info())
    
    def save_as_parquet(self, df):
        """Save cleaned data as Parquet."""
        try:
            logger.info(f"Saving cleaned data to: {CLEANED_DATA_FILE}")
            
            # Ensure directory exists
            CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to pandas and save (workaround for Windows Hadoop issue)
            logger.info("Converting Spark DataFrame to Pandas...")
            pdf = df.toPandas()
            
            # Try parquet, fallback to CSV if OneDrive blocks it
            try:
                pdf.to_parquet(str(CLEANED_DATA_FILE), index=False, engine='pyarrow')
                logger.info("Data saved successfully as Parquet")
            except PermissionError:
                logger.warning("Parquet write blocked by OneDrive, saving as CSV instead...")
                csv_file = CLEANED_DATA_FILE.with_suffix('.csv')
                pdf.to_csv(str(csv_file), index=False)
                logger.info(f"Data saved as CSV: {csv_file}")
                logger.info("Note: For production, move project outside OneDrive folder")
            
        except Exception as e:
            raise CustomException(f"Failed to save data: {str(e)}", sys.exc_info())
    
    def save_metadata(self, df):
        """Save ingestion metadata."""
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "raw_data_path": str(RAW_DATA_FILE),
                "cleaned_data_path": str(CLEANED_DATA_FILE),
                "row_count": df.count(),
                "column_count": len(df.columns),
                "columns": df.columns,
                "schema": {col_name: str(df.schema[col_name].dataType) 
                          for col_name in df.columns}
            }
            
            metadata_file = self.artifact_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to: {metadata_file}")
            
        except Exception as e:
            raise CustomException(f"Failed to save metadata: {str(e)}", sys.exc_info())
    
    def run_etl(self):
        """Execute full ETL pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Ingestion ETL Pipeline")
            logger.info("=" * 60)
            
            # Initialize Spark
            self.initialize_spark()
            
            # Read malformed CSV
            df = self.read_malformed_csv()
            
            # Apply type casting
            df = self.apply_type_casting(df)
            
            # Handle nulls
            df = self.handle_nulls(df)
            
            # Show sample
            logger.info("Sample of cleaned data:")
            df.show(5, truncate=False)
            
            # Save as Parquet
            self.save_as_parquet(df)
            
            # Save metadata
            self.save_metadata(df)
            
            logger.info("=" * 60)
            logger.info("Data Ingestion ETL Pipeline Completed Successfully")
            logger.info("=" * 60)
            
            return str(CLEANED_DATA_FILE)
            
        except Exception as e:
            raise CustomException(f"ETL pipeline failed: {str(e)}", sys.exc_info())
        
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("Spark session stopped")


if __name__ == "__main__":
    ingestion = DataIngestion()
    output_path = ingestion.run_etl()
    print(f"\nCleaned data saved to: {output_path}")
