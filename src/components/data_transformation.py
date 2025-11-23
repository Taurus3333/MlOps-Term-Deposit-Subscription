"""Data transformation component using experiment results."""
import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.constants import CLEANED_DATA_FILE, RESULTS_FILE, TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact

logger = get_logger(__name__)


class DataTransformation:
    """Transform data using experiment results."""
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load cleaned data."""
        try:
            logger.info(f"Loading data from: {self.config.data_path}")
            
            csv_file = self.config.data_path.with_suffix('.csv')
            if csv_file.exists():
                df = pd.read_csv(csv_file)
            else:
                df = pd.read_parquet(self.config.data_path)
            
            logger.info(f"Data loaded: {df.shape}")
            return df
            
        except Exception as e:
            raise CustomException(f"Failed to load data: {str(e)}", sys.exc_info())
    
    def load_experiment_results(self):
        """Load experiment results."""
        try:
            logger.info(f"Loading experiment results from: {self.config.results_path}")
            
            with open(self.config.results_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Best model: {results['best_model']['name']}")
            logger.info(f"Sampling method: {results['best_model']['sampling_method']}")
            
            return results
            
        except Exception as e:
            raise CustomException(f"Failed to load results: {str(e)}", sys.exc_info())
    
    def encode_features(self, df, results):
        """Encode features using experiment strategy."""
        try:
            logger.info("Encoding features...")
            
            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]
            
            # Encode target
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            logger.info(f"Target classes: {self.target_encoder.classes_}")
            
            # Encode categorical features
            categorical_cols = results['preprocessing']['categorical_columns']
            X_encoded = X.copy()
            
            for col in categorical_cols:
                if col in X_encoded.columns:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col])
                    self.label_encoders[col] = le
            
            logger.info(f"Encoded {len(categorical_cols)} categorical features")
            
            return X_encoded, y_encoded
            
        except Exception as e:
            raise CustomException(f"Feature encoding failed: {str(e)}", sys.exc_info())
    
    def split_data(self, X, y):
        """Split data into train and test sets."""
        try:
            logger.info("Splitting data...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
            
            logger.info(f"Train set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Train class distribution: {np.bincount(y_train)}")
            logger.info(f"Test class distribution: {np.bincount(y_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise CustomException(f"Data split failed: {str(e)}", sys.exc_info())
    
    def apply_sampling(self, X_train, y_train, sampling_method):
        """Apply imbalanced data sampling."""
        try:
            logger.info(f"Applying sampling method: {sampling_method}")
            
            if sampling_method == 'original':
                logger.info("No resampling applied")
                return X_train, y_train
            
            elif sampling_method == 'smote':
                sampler = SMOTE(random_state=self.config.random_state)
            
            elif sampling_method == 'tomek':
                sampler = TomekLinks()
            
            elif sampling_method == 'smote_tomek':
                sampler = SMOTETomek(random_state=self.config.random_state)
            
            else:
                logger.warning(f"Unknown sampling method: {sampling_method}, using original")
                return X_train, y_train
            
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
            logger.info(f"Before sampling: {np.bincount(y_train)}")
            logger.info(f"After sampling: {np.bincount(y_resampled)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            raise CustomException(f"Sampling failed: {str(e)}", sys.exc_info())
    
    def save_data(self, X_train, X_test, y_train, y_test):
        """Save transformed data."""
        try:
            logger.info("Saving transformed data...")
            
            # Save train data
            train_data = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns') else None)
            train_data['target'] = y_train
            train_path = self.config.artifact_dir / "train.csv"
            train_data.to_csv(train_path, index=False)
            logger.info(f"Train data saved: {train_path}")
            
            # Save test data
            test_data = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
            test_data['target'] = y_test
            test_path = self.config.artifact_dir / "test.csv"
            test_data.to_csv(test_path, index=False)
            logger.info(f"Test data saved: {test_path}")
            
            return train_path, test_path
            
        except Exception as e:
            raise CustomException(f"Failed to save data: {str(e)}", sys.exc_info())
    
    def save_preprocessor(self):
        """Save preprocessing objects."""
        try:
            logger.info("Saving preprocessor...")
            
            preprocessor = {
                'label_encoders': self.label_encoders,
                'target_encoder': self.target_encoder,
                'scaler': self.scaler
            }
            
            preprocessor_path = self.config.artifact_dir / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logger.info(f"Preprocessor saved: {preprocessor_path}")
            return preprocessor_path
            
        except Exception as e:
            raise CustomException(f"Failed to save preprocessor: {str(e)}", sys.exc_info())
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Execute data transformation pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Transformation")
            logger.info("=" * 60)
            
            # Load data and results
            df = self.load_data()
            results = self.load_experiment_results()
            
            # Encode features
            X_encoded, y_encoded = self.encode_features(df, results)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X_encoded, y_encoded)
            
            # Apply sampling from best model
            sampling_method = results['best_model']['sampling_method']
            X_train_sampled, y_train_sampled = self.apply_sampling(X_train, y_train, sampling_method)
            
            # Save data
            train_path, test_path = self.save_data(X_train_sampled, X_test, y_train_sampled, y_test)
            
            # Save preprocessor
            preprocessor_path = self.save_preprocessor()
            
            # Create artifact
            artifact = DataTransformationArtifact(
                train_data_path=train_path,
                test_data_path=test_path,
                preprocessor_path=preprocessor_path,
                feature_names=list(X_encoded.columns),
                target_name=TARGET_COLUMN,
                train_shape=(len(X_train_sampled), len(X_encoded.columns)),
                test_shape=(len(X_test), len(X_encoded.columns))
            )
            
            logger.info("=" * 60)
            logger.info("Data Transformation Completed Successfully")
            logger.info("=" * 60)
            
            return artifact
            
        except Exception as e:
            raise CustomException(f"Data transformation failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    config = DataTransformationConfig()
    transformation = DataTransformation(config)
    artifact = transformation.initiate_data_transformation()
    
    print(f"\nTrain data: {artifact.train_data_path}")
    print(f"Test data: {artifact.test_data_path}")
    print(f"Preprocessor: {artifact.preprocessor_path}")
