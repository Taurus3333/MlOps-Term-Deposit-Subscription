"""Model monitoring for drift detection and performance tracking."""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from src.logging.custom_logger import get_logger
from src.exception.custom_exception import CustomException
from src.constants import CLEANED_DATA_FILE

logger = get_logger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently import ColumnMapping
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not available. Drift reports will be limited.")


class ModelMonitor:
    """Monitor model performance and data drift."""
    
    def __init__(self, monitoring_dir: Path = None):
        self.monitoring_dir = monitoring_dir or Path("artifacts/monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.reference_data = None
        self.current_data = None
        
    def load_reference_data(self, reference_path: Path = None):
        """Load reference data for comparison."""
        try:
            if reference_path is None:
                reference_path = CLEANED_DATA_FILE.with_suffix('.csv')
            
            logger.info(f"Loading reference data from: {reference_path}")
            self.reference_data = pd.read_csv(reference_path)
            logger.info(f"Reference data loaded: {self.reference_data.shape}")
            
            return self.reference_data
            
        except Exception as e:
            raise CustomException(f"Failed to load reference data: {str(e)}", sys.exc_info())
    
    def load_current_data(self, current_path: Path):
        """Load current/production data for monitoring."""
        try:
            logger.info(f"Loading current data from: {current_path}")
            self.current_data = pd.read_csv(current_path)
            logger.info(f"Current data loaded: {self.current_data.shape}")
            
            return self.current_data
            
        except Exception as e:
            raise CustomException(f"Failed to load current data: {str(e)}", sys.exc_info())
    
    def detect_data_drift(self) -> Dict:
        """Detect data drift using KS test and Evidently."""
        try:
            logger.info("Detecting data drift...")
            
            if self.reference_data is None or self.current_data is None:
                raise ValueError("Reference and current data must be loaded first")
            
            # KS test for numerical columns
            numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'y' in numerical_cols:
                numerical_cols.remove('y')
            
            drift_results = {}
            drift_detected_cols = []
            threshold = 0.05
            
            for col in numerical_cols:
                if col in self.current_data.columns:
                    statistic, p_value = ks_2samp(
                        self.reference_data[col],
                        self.current_data[col]
                    )
                    
                    is_drift = p_value < threshold
                    drift_results[col] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift_detected": bool(is_drift)
                    }
                    
                    if is_drift:
                        drift_detected_cols.append(col)
                        logger.warning(f"Drift detected in {col}: p-value={p_value:.4f}")
            
            # Evidently report (optional)
            if EVIDENTLY_AVAILABLE:
                try:
                    column_mapping = ColumnMapping()
                    column_mapping.target = 'y'
                    column_mapping.numerical_features = numerical_cols
                    
                    report = Report(metrics=[
                        DataDriftPreset(),
                        DataQualityPreset()
                    ])
                    
                    report.run(
                        reference_data=self.reference_data,
                        current_data=self.current_data,
                        column_mapping=column_mapping
                    )
                    
                    # Save report
                    report_path = self.monitoring_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    report.save_html(str(report_path))
                    logger.info(f"Evidently report saved: {report_path}")
                    
                except Exception as e:
                    logger.warning(f"Evidently report generation failed: {str(e)}")
            else:
                logger.info("Evidently not available. Skipping visual drift report.")
            
            drift_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_features": len(numerical_cols),
                "drifted_features": len(drift_detected_cols),
                "drift_percentage": (len(drift_detected_cols) / len(numerical_cols)) * 100 if numerical_cols else 0,
                "drifted_columns": drift_detected_cols,
                "drift_details": drift_results
            }
            
            logger.info(f"Drift detection complete: {len(drift_detected_cols)}/{len(numerical_cols)} features drifted")
            
            return drift_summary
            
        except Exception as e:
            raise CustomException(f"Drift detection failed: {str(e)}", sys.exc_info())
    
    def track_model_performance(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> Dict:
        """Track model performance metrics."""
        try:
            logger.info("Tracking model performance...")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": float(accuracy_score(actuals, predictions)),
                "precision": float(precision_score(actuals, predictions, zero_division=0)),
                "recall": float(recall_score(actuals, predictions, zero_division=0)),
                "f1_score": float(f1_score(actuals, predictions, zero_division=0))
            }
            
            logger.info(f"Performance metrics: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise CustomException(f"Performance tracking failed: {str(e)}", sys.exc_info())
    
    def check_performance_degradation(self, current_f1: float, baseline_f1: float, threshold: float = 0.05) -> Tuple[bool, str]:
        """Check if model performance has degraded."""
        try:
            degradation = baseline_f1 - current_f1
            degradation_pct = (degradation / baseline_f1) * 100
            
            is_degraded = degradation > threshold
            
            if is_degraded:
                message = f"Performance degraded by {degradation:.4f} ({degradation_pct:.2f}%). Baseline F1: {baseline_f1:.4f}, Current F1: {current_f1:.4f}"
                logger.warning(message)
            else:
                message = f"Performance stable. Baseline F1: {baseline_f1:.4f}, Current F1: {current_f1:.4f}"
                logger.info(message)
            
            return is_degraded, message
            
        except Exception as e:
            raise CustomException(f"Performance check failed: {str(e)}", sys.exc_info())
    
    def should_trigger_retraining(self, drift_summary: Dict, performance_degraded: bool) -> Tuple[bool, str]:
        """Determine if retraining should be triggered."""
        try:
            logger.info("Evaluating retraining trigger conditions...")
            
            reasons = []
            
            # Check drift threshold
            drift_threshold = 30  # 30% of features drifted
            if drift_summary['drift_percentage'] > drift_threshold:
                reasons.append(f"Data drift detected in {drift_summary['drift_percentage']:.1f}% of features (threshold: {drift_threshold}%)")
            
            # Check performance degradation
            if performance_degraded:
                reasons.append("Model performance degradation detected")
            
            should_retrain = len(reasons) > 0
            
            if should_retrain:
                message = "RETRAINING TRIGGERED. Reasons: " + "; ".join(reasons)
                logger.warning(message)
            else:
                message = "No retraining needed. Model performance and data distribution are stable."
                logger.info(message)
            
            return should_retrain, message
            
        except Exception as e:
            raise CustomException(f"Retraining trigger check failed: {str(e)}", sys.exc_info())
    
    def save_monitoring_report(self, drift_summary: Dict, performance_metrics: Dict, retraining_decision: Dict):
        """Save comprehensive monitoring report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "drift_analysis": drift_summary,
                "performance_metrics": performance_metrics,
                "retraining_decision": retraining_decision
            }
            
            report_path = self.monitoring_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Monitoring report saved: {report_path}")
            
            return report_path
            
        except Exception as e:
            raise CustomException(f"Failed to save monitoring report: {str(e)}", sys.exc_info())
    
    def run_monitoring(self, current_data_path: Path = None, baseline_f1: float = 0.50) -> Dict:
        """Run complete monitoring pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Monitoring")
            logger.info("=" * 60)
            
            # Load data
            self.load_reference_data()
            
            if current_data_path:
                self.load_current_data(current_data_path)
            else:
                # Use reference data as current for simulation
                logger.info("No current data provided. Using reference data for simulation.")
                self.current_data = self.reference_data.copy()
            
            # Detect drift
            drift_summary = self.detect_data_drift()
            
            # Simulate performance metrics (in production, use actual predictions)
            logger.info("Simulating performance metrics...")
            current_f1 = baseline_f1 - 0.02  # Simulate slight degradation
            performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "current_f1": current_f1,
                "baseline_f1": baseline_f1,
                "note": "Simulated metrics - replace with actual predictions in production"
            }
            
            # Check performance degradation
            performance_degraded, perf_message = self.check_performance_degradation(current_f1, baseline_f1)
            
            # Retraining decision
            should_retrain, retrain_message = self.should_trigger_retraining(drift_summary, performance_degraded)
            
            retraining_decision = {
                "should_retrain": should_retrain,
                "message": retrain_message,
                "performance_degraded": performance_degraded,
                "drift_detected": drift_summary['drift_percentage'] > 30
            }
            
            # Save report
            report_path = self.save_monitoring_report(drift_summary, performance_metrics, retraining_decision)
            
            logger.info("=" * 60)
            logger.info("Model Monitoring Completed")
            logger.info("=" * 60)
            
            return {
                "drift_summary": drift_summary,
                "performance_metrics": performance_metrics,
                "retraining_decision": retraining_decision,
                "report_path": str(report_path)
            }
            
        except Exception as e:
            raise CustomException(f"Monitoring pipeline failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    monitor = ModelMonitor()
    result = monitor.run_monitoring(baseline_f1=0.54)
    
    print(f"\n{'='*60}")
    print("MONITORING SUMMARY")
    print(f"{'='*60}")
    print(f"Drift Detected: {result['drift_summary']['drifted_features']}/{result['drift_summary']['total_features']} features")
    print(f"Drift Percentage: {result['drift_summary']['drift_percentage']:.2f}%")
    print(f"Should Retrain: {'YES' if result['retraining_decision']['should_retrain'] else 'NO'}")
    print(f"Report: {result['report_path']}")
    print(f"{'='*60}")
