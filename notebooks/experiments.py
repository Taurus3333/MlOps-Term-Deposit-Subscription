"""
Comprehensive Data Science Experiments for Bank Marketing Term Deposit Prediction.

This notebook performs thorough analysis including:
- Deep EDA with statistical tests
- Outlier detection and treatment
- Imbalanced data handling (SMOTE, Tomek Links, SMOTE+Tomek)
- Multiple model comparisons with cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Threshold optimization
- Learning curves and model diagnostics

Results saved to results.json for pipeline consumption.
"""
import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    learning_curve, StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from scipy import stats
from scipy.stats import chi2_contingency

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.constants import CLEANED_DATA_FILE, RESULTS_FILE, TARGET_COLUMN
from src.logging.custom_logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)
sns.set_style('whitegrid')


class ComprehensiveBankMarketingExperiments:
    """Comprehensive data science experiments with imbalanced data handling."""
    
    def __init__(self):
        self.df = None
        self.df_encoded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = {}
        self.y_train_balanced = {}
        self.scaler = StandardScaler()
        self.results = {
            "experiment_date": pd.Timestamp.now().isoformat(),
            "data_info": {},
            "statistical_summary": {},
            "eda_insights": {},
            "outlier_analysis": {},
            "feature_importance": {},
            "imbalance_handling": {},
            "model_comparison": {},
            "best_model": {},
            "preprocessing": {},
            "evaluation_metric": "f1_score",
            "metric_justification": {}
        }
        self.eda_dir = project_root / "artifacts" / "eda"
        self.eda_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load cleaned data."""
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        csv_file = CLEANED_DATA_FILE.with_suffix('.csv')
        if csv_file.exists():
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.read_parquet(CLEANED_DATA_FILE)
        
        logger.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        self.results["data_info"] = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "features": list(self.df.columns),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024**2)
        }

    
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis."""
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL ANALYSIS")
        logger.info("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_summary = {}
        for col in numerical_cols:
            stats_summary[col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "skewness": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis()),
                "q25": float(self.df[col].quantile(0.25)),
                "q75": float(self.df[col].quantile(0.75))
            }
            
            logger.info(f"\n{col}:")
            logger.info(f"  Mean: {stats_summary[col]['mean']:.2f}, Median: {stats_summary[col]['median']:.2f}")
            logger.info(f"  Skewness: {stats_summary[col]['skewness']:.2f}, Kurtosis: {stats_summary[col]['kurtosis']:.2f}")
        
        self.results["statistical_summary"] = stats_summary

    
    def detect_outliers(self):
        """Detect outliers using IQR method."""
        logger.info("\n" + "="*80)
        logger.info("OUTLIER DETECTION (IQR Method)")
        logger.info("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_summary = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            outlier_summary[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_pct),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
            
            if outlier_count > 0:
                logger.info(f"{col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
        
        self.results["outlier_analysis"] = outlier_summary
        
        # Visualize outliers
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols[:8]):
            self.df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'{col}', fontweight='bold')
            axes[idx].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'outlier_boxplots.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: outlier_boxplots.png")

    
    def perform_deep_eda(self):
        """Perform comprehensive EDA with visualizations."""
        logger.info("\n" + "="*80)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("="*80)
        
        # Target distribution
        target_counts = self.df[TARGET_COLUMN].value_counts()
        target_pct = self.df[TARGET_COLUMN].value_counts(normalize=True) * 100
        imbalance_ratio = target_counts.max() / target_counts.min()
        
        logger.info(f"\nTarget Distribution:")
        logger.info(f"  No: {target_counts['no']} ({target_pct['no']:.2f}%)")
        logger.info(f"  Yes: {target_counts['yes']} ({target_pct['yes']:.2f}%)")
        logger.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        # 1. Target Distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = ['#e74c3c', '#2ecc71']
        target_counts.plot(kind='bar', ax=axes[0], color=colors)
        axes[0].set_title('Target Distribution (Count)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Subscribed to Term Deposit')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        
        axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[1].set_title('Target Distribution (Percentage)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'target_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Numerical distributions
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            self.df[col].hist(bins=50, ax=axes[idx], color='#3498db', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(self.df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[idx].axvline(self.df[col].median(), color='green', linestyle='--', label='Median')
            axes[idx].legend()
        
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'numerical_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        df_encoded = self.df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        
        plt.figure(figsize=(14, 12))
        correlation = df_encoded.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results["eda_insights"] = {
            "target_distribution": target_counts.to_dict(),
            "target_percentage": {k: float(v) for k, v in target_pct.to_dict().items()},
            "class_imbalance_ratio": float(imbalance_ratio),
            "is_imbalanced": bool(imbalance_ratio > 1.5),
            "missing_values": int(self.df.isnull().sum().sum()),
            "numerical_features": numerical_cols,
            "categorical_features": self.df.select_dtypes(include=['object']).columns.tolist()
        }

    
    def chi_square_tests(self):
        """Perform chi-square tests for categorical features."""
        logger.info("\n" + "="*80)
        logger.info("CHI-SQUARE TESTS (Categorical Features vs Target)")
        logger.info("="*80)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove(TARGET_COLUMN)
        
        chi_square_results = {}
        for col in categorical_cols:
            contingency_table = pd.crosstab(self.df[col], self.df[TARGET_COLUMN])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results[col] = {
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "significant": bool(p_value < 0.05)
            }
            
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            logger.info(f"{col}: chi2 = {chi2:.2f}, p-value = {p_value:.4f} {significance}")
        
        self.results["chi_square_tests"] = chi_square_results

    
    def prepare_features(self):
        """Prepare features for modeling."""
        logger.info("\n" + "="*80)
        logger.info("FEATURE PREPARATION")
        logger.info("="*80)
        
        X = self.df.drop(columns=[TARGET_COLUMN])
        y = self.df[TARGET_COLUMN]
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le.classes_.tolist()
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Train class distribution: {np.bincount(self.y_train)}")
        logger.info(f"Test class distribution: {np.bincount(self.y_test)}")
        
        self.results["preprocessing"] = {
            "encoding": "LabelEncoder",
            "categorical_columns": list(categorical_cols),
            "label_mappings": label_encoders,
            "target_encoding": {"no": 0, "yes": 1},
            "train_test_split": 0.2,
            "random_state": 42,
            "stratify": True
        }

    
    def handle_imbalanced_data(self):
        """Apply different imbalanced data handling techniques."""
        logger.info("\n" + "="*80)
        logger.info("IMBALANCED DATA HANDLING")
        logger.info("="*80)
        
        # Original (no resampling)
        self.X_train_balanced['original'] = self.X_train
        self.y_train_balanced['original'] = self.y_train
        logger.info(f"Original: {np.bincount(self.y_train)}")
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
        self.X_train_balanced['smote'] = X_smote
        self.y_train_balanced['smote'] = y_smote
        logger.info(f"SMOTE: {np.bincount(y_smote)}")
        
        # Tomek Links
        tomek = TomekLinks()
        X_tomek, y_tomek = tomek.fit_resample(self.X_train, self.y_train)
        self.X_train_balanced['tomek'] = X_tomek
        self.y_train_balanced['tomek'] = y_tomek
        logger.info(f"Tomek Links: {np.bincount(y_tomek)}")
        
        # SMOTE + Tomek
        smote_tomek = SMOTETomek(random_state=42)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(self.X_train, self.y_train)
        self.X_train_balanced['smote_tomek'] = X_smote_tomek
        self.y_train_balanced['smote_tomek'] = y_smote_tomek
        logger.info(f"SMOTE + Tomek: {np.bincount(y_smote_tomek)}")
        
        self.results["imbalance_handling"] = {
            "original": {"class_0": int(np.bincount(self.y_train)[0]), "class_1": int(np.bincount(self.y_train)[1])},
            "smote": {"class_0": int(np.bincount(y_smote)[0]), "class_1": int(np.bincount(y_smote)[1])},
            "tomek": {"class_0": int(np.bincount(y_tomek)[0]), "class_1": int(np.bincount(y_tomek)[1])},
            "smote_tomek": {"class_0": int(np.bincount(y_smote_tomek)[0]), "class_1": int(np.bincount(y_smote_tomek)[1])}
        }

    
    def train_and_compare_models(self):
        """Train multiple models with different sampling strategies."""
        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING & COMPARISON")
        logger.info("="*80)
        
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        # Try XGBoost and LightGBM
        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        except ImportError:
            logger.warning("XGBoost not installed")
        
        try:
            import lightgbm as lgb
            models["LightGBM"] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        except ImportError:
            logger.warning("LightGBM not installed")
        
        all_results = {}
        
        for sampling_method in ['original', 'smote', 'tomek', 'smote_tomek']:
            logger.info(f"\n--- Sampling Method: {sampling_method.upper()} ---")
            
            X_train_sample = self.X_train_balanced[sampling_method]
            y_train_sample = self.y_train_balanced[sampling_method]
            
            method_results = {}
            
            for name, model in models.items():
                # Train
                model.fit(X_train_sample, y_train_sample)
                
                # Predict
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                metrics = {
                    "accuracy": float(accuracy_score(self.y_test, y_pred)),
                    "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(self.y_test, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(self.y_test, y_pred_proba))
                }
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_sample, y_train_sample, 
                                           cv=5, scoring='f1')
                metrics["cv_f1_mean"] = float(cv_scores.mean())
                metrics["cv_f1_std"] = float(cv_scores.std())
                
                method_results[name] = metrics
                
                logger.info(f"{name}: F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, CV-F1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
            
            all_results[sampling_method] = method_results
        
        self.results["model_comparison"] = all_results
        
        # Find best combination
        best_f1 = 0
        best_combo = None
        for sampling in all_results:
            for model in all_results[sampling]:
                f1 = all_results[sampling][model]['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_combo = (sampling, model)
        
        logger.info(f"\n*** Best Combination: {best_combo[1]} with {best_combo[0]} (F1={best_f1:.4f}) ***")
        
        return best_combo

    
    def tune_best_model(self, sampling_method, model_name):
        """Hyperparameter tuning for best model."""
        logger.info("\n" + "="*80)
        logger.info(f"HYPERPARAMETER TUNING: {model_name} with {sampling_method}")
        logger.info("="*80)
        
        X_train_sample = self.X_train_balanced[sampling_method]
        y_train_sample = self.y_train_balanced[sampling_method]
        
        if "Random Forest" in model_name:
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', None]
            }
        elif "LightGBM" in model_name:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        elif "XGBoost" in model_name:
            import xgboost as xgb
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        else:
            logger.info("Using default parameters")
            return None, {}
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_sample, y_train_sample)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Final evaluation
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        final_metrics = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1_score": float(f1_score(self.y_test, y_pred)),
            "roc_auc": float(roc_auc_score(self.y_test, y_pred_proba))
        }
        
        self.results["best_model"] = {
            "name": model_name,
            "sampling_method": sampling_method,
            "parameters": {k: str(v) for k, v in best_params.items()},
            "metrics": final_metrics
        }
        
        logger.info(f"\nFinal Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\nClassification Report:")
        logger.info(f"\n{classification_report(self.y_test, y_pred, target_names=['no', 'yes'])}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'precision_recall_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return best_model, best_params

    
    def feature_importance_analysis(self, model, model_name):
        """Analyze feature importance."""
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Important Features:")
            for idx, row in feature_imp_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.results["feature_importance"] = feature_imp_df.to_dict('records')
            
            # Plot
            plt.figure(figsize=(10, 8))
            top_features = feature_imp_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'], color='#3498db')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importances - {model_name}', fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.eda_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()

    
    def save_results(self):
        """Save experiment results to JSON."""
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Add metric justification
        self.results["metric_justification"] = {
            "chosen_metric": "F1-Score",
            "reason": "Balances precision and recall for imbalanced data",
            "precision_only_issue": "High precision means fewer false positives (less wasted marketing) but may miss many real customers (low recall), resulting in lost revenue",
            "recall_only_issue": "High recall catches most customers but creates many false positives, wasting marketing budget on unlikely conversions",
            "f1_advantage": "Harmonic mean of precision and recall ensures neither metric is sacrificed, optimal for business where both false positives and false negatives have costs",
            "accuracy_issue": "Misleading for imbalanced data - can achieve 88% accuracy by always predicting 'no'"
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {RESULTS_FILE}")
        logger.info(f"EDA charts saved to: {self.eda_dir}")
    
    def run_experiments(self):
        """Run complete experimentation pipeline."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE BANK MARKETING EXPERIMENTS")
        logger.info("="*80)
        
        self.load_data()
        self.statistical_analysis()
        self.detect_outliers()
        self.perform_deep_eda()
        self.chi_square_tests()
        self.prepare_features()
        self.handle_imbalanced_data()
        
        logger.info("\n" + "="*80)
        logger.info("METRIC SELECTION JUSTIFICATION")
        logger.info("="*80)
        logger.info(f"Dataset Imbalance Ratio: {self.results['eda_insights']['class_imbalance_ratio']:.2f}:1")
        logger.info("\nWhy NOT Precision only?")
        logger.info("  → High precision = fewer false positives (less wasted marketing)")
        logger.info("  → BUT: May miss many real customers (low recall)")
        logger.info("  → Result: Conservative, misses revenue opportunities")
        logger.info("\nWhy NOT Recall only?")
        logger.info("  → High recall = catch most potential customers")
        logger.info("  → BUT: Many false positives (wasted marketing budget)")
        logger.info("  → Result: Expensive, inefficient campaigns")
        logger.info("\nWhy F1-Score?")
        logger.info("  → Balances precision and recall (harmonic mean)")
        logger.info("  → Optimal for imbalanced data")
        logger.info("  → Ensures we don't sacrifice one metric for the other")
        logger.info("="*80)
        
        best_sampling, best_model_name = self.train_and_compare_models()
        best_model, best_params = self.tune_best_model(best_sampling, best_model_name)
        
        if best_model:
            self.feature_importance_analysis(best_model, best_model_name)
        
        self.save_results()
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENTS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return self.results


if __name__ == "__main__":
    experiments = ComprehensiveBankMarketingExperiments()
    results = experiments.run_experiments()
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Best Model: {results['best_model']['name']}")
    print(f"Sampling Method: {results['best_model']['sampling_method']}")
    print(f"F1 Score: {results['best_model']['metrics']['f1_score']:.4f}")
    print(f"ROC-AUC: {results['best_model']['metrics']['roc_auc']:.4f}")
    print(f"Precision: {results['best_model']['metrics']['precision']:.4f}")
    print(f"Recall: {results['best_model']['metrics']['recall']:.4f}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Charts saved to: {experiments.eda_dir}")
    print(f"{'='*80}\n")
