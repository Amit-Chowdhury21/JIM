"""
Feature Pruner (V2.0 Overhaul)
==============================
Analyzes the 50+ generated features to identify the "Top 20" most predictive ones.
Uses:
1. Pairwise Correlation Pruning (>0.90) to remove duplicates.
2. RandomForest Permutation Importance to rank features by their actual impact on accuracy.
"""

import pandas as pd
import numpy as np
from loguru import logger
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FeatureSelector:
    def __init__(self, corr_threshold=0.90, top_k=20):
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.selected_features = []

    def correlation_prune(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Removes highly collinear columns to reduce redundancy."""
        logger.info(f"Starting correlation pruning (threshold > {self.corr_threshold})...")
        corr_matrix = feature_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        
        return feature_df.drop(columns=to_drop)

    def permutation_importance(self, model, X_val, y_val):
        """Shuffles one feature at a time to measure true importance."""
        logger.info("Running Permutation Importance (this may take a few minutes)...")
        # Subsample to prevent Windows ArrayMemoryError
        if len(X_val) > 50000:
            logger.info("Subsampling validation set to 50,000 rows for memory-safe permutation importance...")
            sample_idx = np.random.choice(len(X_val), 50000, replace=False)
            X_val_sample = X_val.iloc[sample_idx]
            y_val_sample = y_val.iloc[sample_idx]
        else:
            X_val_sample = X_val
            y_val_sample = y_val
            
        result = permutation_importance(model, X_val_sample, y_val_sample, n_repeats=5, random_state=42, n_jobs=2)
        importance_df = pd.DataFrame({
            "feature": X_val_sample.columns,
            "importance": result.importances_mean,
            "std": result.importances_std
        }).sort_values("importance", ascending=False)
        return importance_df

    def select_top_features(self, importance_table: pd.DataFrame) -> list:
        """Keeps features that actively add value (importance > 0)."""
        valid_features = importance_table[importance_table["importance"] > 0]
        top_features = valid_features["feature"].tolist()
        self.selected_features = top_features
        logger.info(f"Selected {len(top_features)} valuable features (importance > 0).")
        return top_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="DataBase/.cache/merged_5min_all_assets.parquet")
    parser.add_argument("--k", type=int, default=20, help="Number of features to keep")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Please run resampler first.")
        return

    logger.info(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    # We need to run FeatureEngineer to get the 58 features on the 5m data
    from src.models.lstm_features import LSTMFeatureEngineer
    logger.info("Calculating all 58 features on 5m data...")
    engineer = LSTMFeatureEngineer()
    features_df = engineer.transform(df)
    
    # We need the original close prices to calculate the future return proxy label.
    # The transform function drops NaN lookback rows, so we align indices.
    close_prices = df["close"].loc[features_df.index]
    
    logger.info("Generating proxy directional labels for feature evaluation...")
    y = (close_prices.shift(-5) > close_prices).astype(int)
    
    # Swap df to the engineered features
    df = features_df
    
    # Exclude basic OHLCV and macro raw series from feature set
    exclude_cols = ["open", "high", "low", "close", "volume", "dxy", "gvz", "us10y", "silver"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Drop rows with NaN (from shifting)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    selector = FeatureSelector(corr_threshold=0.90, top_k=args.k)
    
    # 1. Correlation Pruning
    X_pruned = selector.correlation_prune(X)
    
    # 2. Train RF Model
    logger.info("Training baseline RandomForestClassifier...")
    split_idx = int(len(X_pruned) * 0.8)
    X_train, X_val = X_pruned.iloc[:split_idx], X_pruned.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    logger.info(f"Baseline RF Validation Accuracy: {val_acc:.4f}")
    
    # 3. Permutation Importance
    importance_table = selector.permutation_importance(model, X_val, y_val)
    logger.info("\n" + importance_table.to_string())
    
    # 4. Final Selection
    top_features = selector.select_top_features(importance_table)
    
    # Save the selected features to a JSON config
    config_path = PROJECT_ROOT / "models" / "v2_selected_features.json"
    with open(config_path, "w") as f:
        json.dump({"selected_features": top_features}, f, indent=4)
        
    logger.info(f"✅ Saved {len(top_features)} mathematically validated features to {config_path}")

if __name__ == "__main__":
    main()
