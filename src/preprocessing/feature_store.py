"""
Feature Store
=============
Manages reading processed features for model training.
In production: connects to Azure ADLS / Databricks Feature Store.
Locally: reads from Parquet files.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from src.utils.config import get_config


class FeatureStore:
    """
    Reads processed features and prepares train/test splits.
    Abstracts data source — swap Parquet for ADLS in production.
    """

    def __init__(self):
        self.config = get_config()
        self.processed_dir = self.config.data["processed_dir"]
        self.prep_cfg = self.config.preprocessing

    def load_features(self) -> pd.DataFrame:
        """Load processed Parquet features into pandas for training."""
        parquet_path = os.path.join(self.processed_dir, "train_processed")

        if not os.path.exists(parquet_path):
            logger.error(f"❌ Processed data not found: {parquet_path}")
            logger.info("   Run spark_pipeline.py first!")
            raise FileNotFoundError(f"Processed data not found: {parquet_path}")

        logger.info(f"📂 Loading features from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    def get_train_test_split(self):
        """
        Load features and return train/test split.
        Uses GroupShuffleSplit to avoid data leakage between engines.
        """
        df = self.load_features()

        # Feature columns — exclude metadata and target
        exclude_cols = ["RUL", "engine_id", "cycle"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["RUL"]
        groups = df["engine_id"]

        # Group split — same engine never in both train and test
        gss = GroupShuffleSplit(
            test_size=self.prep_cfg["test_size"],
            random_state=self.prep_cfg["random_state"]
        )

        for train_idx, test_idx in gss.split(X, y, groups):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

        logger.info(f"✅ Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        logger.info(f"   Features: {len(feature_cols)}")

        return X_train, X_test, y_train, y_test, feature_cols

    def get_feature_names(self) -> list:
        """Return list of all feature column names."""
        df = self.load_features()
        exclude_cols = ["RUL", "engine_id", "cycle"]
        return [c for c in df.columns if c not in exclude_cols]
