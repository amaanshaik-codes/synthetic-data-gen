"""
Noise injector for features and labels.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import NoiseConfig


class NoiseInjector:
    """Inject noise into features and labels."""
    
    def __init__(self, config: NoiseConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def inject_label_noise(
        self,
        df: pd.DataFrame,
        label_column: str,
    ) -> pd.DataFrame:
        """Inject noise into the label column."""
        if self.config.label_noise_rate <= 0:
            return df
        
        if label_column not in df.columns:
            return df
        
        df = df.copy()
        
        # Get unique values of the label
        unique_labels = df[label_column].dropna().unique()
        
        if len(unique_labels) < 2:
            return df
        
        # Select rows to flip labels
        num_to_flip = int(len(df) * self.config.label_noise_rate)
        flip_indices = np.random.choice(df.index, size=num_to_flip, replace=False)
        
        for idx in flip_indices:
            current_label = df.loc[idx, label_column]
            if pd.notna(current_label):
                # Choose a different label
                other_labels = [l for l in unique_labels if l != current_label]
                if other_labels:
                    df.loc[idx, label_column] = random.choice(other_labels)
        
        return df
    
    def inject_feature_noise(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Inject noise into numeric feature columns."""
        if self.config.feature_noise_rate <= 0:
            return df
        
        df = df.copy()
        exclude_columns = exclude_columns or []
        
        # Add ID columns to exclusions
        exclude_columns.extend([c for c in df.columns if c.endswith('_id')])
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_columns]
        
        for col in numeric_cols:
            df = self._inject_noise_to_column(df, col)
        
        return df
    
    def _inject_noise_to_column(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.DataFrame:
        """Inject noise to a specific column."""
        col_data = df[column].copy()
        non_null_mask = col_data.notna()
        
        if non_null_mask.sum() == 0:
            return df
        
        std = col_data[non_null_mask].std()
        if std == 0:
            std = abs(col_data[non_null_mask].mean()) * 0.1
            if std == 0:
                std = 1
        
        # Generate noise based on type
        noise_scale = std * self.config.feature_noise_rate
        
        if self.config.noise_type == "gaussian":
            noise = np.random.normal(0, noise_scale, len(df))
        elif self.config.noise_type == "uniform":
            noise = np.random.uniform(-noise_scale, noise_scale, len(df))
        elif self.config.noise_type == "salt_pepper":
            noise = np.zeros(len(df))
            salt_pepper_mask = np.random.random(len(df)) < self.config.feature_noise_rate
            noise[salt_pepper_mask] = np.random.choice([-1, 1], salt_pepper_mask.sum()) * std * 3
        else:
            noise = np.random.normal(0, noise_scale, len(df))
        
        # Apply noise only to non-null values
        df.loc[non_null_mask, column] = col_data[non_null_mask] + noise[non_null_mask]
        
        return df
