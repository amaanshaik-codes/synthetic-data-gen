"""
Missing value injector.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from synthdata.config import MissingValuesConfig


class MissingValueInjector:
    """Inject missing values into a DataFrame."""
    
    def __init__(self, config: MissingValuesConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def inject(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """Inject missing values into the DataFrame."""
        df = df.copy()
        exclude_columns = exclude_columns or {"id", "customer_id", "transaction_id", "product_id"}
        
        # Add any column ending with '_id' to exclusions
        for col in df.columns:
            if col.endswith("_id"):
                exclude_columns.add(col)
        
        for column in df.columns:
            if column in exclude_columns:
                continue
            
            # Get missing rate for this column
            missing_rate = self.config.per_column.get(column, self.config.global_rate)
            
            if missing_rate <= 0:
                continue
            
            # Select pattern for this column
            pattern = random.choice(self.config.patterns)
            
            # Inject missing values based on pattern
            if pattern == "MCAR":
                df = self._inject_mcar(df, column, missing_rate)
            elif pattern == "MAR":
                df = self._inject_mar(df, column, missing_rate)
            elif pattern == "MNAR":
                df = self._inject_mnar(df, column, missing_rate)
        
        return df
    
    def _inject_mcar(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
    ) -> pd.DataFrame:
        """Missing Completely At Random - random missing values."""
        mask = np.random.random(len(df)) < rate
        
        # Use various null representations
        null_value = random.choice(self.config.null_representations)
        
        if null_value in ["", "NULL", "N/A", "None", "-", "?", "NA"]:
            # For non-numeric types, use NaN to avoid dtype issues
            if df[column].dtype in [np.float64, np.int64, np.float32, np.int32]:
                df.loc[mask, column] = np.nan
            elif df[column].dtype == object:
                df.loc[mask, column] = null_value
            else:
                # For datetime, bool, etc. - use None which pandas handles properly
                df[column] = df[column].astype(object)
                df.loc[mask, column] = None
        elif null_value in ["nan", "NaN"]:
            if df[column].dtype in [bool, 'bool']:
                df[column] = df[column].astype(object)
            df.loc[mask, column] = np.nan
        else:
            df.loc[mask, column] = np.nan
        
        return df
    
    def _inject_mar(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
    ) -> pd.DataFrame:
        """Missing At Random - missing depends on other observed variables."""
        # Find a related column to condition on
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols or column not in df.columns:
            return self._inject_mcar(df, column, rate)
        
        # Condition on another column - higher values more likely to be missing
        condition_col = random.choice([c for c in numeric_cols if c != column] or numeric_cols)
        
        # Create probability based on condition column
        col_values = df[condition_col].fillna(df[condition_col].median())
        percentiles = col_values.rank(pct=True)
        
        # Higher percentile = higher probability of missing
        probabilities = percentiles * rate * 2
        mask = np.random.random(len(df)) < probabilities
        
        # Handle non-numeric types that can't hold NaN directly
        if df[column].dtype in [bool, 'bool'] or not pd.api.types.is_numeric_dtype(df[column]):
            if df[column].dtype != object:
                df[column] = df[column].astype(object)
        
        df.loc[mask, column] = np.nan
        
        return df
    
    def _inject_mnar(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
    ) -> pd.DataFrame:
        """Missing Not At Random - missing depends on the unobserved value itself."""
        if df[column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            return self._inject_mcar(df, column, rate)
        
        # Missing more likely for extreme values
        col_values = df[column].fillna(df[column].median())
        z_scores = np.abs((col_values - col_values.mean()) / (col_values.std() + 1e-10))
        
        # Higher z-score = higher probability of missing
        probabilities = np.minimum(z_scores * rate, 0.9)
        mask = np.random.random(len(df)) < probabilities
        
        df.loc[mask, column] = np.nan
        
        return df
