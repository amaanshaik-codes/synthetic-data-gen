"""
Data drift injector.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import DataDriftConfig


class DataDriftInjector:
    """Inject data drift into time-series data."""
    
    def __init__(self, config: DataDriftConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def inject(
        self,
        df: pd.DataFrame,
        time_column: str = "timestamp",
    ) -> pd.DataFrame:
        """Inject data drift into the DataFrame."""
        if not self.config.enabled:
            return df
        
        if time_column not in df.columns:
            # Try to find a date column
            date_cols = [c for c in df.columns if any(
                word in c.lower() for word in ['date', 'time', 'created', 'timestamp']
            )]
            if date_cols:
                time_column = date_cols[0]
            else:
                return df
        
        df = df.copy()
        
        # Sort by time
        df = df.sort_values(time_column).reset_index(drop=True)
        
        # Determine drift start point
        drift_start_idx = int(len(df) * self.config.drift_start_percentage)
        
        # Get columns to apply drift
        drift_columns = self.config.drift_columns
        if not drift_columns:
            # Auto-select numeric columns
            drift_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            drift_columns = [c for c in drift_columns if not c.endswith('_id')]
        
        for col in drift_columns:
            if col not in df.columns:
                continue
            
            if self.config.drift_type == "gradual":
                df = self._inject_gradual_drift(df, col, drift_start_idx)
            elif self.config.drift_type == "sudden":
                df = self._inject_sudden_drift(df, col, drift_start_idx)
            elif self.config.drift_type == "seasonal":
                df = self._inject_seasonal_drift(df, col, time_column)
        
        return df
    
    def _inject_gradual_drift(
        self,
        df: pd.DataFrame,
        column: str,
        drift_start_idx: int,
    ) -> pd.DataFrame:
        """Inject gradual drift - values slowly change over time."""
        if df[column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            return df
        
        # Calculate drift amount
        col_std = df[column].std()
        if col_std == 0:
            col_std = abs(df[column].mean()) * 0.1 or 1
        
        max_drift = col_std * self.config.drift_magnitude
        
        # Apply gradual drift after start point
        drift_length = len(df) - drift_start_idx
        
        for i in range(drift_start_idx, len(df)):
            progress = (i - drift_start_idx) / drift_length
            drift_amount = max_drift * progress
            
            if pd.notna(df.loc[i, column]):
                df.loc[i, column] = df.loc[i, column] + drift_amount
        
        return df
    
    def _inject_sudden_drift(
        self,
        df: pd.DataFrame,
        column: str,
        drift_start_idx: int,
    ) -> pd.DataFrame:
        """Inject sudden drift - values shift abruptly."""
        if df[column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            return df
        
        # Calculate shift amount
        col_std = df[column].std()
        if col_std == 0:
            col_std = abs(df[column].mean()) * 0.1 or 1
        
        shift_amount = col_std * self.config.drift_magnitude * 2
        
        # Apply sudden shift after start point
        mask = df.index >= drift_start_idx
        df.loc[mask, column] = df.loc[mask, column] + shift_amount
        
        return df
    
    def _inject_seasonal_drift(
        self,
        df: pd.DataFrame,
        column: str,
        time_column: str,
    ) -> pd.DataFrame:
        """Inject seasonal drift - values follow a cyclic pattern."""
        if df[column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            return df
        
        # Calculate amplitude
        col_std = df[column].std()
        if col_std == 0:
            col_std = abs(df[column].mean()) * 0.1 or 1
        
        amplitude = col_std * self.config.drift_magnitude
        
        # Create seasonal pattern based on time
        try:
            times = pd.to_datetime(df[time_column])
            # Use day of year for seasonality
            day_of_year = times.dt.dayofyear
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # Apply seasonal adjustment
            for i in range(len(df)):
                if pd.notna(df.loc[i, column]):
                    df.loc[i, column] = df.loc[i, column] + (amplitude * seasonal_factor.iloc[i])
        except:
            pass
        
        return df
