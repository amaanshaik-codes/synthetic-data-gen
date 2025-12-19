"""
Outlier injector.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import OutliersConfig


class OutlierInjector:
    """Inject outliers and anomalies into a DataFrame."""
    
    def __init__(self, config: OutliersConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def inject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject outliers into the DataFrame."""
        if self.config.rate <= 0:
            return df
        
        df = df.copy()
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID-like columns
        numeric_cols = [c for c in numeric_cols if not c.endswith('_id') and 'id' not in c.lower()]
        
        for col in numeric_cols:
            # Get column-specific rate or use global
            col_rate = self.config.per_column.get(col, self.config.rate)
            
            if col_rate <= 0:
                continue
            
            df = self._inject_column_outliers(df, col, col_rate)
        
        return df
    
    def _inject_column_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
    ) -> pd.DataFrame:
        """Inject outliers into a specific column."""
        # Calculate statistics
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            return df
        
        mean = col_data.mean()
        std = col_data.std()
        
        if std == 0:
            std = abs(mean) * 0.1 if mean != 0 else 1
        
        # Select rows to make outliers
        num_outliers = int(len(df) * rate)
        outlier_indices = np.random.choice(df.index, size=num_outliers, replace=False)
        
        for idx in outlier_indices:
            outlier_type = random.choice(self.config.outlier_types)
            
            if outlier_type == "extreme_high":
                # Generate extremely high value
                multiplier = random.uniform(self.config.magnitude, self.config.magnitude * 2)
                new_value = mean + (std * multiplier)
            
            elif outlier_type == "extreme_low":
                # Generate extremely low value
                multiplier = random.uniform(self.config.magnitude, self.config.magnitude * 2)
                new_value = mean - (std * multiplier)
                
                # Ensure non-negative for certain column types
                if any(word in column.lower() for word in ['amount', 'price', 'count', 'quantity', 'age']):
                    new_value = max(0, new_value)
            
            elif outlier_type == "impossible_values":
                # Generate impossible values based on column name
                new_value = self._generate_impossible_value(column, mean, std)
            
            else:
                # Default to extreme high
                multiplier = random.uniform(self.config.magnitude, self.config.magnitude * 2)
                new_value = mean + (std * multiplier)
            
            df.loc[idx, column] = new_value
        
        return df
    
    def _generate_impossible_value(
        self,
        column: str,
        mean: float,
        std: float,
    ) -> float:
        """Generate an impossible or clearly erroneous value."""
        column_lower = column.lower()
        
        # Age-related columns
        if 'age' in column_lower:
            return random.choice([-5, 0, 150, 200, 999])
        
        # Percentage columns
        if any(word in column_lower for word in ['rate', 'percentage', 'pct', 'ratio']):
            return random.choice([-10, 150, 500, 1000])
        
        # Rating columns
        if 'rating' in column_lower:
            return random.choice([-1, 0, 6, 10, 100])
        
        # Count columns
        if any(word in column_lower for word in ['count', 'num', 'quantity']):
            return random.choice([-100, -1, 999999])
        
        # Price/amount columns
        if any(word in column_lower for word in ['price', 'amount', 'cost', 'value']):
            return random.choice([-1000, 0.001, mean * 100, 9999999])
        
        # Duration columns
        if any(word in column_lower for word in ['hours', 'minutes', 'seconds', 'duration']):
            return random.choice([-24, -1, 10000, 999999])
        
        # Default: just use extreme values
        return mean + (std * self.config.magnitude * 3)
