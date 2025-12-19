"""
Duplicate record injector.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import DuplicatesConfig


class DuplicateInjector:
    """Inject duplicate records into a DataFrame."""
    
    def __init__(self, config: DuplicatesConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def inject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject duplicate records into the DataFrame."""
        if self.config.rate <= 0:
            return df
        
        df = df.copy()
        num_duplicates = int(len(df) * self.config.rate)
        
        if num_duplicates == 0:
            return df
        
        # Select random rows to duplicate
        duplicate_indices = np.random.choice(df.index, size=num_duplicates, replace=True)
        duplicates = df.loc[duplicate_indices].copy()
        
        if self.config.exact_duplicates and not self.config.near_duplicates:
            # Exact duplicates - just copy the rows
            pass
        elif self.config.near_duplicates:
            # Near duplicates - introduce slight variations
            duplicates = self._create_near_duplicates(duplicates)
        
        # Append duplicates to the original DataFrame
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Shuffle to mix duplicates throughout
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df
    
    def _create_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create near-duplicates with slight variations."""
        df = df.copy()
        
        # Determine which columns to modify
        if self.config.near_duplicate_columns:
            columns_to_modify = [c for c in self.config.near_duplicate_columns if c in df.columns]
        else:
            # Auto-select columns that are good candidates for near-duplication
            columns_to_modify = []
            for col in df.columns:
                if col.endswith("_id"):
                    continue
                if df[col].dtype == object:
                    columns_to_modify.append(col)
                elif df[col].dtype in [np.float64, np.float32]:
                    columns_to_modify.append(col)
        
        # Skip if no columns to modify
        if not columns_to_modify:
            return df
        
        for idx in df.index:
            # Modify 1-3 columns per duplicate
            num_cols_to_modify = random.randint(1, min(3, len(columns_to_modify)))
            cols_to_modify = random.sample(columns_to_modify, num_cols_to_modify)
            
            for col in cols_to_modify:
                original_value = df.loc[idx, col]
                
                if pd.isna(original_value):
                    continue
                
                if isinstance(original_value, str):
                    df.loc[idx, col] = self._modify_string(original_value)
                elif isinstance(original_value, (int, float)):
                    df.loc[idx, col] = self._modify_number(original_value)
        
        return df
    
    def _modify_string(self, value: str) -> str:
        """Introduce slight modifications to a string."""
        if not value or len(value) < 2:
            return value
        
        modification = random.choice(["typo", "case", "whitespace", "truncate"])
        
        if modification == "typo":
            # Swap two adjacent characters
            if len(value) > 2:
                pos = random.randint(0, len(value) - 2)
                value = value[:pos] + value[pos + 1] + value[pos] + value[pos + 2:]
        
        elif modification == "case":
            # Change case
            if random.random() > 0.5:
                value = value.lower()
            else:
                value = value.upper()
        
        elif modification == "whitespace":
            # Add leading/trailing space
            if random.random() > 0.5:
                value = " " + value
            else:
                value = value + " "
        
        elif modification == "truncate":
            # Slightly truncate
            if len(value) > 5:
                value = value[:-1]
        
        return value
    
    def _modify_number(self, value: float) -> float:
        """Introduce slight modifications to a number."""
        # Small random perturbation
        perturbation = random.uniform(-0.01, 0.01) * abs(value)
        return value + perturbation
