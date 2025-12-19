"""
Main data quality injector that combines all quality injection methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pandas as pd

from synthdata.config import DataQualityConfig, Difficulty
from synthdata.quality.missing import MissingValueInjector
from synthdata.quality.duplicates import DuplicateInjector
from synthdata.quality.inconsistencies import InconsistencyInjector
from synthdata.quality.outliers import OutlierInjector
from synthdata.quality.noise import NoiseInjector
from synthdata.quality.drift import DataDriftInjector


class DataQualityInjector:
    """Main class to inject data quality issues into DataFrames."""
    
    def __init__(
        self,
        config: DataQualityConfig,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.difficulty = difficulty
        self.seed = seed
        
        # Initialize sub-injectors
        self.missing_injector = MissingValueInjector(config.missing_values, seed)
        self.duplicate_injector = DuplicateInjector(config.duplicates, seed)
        self.inconsistency_injector = InconsistencyInjector(config.inconsistencies, seed)
        self.outlier_injector = OutlierInjector(config.outliers, seed)
        self.noise_injector = NoiseInjector(config.noise, seed)
        self.drift_injector = DataDriftInjector(config.data_drift, seed)
    
    def inject(
        self,
        df: pd.DataFrame,
        table_name: str = "",
        primary_key: Optional[str] = None,
        exclude_columns: Optional[Set[str]] = None,
        label_column: Optional[str] = None,
        time_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Inject data quality issues into a DataFrame.
        
        Args:
            df: The DataFrame to inject issues into
            table_name: Name of the table (for logging)
            primary_key: Primary key column to never modify
            exclude_columns: Columns to exclude from modifications
            label_column: Target/label column for label noise
            time_column: Time column for drift injection
        
        Returns:
            DataFrame with injected quality issues
        """
        exclude_columns = exclude_columns or set()
        
        # Always exclude primary key and ID columns
        if primary_key:
            exclude_columns.add(primary_key)
        
        # Add all ID columns to exclusions
        for col in df.columns:
            if col.endswith("_id") or col == "id":
                exclude_columns.add(col)
        
        # Apply injections in order
        df = self._apply_injections(
            df,
            exclude_columns,
            label_column,
            time_column,
        )
        
        return df
    
    def _apply_injections(
        self,
        df: pd.DataFrame,
        exclude_columns: Set[str],
        label_column: Optional[str],
        time_column: Optional[str],
    ) -> pd.DataFrame:
        """Apply all data quality injections."""
        
        # 1. Inject duplicates (do this first to avoid duplicating issues)
        df = self.duplicate_injector.inject(df)
        
        # 2. Inject missing values
        df = self.missing_injector.inject(df, exclude_columns)
        
        # 3. Inject inconsistencies (formats, typos, etc.)
        df = self.inconsistency_injector.inject(df)
        
        # 4. Inject outliers
        df = self.outlier_injector.inject(df)
        
        # 5. Inject noise
        if label_column:
            df = self.noise_injector.inject_label_noise(df, label_column)
        df = self.noise_injector.inject_feature_noise(df, list(exclude_columns))
        
        # 6. Inject data drift
        if time_column:
            df = self.drift_injector.inject(df, time_column)
        
        return df
    
    def get_quality_report(self, df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
        """Generate a report of injected quality issues."""
        report = {
            "total_rows": len(df),
            "original_rows": len(original_df),
            "duplicates_added": len(df) - len(original_df),
            "columns": {},
        }
        
        for col in df.columns:
            col_report = {
                "missing_count": df[col].isna().sum(),
                "missing_rate": df[col].isna().mean(),
            }
            
            if col in original_df.columns:
                orig_missing = original_df[col].isna().sum()
                col_report["missing_injected"] = col_report["missing_count"] - orig_missing
            
            report["columns"][col] = col_report
        
        return report


def apply_difficulty_preset(
    config: DataQualityConfig,
    difficulty: Difficulty,
) -> DataQualityConfig:
    """Apply difficulty preset to data quality configuration."""
    presets = {
        Difficulty.EASY: {
            "missing_rate": 0.02,
            "duplicate_rate": 0.005,
            "outlier_rate": 0.01,
            "typo_rate": 0.01,
            "noise_rate": 0.0,
        },
        Difficulty.MEDIUM: {
            "missing_rate": 0.08,
            "duplicate_rate": 0.02,
            "outlier_rate": 0.03,
            "typo_rate": 0.03,
            "noise_rate": 0.02,
        },
        Difficulty.HARD: {
            "missing_rate": 0.15,
            "duplicate_rate": 0.05,
            "outlier_rate": 0.05,
            "typo_rate": 0.05,
            "noise_rate": 0.05,
        },
        Difficulty.CHAOTIC: {
            "missing_rate": 0.25,
            "duplicate_rate": 0.10,
            "outlier_rate": 0.08,
            "typo_rate": 0.10,
            "noise_rate": 0.10,
        },
    }
    
    preset = presets.get(difficulty, presets[Difficulty.MEDIUM])
    
    config.missing_values.global_rate = preset["missing_rate"]
    config.duplicates.rate = preset["duplicate_rate"]
    config.outliers.rate = preset["outlier_rate"]
    config.inconsistencies.typo_rate = preset["typo_rate"]
    config.noise.label_noise_rate = preset["noise_rate"]
    config.noise.feature_noise_rate = preset["noise_rate"]
    
    # Enable/disable features based on difficulty
    if difficulty == Difficulty.EASY:
        config.inconsistencies.date_formats = False
        config.inconsistencies.currency_formats = False
        config.inconsistencies.category_typos = False
        config.data_drift.enabled = False
    elif difficulty == Difficulty.CHAOTIC:
        config.inconsistencies.encoding_issues = True
        config.duplicates.near_duplicates = True
        config.data_drift.enabled = True
    
    return config
