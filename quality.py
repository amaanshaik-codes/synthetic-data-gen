"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  QUALITY INJECTOR - Makes synthetic data realistically messy                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import random
from typing import List, Optional

import numpy as np
import pandas as pd


class QualityInjector:
    """Injects realistic quality issues into clean data."""
    
    # Columns to skip when injecting issues
    PROTECTED_COLUMNS = {"customer_id", "product_id", "transaction_id", "ticket_id"}
    
    # Common typos/inconsistencies
    TYPO_MAP = {
        "USA": ["US", "U.S.A", "United States", "usa", "U.S.", "UNITED STATES"],
        "UK": ["U.K.", "United Kingdom", "Great Britain", "uk", "GB"],
        "Canada": ["CA", "canada", "CANADA", "CAN"],
        "Germany": ["DE", "germany", "Deutschland", "GERMANY"],
        "France": ["FR", "france", "FRANCE"],
        "Australia": ["AU", "australia", "AUSTRALIA", "AUS"],
    }
    
    DATE_FORMATS = [
        "%Y-%m-%d",      # 2024-01-15
        "%d/%m/%Y",      # 15/01/2024
        "%m/%d/%Y",      # 01/15/2024
        "%d-%m-%Y",      # 15-01-2024
        "%Y/%m/%d",      # 2024/01/15
        "%b %d, %Y",     # Jan 15, 2024
        "%d %b %Y",      # 15 Jan 2024
    ]
    
    def __init__(self, quality_rate: float = 0.08, seed: int = 42):
        """
        Initialize quality injector.
        
        Args:
            quality_rate: Overall rate of quality issues (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.quality_rate = quality_rate
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def inject(self, df: pd.DataFrame, table_name: str = "data") -> pd.DataFrame:
        """
        Inject all quality issues into a DataFrame.
        
        Args:
            df: Input DataFrame
            table_name: Name of the table (for context-aware injection)
        
        Returns:
            DataFrame with quality issues injected
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Apply different types of quality issues
        df = self._inject_missing_values(df)
        df = self._inject_duplicates(df)
        df = self._inject_outliers(df)
        df = self._inject_inconsistencies(df)
        df = self._inject_typos(df)
        df = self._inject_date_format_issues(df)
        df = self._inject_case_issues(df)
        df = self._inject_whitespace_issues(df)
        
        return df
    
    def _inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject missing values (NaN, None, empty strings)."""
        missing_rate = self.quality_rate * 0.8  # 80% of quality rate
        
        for col in df.columns:
            if col in self.PROTECTED_COLUMNS:
                continue
            
            # Skip boolean columns
            if df[col].dtype == 'bool':
                continue
            
            # Vary missing rate by column (some columns more likely to be missing)
            col_rate = missing_rate * random.uniform(0.5, 1.5)
            col_rate = min(col_rate, 0.3)  # Cap at 30%
            
            mask = np.random.random(len(df)) < col_rate
            num_missing = mask.sum()
            
            if num_missing > 0:
                # Use different missing value representations
                if df[col].dtype == 'object':
                    # For string columns: NaN, empty string, "N/A", "null", etc.
                    missing_values = [
                        np.nan,      # 50%
                        "",          # 20%
                        "N/A",       # 10%
                        "null",      # 10%
                        "None",      # 5%
                        "-",         # 5%
                    ]
                    weights = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
                    
                    values = random.choices(missing_values, weights=weights, k=num_missing)
                    df.loc[mask, col] = values
                else:
                    # For numeric columns: NaN - need to convert to float first
                    df[col] = df[col].astype(float)
                    df.loc[mask, col] = np.nan
        
        return df
    
    def _inject_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject duplicate rows."""
        if len(df) < 10:
            return df
        
        dup_rate = self.quality_rate * 0.3  # 30% of quality rate for duplicates
        num_dups = int(len(df) * dup_rate)
        
        if num_dups > 0:
            # Select random rows to duplicate
            dup_indices = np.random.choice(df.index, size=num_dups, replace=True)
            duplicates = df.loc[dup_indices].copy().reset_index(drop=True)
            
            # Some duplicates are exact, some have slight variations
            numeric_cols = [c for c in df.columns 
                          if c not in self.PROTECTED_COLUMNS 
                          and df[c].dtype in ['int64', 'float64']]
            
            for i in range(len(duplicates)):
                if random.random() > 0.5 and numeric_cols:
                    col = random.choice(numeric_cols)
                    val = duplicates.at[i, col]
                    if pd.notna(val):
                        duplicates.at[i, col] = val * random.uniform(0.99, 1.01)
            
            df = pd.concat([df, duplicates], ignore_index=True)
        
        return df
    
    def _inject_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject outliers in numeric columns."""
        outlier_rate = self.quality_rate * 0.2  # 20% of quality rate
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.PROTECTED_COLUMNS:
                continue
            
            # Skip columns with too few unique values
            if df[col].nunique() < 5:
                continue
            
            valid_data = df[col].dropna()
            if len(valid_data) < 10:
                continue
            
            mean = valid_data.mean()
            std = valid_data.std()
            
            if std == 0:
                continue
            
            mask = np.random.random(len(df)) < outlier_rate
            num_outliers = mask.sum()
            
            if num_outliers > 0:
                # Generate outliers (3-10 standard deviations away)
                directions = np.random.choice([-1, 1], size=num_outliers)
                magnitudes = np.random.uniform(3, 10, size=num_outliers)
                outlier_values = mean + directions * magnitudes * std
                
                # For certain columns, ensure positive values
                if col in ['price', 'cost', 'total_amount', 'quantity', 'lifetime_value']:
                    outlier_values = np.abs(outlier_values)
                
                df.loc[mask, col] = outlier_values
        
        return df
    
    def _inject_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject data inconsistencies."""
        # For transaction data: some negative quantities or prices
        if 'quantity' in df.columns:
            neg_rate = self.quality_rate * 0.05
            mask = np.random.random(len(df)) < neg_rate
            df.loc[mask, 'quantity'] = -df.loc[mask, 'quantity'].abs()
        
        if 'total_amount' in df.columns and 'quantity' in df.columns and 'unit_price' in df.columns:
            # Some totals don't match quantity * price
            inconsist_rate = self.quality_rate * 0.1
            mask = np.random.random(len(df)) < inconsist_rate
            multiplier = random.uniform(0.8, 1.2)
            df.loc[mask, 'total_amount'] = df.loc[mask, 'total_amount'] * multiplier
        
        return df
    
    def _inject_typos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject typos and variations."""
        typo_rate = self.quality_rate * 0.15
        
        for col in ['country', 'status', 'segment', 'category', 'payment_method']:
            if col not in df.columns:
                continue
            
            mask = np.random.random(len(df)) < typo_rate
            
            if col == 'country':
                for idx in df[mask].index:
                    original = df.at[idx, col]
                    if original in self.TYPO_MAP:
                        df.at[idx, col] = random.choice(self.TYPO_MAP[original])
            else:
                # Generic typos: swap characters, add/remove letters
                for idx in df[mask].index:
                    val = str(df.at[idx, col])
                    if len(val) > 3:
                        typo_type = random.choice(['swap', 'double', 'remove', 'case'])
                        if typo_type == 'swap' and len(val) > 2:
                            i = random.randint(0, len(val) - 2)
                            val = val[:i] + val[i+1] + val[i] + val[i+2:]
                        elif typo_type == 'double':
                            i = random.randint(0, len(val) - 1)
                            val = val[:i] + val[i] + val[i:]
                        elif typo_type == 'remove' and len(val) > 3:
                            i = random.randint(1, len(val) - 2)
                            val = val[:i] + val[i+1:]
                        elif typo_type == 'case':
                            val = val.upper() if random.random() > 0.5 else val.lower()
                        df.at[idx, col] = val
        
        return df
    
    def _inject_date_format_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject inconsistent date formats."""
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        format_rate = self.quality_rate * 0.2
        
        for col in date_cols:
            mask = np.random.random(len(df)) < format_rate
            
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or val is None:
                    continue
                
                try:
                    # Parse the date
                    if isinstance(val, str):
                        parsed = pd.to_datetime(val)
                    else:
                        parsed = val
                    
                    # Reformat to random format
                    new_format = random.choice(self.DATE_FORMATS)
                    df.at[idx, col] = parsed.strftime(new_format)
                except:
                    pass
        
        return df
    
    def _inject_case_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject inconsistent casing."""
        case_rate = self.quality_rate * 0.1
        
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            if col in self.PROTECTED_COLUMNS:
                continue
            if 'id' in col.lower() or 'date' in col.lower():
                continue
            
            mask = np.random.random(len(df)) < case_rate
            
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or not isinstance(val, str):
                    continue
                
                case_type = random.choice(['upper', 'lower', 'title', 'random'])
                if case_type == 'upper':
                    df.at[idx, col] = val.upper()
                elif case_type == 'lower':
                    df.at[idx, col] = val.lower()
                elif case_type == 'title':
                    df.at[idx, col] = val.title()
                else:
                    # Random case per character
                    df.at[idx, col] = ''.join(
                        c.upper() if random.random() > 0.5 else c.lower() 
                        for c in val
                    )
        
        return df
    
    def _inject_whitespace_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject whitespace issues (leading/trailing spaces, extra spaces)."""
        ws_rate = self.quality_rate * 0.1
        
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            if col in self.PROTECTED_COLUMNS:
                continue
            
            mask = np.random.random(len(df)) < ws_rate
            
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or not isinstance(val, str):
                    continue
                
                ws_type = random.choice(['leading', 'trailing', 'both', 'extra'])
                if ws_type == 'leading':
                    df.at[idx, col] = '   ' + val
                elif ws_type == 'trailing':
                    df.at[idx, col] = val + '   '
                elif ws_type == 'both':
                    df.at[idx, col] = '  ' + val + '  '
                else:
                    # Add extra spaces between words
                    words = val.split()
                    df.at[idx, col] = '   '.join(words)
        
        return df
