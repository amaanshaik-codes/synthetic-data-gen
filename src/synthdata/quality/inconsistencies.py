"""
Data inconsistency injector.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import InconsistenciesConfig


class InconsistencyInjector:
    """Inject data inconsistencies into a DataFrame."""
    
    def __init__(self, config: InconsistenciesConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Common typos mapping
        self.typo_map = {
            'a': ['s', 'q', 'z'],
            'b': ['v', 'n', 'g'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'e'],
            'e': ['w', 'r', 'd'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', 'k'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'k', 'j'],
            'n': ['b', 'm', 'h'],
            'o': ['i', 'p', 'l'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 't', 'f'],
            's': ['a', 'd', 'w'],
            't': ['r', 'y', 'g'],
            'u': ['y', 'i', 'j'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', 's'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', 'h'],
            'z': ['a', 'x'],
        }
    
    def inject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject inconsistencies into the DataFrame."""
        df = df.copy()
        
        if self.config.date_formats:
            df = self._inject_date_inconsistencies(df)
        
        if self.config.currency_formats:
            df = self._inject_currency_inconsistencies(df)
        
        if self.config.category_typos:
            df = self._inject_category_typos(df)
        
        if self.config.case_inconsistencies:
            df = self._inject_case_inconsistencies(df)
        
        if self.config.whitespace_issues:
            df = self._inject_whitespace_issues(df)
        
        if self.config.encoding_issues:
            df = self._inject_encoding_issues(df)
        
        return df
    
    def _inject_date_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject inconsistent date formats."""
        # Find datetime columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also find object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            if any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'updated']):
                date_cols.append(col)
        
        for col in date_cols:
            if col not in df.columns:
                continue
            
            # Convert to object to allow mixed formats
            if df[col].dtype == 'datetime64[ns]':
                # Select random rows to convert to different formats
                mask = np.random.random(len(df)) < self.config.typo_rate * 2
                
                # Convert the column to object type to allow mixed formats
                df[col] = df[col].astype(object)
                
                for idx in df[mask].index:
                    if pd.notna(df.loc[idx, col]):
                        date_val = pd.to_datetime(df.loc[idx, col])
                        format_choice = random.choice(self.config.date_format_variations)
                        try:
                            df.loc[idx, col] = date_val.strftime(format_choice)
                        except:
                            pass
        
        return df
    
    def _inject_currency_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject inconsistent currency formats."""
        # Find currency-related columns
        currency_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['amount', 'price', 'cost', 'revenue', 'value', 'total', 'spent']):
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    currency_cols.append(col)
        
        for col in currency_cols:
            # Select random rows to format differently
            mask = np.random.random(len(df)) < self.config.typo_rate
            
            # Convert to object to allow string formatting
            df[col] = df[col].astype(object)
            
            for idx in df[mask].index:
                if pd.notna(df.loc[idx, col]):
                    value = float(df.loc[idx, col])
                    format_choice = random.choice(self.config.currency_variations)
                    
                    if format_choice == "$1,234.56":
                        df.loc[idx, col] = f"${value:,.2f}"
                    elif format_choice == "1234.56":
                        df.loc[idx, col] = f"{value:.2f}"
                    elif format_choice == "$1234.56":
                        df.loc[idx, col] = f"${value:.2f}"
                    elif format_choice == "1,234.56 USD":
                        df.loc[idx, col] = f"{value:,.2f} USD"
                    elif format_choice == "USD 1234.56":
                        df.loc[idx, col] = f"USD {value:.2f}"
        
        return df
    
    def _inject_category_typos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject typos into categorical columns."""
        # Find categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude ID-like columns
        cat_cols = [c for c in cat_cols if not c.endswith('_id') and 'id' not in c.lower()]
        
        for col in cat_cols:
            mask = np.random.random(len(df)) < self.config.typo_rate
            
            for idx in df[mask].index:
                if pd.notna(df.loc[idx, col]) and isinstance(df.loc[idx, col], str):
                    df.loc[idx, col] = self._introduce_typo(df.loc[idx, col])
        
        return df
    
    def _introduce_typo(self, text: str) -> str:
        """Introduce a typo into text."""
        if len(text) < 3:
            return text
        
        typo_type = random.choice(["swap", "replace", "double", "skip"])
        pos = random.randint(1, len(text) - 2)
        
        if typo_type == "swap":
            # Swap two adjacent characters
            text = text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
        
        elif typo_type == "replace":
            # Replace with nearby key
            char = text[pos].lower()
            if char in self.typo_map:
                replacement = random.choice(self.typo_map[char])
                if text[pos].isupper():
                    replacement = replacement.upper()
                text = text[:pos] + replacement + text[pos + 1:]
        
        elif typo_type == "double":
            # Double a character
            text = text[:pos] + text[pos] + text[pos:]
        
        elif typo_type == "skip":
            # Skip a character
            if len(text) > 4:
                text = text[:pos] + text[pos + 1:]
        
        return text
    
    def _inject_case_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject case inconsistencies in categorical columns."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if not c.endswith('_id')]
        
        for col in cat_cols:
            mask = np.random.random(len(df)) < self.config.typo_rate
            
            for idx in df[mask].index:
                if pd.notna(df.loc[idx, col]) and isinstance(df.loc[idx, col], str):
                    case_type = random.choice(["lower", "upper", "title", "mixed"])
                    text = df.loc[idx, col]
                    
                    if case_type == "lower":
                        df.loc[idx, col] = text.lower()
                    elif case_type == "upper":
                        df.loc[idx, col] = text.upper()
                    elif case_type == "title":
                        df.loc[idx, col] = text.title()
                    elif case_type == "mixed":
                        df.loc[idx, col] = ''.join(
                            c.upper() if random.random() > 0.5 else c.lower()
                            for c in text
                        )
        
        return df
    
    def _inject_whitespace_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject whitespace issues (leading/trailing spaces, double spaces)."""
        str_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in str_cols:
            mask = np.random.random(len(df)) < self.config.typo_rate
            
            for idx in df[mask].index:
                if pd.notna(df.loc[idx, col]) and isinstance(df.loc[idx, col], str):
                    ws_type = random.choice(["leading", "trailing", "double", "tabs"])
                    text = df.loc[idx, col]
                    
                    # Skip empty strings
                    if not text:
                        continue
                    
                    if ws_type == "leading":
                        df.loc[idx, col] = "  " + text
                    elif ws_type == "trailing":
                        df.loc[idx, col] = text + "  "
                    elif ws_type == "double":
                        # Insert double space somewhere
                        if len(text) > 0:
                            pos = random.randint(0, len(text) - 1)
                            df.loc[idx, col] = text[:pos] + "  " + text[pos:]
                    elif ws_type == "tabs":
                        df.loc[idx, col] = text.replace(" ", "\t", 1)
        
        return df
    
    def _inject_encoding_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject encoding issues (special characters, unicode problems)."""
        str_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        problematic_chars = {
            'e': 'é',
            'a': 'à',
            'o': 'ö',
            'u': 'ü',
            'n': 'ñ',
            'c': 'ç',
        }
        
        for col in str_cols:
            mask = np.random.random(len(df)) < self.config.typo_rate * 0.5
            
            for idx in df[mask].index:
                if pd.notna(df.loc[idx, col]) and isinstance(df.loc[idx, col], str):
                    text = df.loc[idx, col]
                    
                    # Replace a random character with its unicode variant
                    for orig, replacement in problematic_chars.items():
                        if orig in text.lower():
                            pos = text.lower().find(orig)
                            if text[pos].isupper():
                                replacement = replacement.upper()
                            text = text[:pos] + replacement + text[pos + 1:]
                            break
                    
                    df.loc[idx, col] = text
        
        return df
