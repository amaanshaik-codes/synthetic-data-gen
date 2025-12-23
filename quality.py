import random
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd


COUNTRY_VARIATIONS = {
    "USA": ["USA", "US", "United States", "U.S.A.", "U.S.", "America", "usa"],
    "UK": ["UK", "United Kingdom", "Great Britain", "GB", "England"],
    "Canada": ["Canada", "CA", "CAN", "canada"],
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y"]


class QualityInjector:
    PROTECTED_PATTERNS = ["_key", "_id", "date_key"]

    def __init__(self, quality_rate: float = 0.05, seed: int = 42):
        self.quality_rate = min(max(quality_rate, 0), 0.5)
        random.seed(seed)
        np.random.seed(seed)

    def _is_protected(self, col: str) -> bool:
        return any(p in col.lower() for p in self.PROTECTED_PATTERNS)

    def inject_issues(self, df: pd.DataFrame, is_dimension: bool = True) -> pd.DataFrame:
        if df.empty or self.quality_rate == 0:
            return df
        df = df.copy()
        
        if is_dimension:
            df = self._inject_case_variations(df)
            df = self._inject_typos(df)
            df = self._inject_whitespace(df)
            df = self._inject_country_variations(df)
        else:
            df = self._inject_duplicates(df)
            df = self._inject_value_errors(df)
        
        df = self._inject_nulls(df, is_dimension)
        return df

    def _inject_case_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        rate = self.quality_rate * 0.3
        text_cols = [c for c in df.select_dtypes(include=["object"]).columns if not self._is_protected(c)]
        
        for col in text_cols:
            mask = np.random.random(len(df)) < rate
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or not isinstance(val, str):
                    continue
                case_type = random.choice(["upper", "lower", "title"])
                if case_type == "upper":
                    df.at[idx, col] = val.upper()
                elif case_type == "lower":
                    df.at[idx, col] = val.lower()
                else:
                    df.at[idx, col] = val.title()
        return df

    def _inject_typos(self, df: pd.DataFrame) -> pd.DataFrame:
        rate = self.quality_rate * 0.1
        name_cols = [c for c in df.columns if any(n in c.lower() for n in ["name", "city"]) and not self._is_protected(c)]
        
        for col in name_cols:
            mask = np.random.random(len(df)) < rate
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or not isinstance(val, str) or len(val) < 3:
                    continue
                pos = random.randint(1, len(val) - 2)
                typo = random.choice(["swap", "delete", "duplicate"])
                if typo == "swap" and pos < len(val) - 1:
                    val = val[:pos] + val[pos + 1] + val[pos] + val[pos + 2:]
                elif typo == "delete":
                    val = val[:pos] + val[pos + 1:]
                else:
                    val = val[:pos] + val[pos] + val[pos:]
                df.at[idx, col] = val
        return df

    def _inject_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        rate = self.quality_rate * 0.15
        text_cols = [c for c in df.select_dtypes(include=["object"]).columns if not self._is_protected(c)]
        
        for col in text_cols:
            mask = np.random.random(len(df)) < rate
            for idx in df[mask].index:
                val = df.at[idx, col]
                if pd.isna(val) or not isinstance(val, str):
                    continue
                ws = random.choice(["lead", "trail", "double"])
                if ws == "lead":
                    df.at[idx, col] = "  " + val
                elif ws == "trail":
                    df.at[idx, col] = val + "  "
                else:
                    df.at[idx, col] = val.replace(" ", "  ")
        return df

    def _inject_country_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        if "country" not in df.columns:
            return df
        rate = self.quality_rate * 0.2
        mask = np.random.random(len(df)) < rate
        for idx in df[mask].index:
            val = df.at[idx, "country"]
            if pd.isna(val):
                continue
            for standard, variations in COUNTRY_VARIATIONS.items():
                if val in variations or val == standard:
                    df.at[idx, "country"] = random.choice(variations)
                    break
        return df

    def _inject_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return df
        rate = self.quality_rate * 0.02
        n_dups = int(len(df) * rate)
        if n_dups > 0:
            dup_idx = np.random.choice(df.index, size=n_dups, replace=True)
            dups = df.loc[dup_idx].copy().reset_index(drop=True)
            df = pd.concat([df, dups], ignore_index=True)
        return df

    def _inject_value_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        rate = self.quality_rate * 0.03
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not self._is_protected(c)]
        
        for col in numeric_cols:
            if "quantity" in col.lower() or "qty" in col.lower():
                mask = np.random.random(len(df)) < rate
                df.loc[mask, col] = -df.loc[mask, col].abs()
            elif "amount" in col.lower() or "total" in col.lower() or "revenue" in col.lower():
                mask = np.random.random(len(df)) < rate
                if mask.sum() > 0:
                    df.loc[mask, col] = df.loc[mask, col] * random.uniform(0.5, 1.5)
        return df

    def _inject_nulls(self, df: pd.DataFrame, is_dimension: bool) -> pd.DataFrame:
        rate = self.quality_rate * (0.3 if is_dimension else 0.5)
        
        for col in df.columns:
            if self._is_protected(col):
                continue
            col_rate = rate
            if any(n in col.lower() for n in ["phone", "fax", "middle"]):
                col_rate *= 2
            elif any(n in col.lower() for n in ["name", "email", "id"]):
                col_rate *= 0.3
            col_rate = min(col_rate, 0.2)
            
            mask = np.random.random(len(df)) < col_rate
            if mask.sum() > 0:
                if df[col].dtype == "object":
                    nulls = [np.nan, "", "N/A", "null", None]
                    weights = [0.5, 0.2, 0.15, 0.1, 0.05]
                    df.loc[mask, col] = random.choices(nulls, weights=weights, k=mask.sum())
                else:
                    df.loc[mask, col] = np.nan
        return df


class LogicalIssueInjector:
    def __init__(self, rate: float = 0.02, seed: int = 42):
        self.rate = rate
        random.seed(seed)

    def inject_time_travelers(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
        if df.empty or start_col not in df.columns or end_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < self.rate
        
        for idx in df[mask].index:
            start = df.at[idx, start_col]
            if pd.isna(start):
                continue
            try:
                start_dt = pd.to_datetime(start)
                wrong_end = start_dt - timedelta(days=random.randint(1, 10))
                df.at[idx, end_col] = wrong_end.strftime("%Y-%m-%d")
            except:
                pass
        return df

    def inject_orphans(self, df: pd.DataFrame, fk_col: str, max_valid: int) -> pd.DataFrame:
        if df.empty or fk_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < self.rate
        df.loc[mask, fk_col] = max_valid + random.randint(1000, 9999)
        return df
