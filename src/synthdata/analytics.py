"""
Analytics enhancement utilities for synthetic data.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthdata.config import AnalyticsConfig, AnalyticsUseCase


class AnalyticsEnhancer:
    """Enhance generated data with analytics-specific features."""
    
    def __init__(self, config: AnalyticsConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def enhance(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply analytics enhancements to tables."""
        use_case = self.config.use_case
        
        if use_case == AnalyticsUseCase.CHURN_PREDICTION:
            tables = self._enhance_for_churn(tables)
        
        elif use_case == AnalyticsUseCase.FRAUD_DETECTION:
            tables = self._enhance_for_fraud(tables)
        
        elif use_case == AnalyticsUseCase.TIME_SERIES:
            tables = self._enhance_for_timeseries(tables)
        
        elif use_case == AnalyticsUseCase.RECOMMENDATION:
            tables = self._enhance_for_recommendation(tables)
        
        elif use_case == AnalyticsUseCase.SEGMENTATION:
            tables = self._enhance_for_segmentation(tables)
        
        elif use_case == AnalyticsUseCase.COHORT_ANALYSIS:
            tables = self._enhance_for_cohort(tables)
        
        # Add feature leakage if configured
        if self.config.feature_leakage:
            tables = self._add_feature_leakage(tables)
        
        return tables
    
    def _enhance_for_churn(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for churn prediction use case."""
        if "customers" not in tables:
            return tables
        
        customers = tables["customers"].copy()
        
        # Ensure churn-related columns exist
        if "is_active" not in customers.columns:
            customers["is_active"] = True
        
        # Create churn target if not exists
        if "churned" not in customers.columns:
            customers["churned"] = ~customers["is_active"]
        
        # Add churn-predictive features
        customers["engagement_score"] = self._generate_engagement_score(
            customers,
            target_col="churned",
        )
        
        customers["recency_days"] = self._generate_correlated_feature(
            customers,
            target_col="churned",
            correlation=0.4,
            feature_name="recency",
        )
        
        customers["support_tickets_30d"] = self._generate_correlated_feature(
            customers,
            target_col="churned",
            correlation=0.3,
            feature_name="tickets",
        )
        
        # Set target column
        self.config.target_column = "churned"
        
        tables["customers"] = customers
        return tables
    
    def _enhance_for_fraud(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for fraud detection use case."""
        if "transactions" not in tables:
            return tables
        
        transactions = tables["transactions"].copy()
        
        # Add fraud-predictive features if not present
        if "is_fraud" not in transactions.columns:
            # Create fraud labels with class imbalance
            fraud_rate = self.config.class_imbalance_ratio or 0.02
            transactions["is_fraud"] = np.random.random(len(transactions)) < fraud_rate
        
        # Add fraud-indicative features
        if "amount_zscore" not in transactions.columns:
            amount_col = transactions["amount"]
            if amount_col.dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean_amt = amount_col.mean()
                std_amt = amount_col.std()
                transactions["amount_zscore"] = (amount_col - mean_amt) / (std_amt + 1e-10)
        
        if "hour_of_day" not in transactions.columns:
            if "transaction_date" in transactions.columns:
                try:
                    dates = pd.to_datetime(transactions["transaction_date"])
                    transactions["hour_of_day"] = dates.dt.hour
                    transactions["is_weekend"] = dates.dt.dayofweek >= 5
                except:
                    pass
        
        # Add velocity features
        transactions["transaction_velocity"] = self._calculate_velocity(transactions)
        
        # Set target column
        self.config.target_column = "is_fraud"
        
        tables["transactions"] = transactions
        return tables
    
    def _enhance_for_timeseries(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for time series forecasting."""
        if "transactions" not in tables:
            return tables
        
        transactions = tables["transactions"].copy()
        
        # Add time-based features
        if "transaction_date" in transactions.columns:
            try:
                dates = pd.to_datetime(transactions["transaction_date"])
                transactions["year"] = dates.dt.year
                transactions["month"] = dates.dt.month
                transactions["week"] = dates.dt.isocalendar().week
                transactions["day_of_week"] = dates.dt.dayofweek
                transactions["day_of_month"] = dates.dt.day
                transactions["is_month_end"] = dates.dt.is_month_end
                transactions["quarter"] = dates.dt.quarter
            except:
                pass
        
        # Create daily aggregates for forecasting
        if "transaction_date" in transactions.columns and "amount" in transactions.columns:
            try:
                daily = transactions.groupby(
                    pd.to_datetime(transactions["transaction_date"]).dt.date
                ).agg({
                    "amount": ["sum", "count", "mean"],
                    "transaction_id": "count",
                }).reset_index()
                daily.columns = ["date", "daily_revenue", "num_transactions", "avg_transaction", "total_count"]
                tables["daily_aggregates"] = daily
            except:
                pass
        
        tables["transactions"] = transactions
        return tables
    
    def _enhance_for_recommendation(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for recommendation systems."""
        if "transactions" not in tables or "customers" not in tables or "products" not in tables:
            return tables
        
        transactions = tables["transactions"]
        
        # Create user-item interaction matrix
        if "customer_id" in transactions.columns and "product_id" in transactions.columns:
            try:
                interactions = transactions.groupby(
                    ["customer_id", "product_id"]
                ).agg({
                    "transaction_id": "count",
                    "amount": "sum",
                }).reset_index()
                interactions.columns = ["customer_id", "product_id", "interaction_count", "total_spent"]
                
                # Add implicit rating
                max_interactions = interactions["interaction_count"].max()
                interactions["implicit_rating"] = (
                    interactions["interaction_count"] / max_interactions * 5
                ).clip(1, 5).round(1)
                
                tables["user_item_interactions"] = interactions
            except:
                pass
        
        return tables
    
    def _enhance_for_segmentation(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for customer segmentation."""
        if "customers" not in tables:
            return tables
        
        customers = tables["customers"].copy()
        
        # Add RFM features if transaction data available
        if "transactions" in tables:
            transactions = tables["transactions"]
            
            try:
                # Calculate RFM
                max_date = pd.to_datetime(transactions["transaction_date"]).max()
                
                rfm = transactions.groupby("customer_id").agg({
                    "transaction_date": lambda x: (max_date - pd.to_datetime(x).max()).days,
                    "transaction_id": "count",
                    "amount": "sum",
                }).reset_index()
                rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
                
                # Merge with customers
                customers = customers.merge(rfm, on="customer_id", how="left")
                customers["recency"] = customers["recency"].fillna(999)
                customers["frequency"] = customers["frequency"].fillna(0)
                customers["monetary"] = customers["monetary"].fillna(0)
                
                tables["customers"] = customers
            except:
                pass
        
        return tables
    
    def _enhance_for_cohort(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhance data for cohort analysis."""
        if "customers" not in tables:
            return tables
        
        customers = tables["customers"].copy()
        
        # Add cohort assignment
        if "signup_date" in customers.columns:
            try:
                signup_dates = pd.to_datetime(customers["signup_date"])
                customers["cohort_month"] = signup_dates.dt.to_period("M").astype(str)
                customers["cohort_quarter"] = signup_dates.dt.to_period("Q").astype(str)
                customers["signup_year"] = signup_dates.dt.year
                
                tables["customers"] = customers
            except:
                pass
        
        return tables
    
    def _generate_engagement_score(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.Series:
        """Generate engagement score correlated with target."""
        n = len(df)
        
        # Base score
        scores = np.random.normal(50, 20, n)
        
        # Adjust based on target
        if target_col in df.columns:
            target = df[target_col].astype(bool)
            # Churned customers have lower engagement
            scores[target] = scores[target] - 20 * self.config.signal_to_noise_ratio
        
        return pd.Series(np.clip(scores, 0, 100), index=df.index).round(2)
    
    def _generate_correlated_feature(
        self,
        df: pd.DataFrame,
        target_col: str,
        correlation: float,
        feature_name: str,
    ) -> pd.Series:
        """Generate a feature correlated with target."""
        n = len(df)
        
        # Base feature
        feature = np.random.normal(0, 1, n)
        
        if target_col in df.columns:
            target = df[target_col].astype(float)
            # Mix target with noise based on correlation
            target_normalized = (target - target.mean()) / (target.std() + 1e-10)
            feature = correlation * target_normalized + (1 - correlation) * feature
        
        # Scale to reasonable range
        if "recency" in feature_name:
            feature = np.abs(feature) * 30 + 1
        elif "tickets" in feature_name:
            feature = np.abs(feature) * 2
            feature = np.round(feature).astype(int)
        
        return pd.Series(feature, index=df.index)
    
    def _calculate_velocity(self, transactions: pd.DataFrame) -> pd.Series:
        """Calculate transaction velocity feature."""
        velocity = pd.Series(np.random.exponential(2, len(transactions)))
        
        if "is_fraud" in transactions.columns:
            # Fraudulent transactions have higher velocity
            fraud_mask = transactions["is_fraud"] == True
            velocity[fraud_mask] = velocity[fraud_mask] * 3
        
        return velocity.round(2)
    
    def _add_feature_leakage(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add intentional feature leakage for educational purposes."""
        target_col = self.config.target_column
        
        if not target_col:
            return tables
        
        for table_name, df in tables.items():
            if target_col in df.columns:
                # Add a highly correlated feature (obvious leakage)
                df["_leaky_feature"] = df[target_col].astype(float) + np.random.normal(0, 0.1, len(df))
                
                # Add a subtle leakage feature
                df["_subtle_leaky"] = df[target_col].astype(float) * 0.8 + np.random.normal(0, 0.5, len(df))
                
                tables[table_name] = df
        
        return tables
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        time_based: bool = False,
        stratified: bool = True,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split of the data."""
        n = len(df)
        train_size = int(n * train_ratio)
        
        if time_based:
            # Time-based split - find a date column
            date_cols = [c for c in df.columns if any(
                word in c.lower() for word in ['date', 'time', 'created']
            )]
            
            if date_cols:
                df_sorted = df.sort_values(date_cols[0])
                train = df_sorted.iloc[:train_size]
                test = df_sorted.iloc[train_size:]
                return train, test
        
        if stratified and target_column and target_column in df.columns:
            # Stratified split
            from sklearn.model_selection import train_test_split
            try:
                train, test = train_test_split(
                    df,
                    train_size=train_ratio,
                    stratify=df[target_column],
                    random_state=self.seed,
                )
                return train, test
            except:
                pass
        
        # Random split
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        return df.iloc[train_indices], df.iloc[test_indices]
