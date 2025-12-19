"""
Main synthetic data generator that orchestrates all components.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from synthdata.config import (
    SynthDataConfig,
    Difficulty,
    AnalyticsUseCase,
    Industry,
)
from synthdata.generators import (
    CustomerGenerator,
    TransactionGenerator,
    ProductGenerator,
    CampaignGenerator,
    SupportTicketGenerator,
    OperationalLogGenerator,
)
from synthdata.quality import DataQualityInjector, apply_difficulty_preset
from synthdata.output import OutputHandler
from synthdata.analytics import AnalyticsEnhancer


class SyntheticDataGenerator:
    """Main class for generating synthetic datasets."""
    
    def __init__(self, config: SynthDataConfig):
        self.config = config
        self.seed = config.initialize_seed()
        
        # Initialize random generators
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize generators
        self._init_generators()
        
        # Initialize quality injector
        self._init_quality_injector()
        
        # Initialize output handler
        self.output_handler = OutputHandler(config.output)
        
        # Initialize analytics enhancer
        self.analytics_enhancer = AnalyticsEnhancer(config.analytics, self.seed)
        
        # Storage for generated data
        self.tables: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {}
    
    def _init_generators(self):
        """Initialize all data generators."""
        self.customer_generator = CustomerGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
        )
        
        self.transaction_generator = TransactionGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
        )
        
        self.product_generator = ProductGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
        )
        
        self.campaign_generator = CampaignGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
        )
        
        self.support_ticket_generator = SupportTicketGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
        )
        
        self.operational_log_generator = OperationalLogGenerator(
            self.config.business_context,
            self.config.business_size,
            self.config.data_quality,
            self.seed,
            self.config.difficulty,
        )
    
    def _init_quality_injector(self):
        """Initialize the data quality injector."""
        # Apply difficulty preset to quality config
        quality_config = apply_difficulty_preset(
            self.config.data_quality,
            self.config.difficulty,
        )
        
        self.quality_injector = DataQualityInjector(
            quality_config,
            self.config.difficulty,
            self.seed,
        )
    
    def generate(
        self,
        tables: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for all configured tables.
        
        Args:
            tables: List of specific tables to generate (None = all)
            show_progress: Whether to show progress bars
        
        Returns:
            Dictionary of table names to DataFrames
        """
        # Default tables to generate
        default_tables = [
            "customers",
            "products",
            "transactions",
            "campaigns",
            "support_tickets",
        ]
        
        # Add operational logs for chaotic difficulty
        if self.config.difficulty == Difficulty.CHAOTIC:
            default_tables.append("operational_logs")
        
        tables_to_generate = tables or default_tables
        
        # Generate tables in order (respecting dependencies)
        generation_order = self._get_generation_order(tables_to_generate)
        
        if show_progress:
            generation_order = tqdm(generation_order, desc="Generating tables")
        
        for table_name in generation_order:
            self._generate_table(table_name)
        
        # Apply analytics enhancements
        self._apply_analytics_enhancements()
        
        # Apply data quality issues
        self._apply_quality_issues()
        
        # Update customer aggregates from transactions
        self._update_customer_aggregates()
        
        # Build metadata
        self._build_metadata()
        
        return self.tables
    
    def _get_generation_order(self, tables: List[str]) -> List[str]:
        """Get the correct order to generate tables (dependencies first)."""
        # Define dependencies
        dependencies = {
            "transactions": ["customers", "products"],
            "support_tickets": ["customers"],
            "operational_logs": ["customers"],
        }
        
        ordered = []
        remaining = set(tables)
        
        while remaining:
            for table in list(remaining):
                deps = dependencies.get(table, [])
                if all(d in ordered or d not in tables for d in deps):
                    ordered.append(table)
                    remaining.remove(table)
                    break
            else:
                # No progress made, just add remaining
                ordered.extend(remaining)
                break
        
        return ordered
    
    def _generate_table(self, table_name: str):
        """Generate a single table."""
        if table_name == "customers":
            self.tables["customers"] = self._generate_customers()
        
        elif table_name == "products":
            self.tables["products"] = self._generate_products()
        
        elif table_name == "transactions":
            self.tables["transactions"] = self._generate_transactions()
        
        elif table_name == "campaigns":
            self.tables["campaigns"] = self._generate_campaigns()
        
        elif table_name == "support_tickets":
            self.tables["support_tickets"] = self._generate_support_tickets()
        
        elif table_name == "operational_logs":
            self.tables["operational_logs"] = self._generate_operational_logs()
    
    def _generate_customers(self) -> pd.DataFrame:
        """Generate customer data."""
        # Determine churn rate based on use case
        churn_rate = 0.15
        if self.config.analytics.use_case == AnalyticsUseCase.CHURN_PREDICTION:
            churn_rate = self.config.analytics.class_imbalance_ratio or 0.15
        
        return self.customer_generator.generate(
            num_rows=self.config.business_size.num_customers,
            include_churn=True,
            churn_rate=churn_rate,
        )
    
    def _generate_products(self) -> pd.DataFrame:
        """Generate product data."""
        return self.product_generator.generate(
            num_rows=self.config.business_size.num_products,
        )
    
    def _generate_transactions(self) -> pd.DataFrame:
        """Generate transaction data."""
        # Get customer info for relationships
        customer_ids = None
        customer_segments = None
        customer_signup_dates = None
        product_ids = None
        
        if "customers" in self.tables:
            customers_df = self.tables["customers"]
            customer_ids = customers_df["customer_id"].tolist()
            customer_segments = dict(zip(
                customers_df["customer_id"],
                customers_df["segment"]
            ))
            customer_signup_dates = dict(zip(
                customers_df["customer_id"],
                pd.to_datetime(customers_df["signup_date"])
            ))
        
        if "products" in self.tables:
            product_ids = self.tables["products"]["product_id"].tolist()
        
        # Calculate number of transactions
        total_days = self.config.business_context.time_span_months * 30
        num_transactions = self.config.business_size.daily_transactions * total_days
        
        # Determine fraud rate for fintech
        include_fraud = (
            self.config.business_context.industry == Industry.FINTECH or
            self.config.analytics.use_case == AnalyticsUseCase.FRAUD_DETECTION
        )
        fraud_rate = self.config.analytics.class_imbalance_ratio or 0.02
        
        return self.transaction_generator.generate(
            num_rows=num_transactions,
            customer_ids=customer_ids,
            customer_segments=customer_segments,
            customer_signup_dates=customer_signup_dates,
            product_ids=product_ids,
            include_fraud_labels=include_fraud,
            fraud_rate=fraud_rate,
        )
    
    def _generate_campaigns(self) -> pd.DataFrame:
        """Generate campaign data."""
        return self.campaign_generator.generate()
    
    def _generate_support_tickets(self) -> pd.DataFrame:
        """Generate support ticket data."""
        customer_ids = None
        if "customers" in self.tables:
            customer_ids = self.tables["customers"]["customer_id"].tolist()
        
        return self.support_ticket_generator.generate(
            customer_ids=customer_ids,
        )
    
    def _generate_operational_logs(self) -> pd.DataFrame:
        """Generate operational log data."""
        user_ids = None
        if "customers" in self.tables:
            user_ids = self.tables["customers"]["customer_id"].tolist()
        
        # Limit log size for performance
        max_logs = min(100000, self.config.business_size.daily_transactions * 30 * 10)
        
        return self.operational_log_generator.generate(
            num_rows=max_logs,
            user_ids=user_ids,
        )
    
    def _apply_analytics_enhancements(self):
        """Apply analytics-specific enhancements."""
        # Add target variables and features based on use case
        self.tables = self.analytics_enhancer.enhance(self.tables)
    
    def _apply_quality_issues(self):
        """Apply data quality issues to all tables."""
        for table_name, df in self.tables.items():
            # Determine time column if present
            time_cols = [c for c in df.columns if any(
                word in c.lower() for word in ['date', 'time', 'created', 'timestamp']
            )]
            time_column = time_cols[0] if time_cols else None
            
            # Determine label column based on use case
            label_column = None
            if self.config.analytics.target_column and self.config.analytics.target_column in df.columns:
                label_column = self.config.analytics.target_column
            
            # Apply quality issues
            self.tables[table_name] = self.quality_injector.inject(
                df,
                table_name=table_name,
                primary_key=f"{table_name[:-1]}_id" if table_name.endswith("s") else f"{table_name}_id",
                label_column=label_column,
                time_column=time_column,
            )
    
    def _update_customer_aggregates(self):
        """Update customer aggregates from transaction data."""
        if "customers" not in self.tables or "transactions" not in self.tables:
            return
        
        customers = self.tables["customers"]
        transactions = self.tables["transactions"]
        
        # Calculate aggregates
        if "amount" in transactions.columns and "customer_id" in transactions.columns:
            # Handle potential type issues from quality injection
            try:
                txn_clean = transactions.copy()
                txn_clean["amount"] = pd.to_numeric(txn_clean["amount"], errors="coerce")
                
                agg = txn_clean.groupby("customer_id").agg({
                    "amount": ["sum", "mean", "count"],
                    "transaction_date": "max",
                }).reset_index()
                
                agg.columns = ["customer_id", "total_spent", "avg_order_value", "total_orders", "last_activity_date"]
                
                # Merge back to customers
                customers = customers.drop(
                    columns=["total_spent", "avg_order_value", "total_orders", "last_activity_date"],
                    errors="ignore"
                )
                customers = customers.merge(agg, on="customer_id", how="left")
                
                # Fill NAs for customers with no transactions
                customers["total_spent"] = customers["total_spent"].fillna(0)
                customers["avg_order_value"] = customers["avg_order_value"].fillna(0)
                customers["total_orders"] = customers["total_orders"].fillna(0)
                
                self.tables["customers"] = customers
            except Exception as e:
                # If aggregation fails due to data quality issues, that's expected
                pass
    
    def _build_metadata(self):
        """Build metadata about the generated dataset."""
        self.metadata = {
            "name": self.config.name,
            "description": self.config.description,
            "generated_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "tables": {},
            "relationships": self._get_relationships(),
            "data_dictionary": self._get_data_dictionary(),
            "quality_issues": self._get_quality_issues_summary(),
            "suggested_questions": self._get_suggested_questions(),
        }
        
        for table_name, df in self.tables.items():
            self.metadata["tables"][table_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isna().sum().to_dict(),
            }
    
    def _get_relationships(self) -> List[Dict[str, str]]:
        """Get table relationships."""
        relationships = []
        
        if "customers" in self.tables and "transactions" in self.tables:
            relationships.append({
                "parent": "customers",
                "child": "transactions",
                "type": "one-to-many",
                "key": "customer_id",
            })
        
        if "products" in self.tables and "transactions" in self.tables:
            if "product_id" in self.tables["transactions"].columns:
                relationships.append({
                    "parent": "products",
                    "child": "transactions",
                    "type": "one-to-many",
                    "key": "product_id",
                })
        
        if "customers" in self.tables and "support_tickets" in self.tables:
            relationships.append({
                "parent": "customers",
                "child": "support_tickets",
                "type": "one-to-many",
                "key": "customer_id",
            })
        
        return relationships
    
    def _get_data_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Get data dictionary with column descriptions."""
        dictionary = {}
        
        generators = {
            "customers": self.customer_generator,
            "products": self.product_generator,
            "transactions": self.transaction_generator,
            "campaigns": self.campaign_generator,
            "support_tickets": self.support_ticket_generator,
            "operational_logs": self.operational_log_generator,
        }
        
        for table_name in self.tables:
            if table_name in generators:
                dictionary[table_name] = generators[table_name].get_column_descriptions()
            else:
                dictionary[table_name] = {}
        
        return dictionary
    
    def _get_quality_issues_summary(self) -> Dict[str, Any]:
        """Get summary of injected quality issues."""
        return {
            "difficulty": self.config.difficulty.value,
            "missing_values_rate": self.config.data_quality.missing_values.global_rate,
            "duplicate_rate": self.config.data_quality.duplicates.rate,
            "outlier_rate": self.config.data_quality.outliers.rate,
            "inconsistencies_enabled": self.config.data_quality.inconsistencies.date_formats,
            "intentional_issues": [
                "Missing values distributed across columns (MCAR/MAR/MNAR patterns)",
                "Duplicate records with slight variations",
                "Inconsistent date formats",
                "Typos in categorical columns",
                "Outliers in numeric columns",
                "Data drift over time (if enabled)",
            ] if self.config.difficulty != Difficulty.EASY else [
                "Minimal missing values",
                "Rare duplicates",
                "Rare outliers",
            ],
        }
    
    def _get_suggested_questions(self) -> List[str]:
        """Get suggested analytics questions based on the dataset."""
        questions = []
        
        # General questions
        questions.extend([
            "What is the distribution of customers across segments?",
            "How has the number of transactions changed over time?",
            "What are the top-selling products by category?",
            "What is the average customer lifetime value by segment?",
        ])
        
        # Use-case specific questions
        if self.config.analytics.use_case == AnalyticsUseCase.CHURN_PREDICTION:
            questions.extend([
                "What factors are most predictive of customer churn?",
                "How does engagement differ between churned and retained customers?",
                "Can you build a model to predict which customers will churn?",
                "What is the optimal intervention point for at-risk customers?",
            ])
        
        elif self.config.analytics.use_case == AnalyticsUseCase.FRAUD_DETECTION:
            questions.extend([
                "What patterns distinguish fraudulent transactions?",
                "How can you handle the class imbalance in fraud detection?",
                "What features are most important for detecting fraud?",
                "What is the trade-off between precision and recall?",
            ])
        
        elif self.config.analytics.use_case == AnalyticsUseCase.TIME_SERIES:
            questions.extend([
                "What seasonal patterns exist in the data?",
                "Can you forecast next month's sales?",
                "How do you handle missing time periods?",
                "What is the trend in key metrics over time?",
            ])
        
        # Data quality questions
        questions.extend([
            "How much data is missing and in which columns?",
            "Are there any duplicate records? How do you identify them?",
            "What outliers exist and how should they be handled?",
            "Are there any data quality issues that need cleaning?",
        ])
        
        return questions
    
    def save(
        self,
        output_dir: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Save generated data to files.
        
        Args:
            output_dir: Output directory (uses config if not specified)
            include_metadata: Whether to save metadata files
        
        Returns:
            Path to the output directory
        """
        output_dir = output_dir or self.config.output.directory
        
        return self.output_handler.save(
            tables=self.tables,
            metadata=self.metadata,
            output_dir=output_dir,
            include_metadata=include_metadata,
        )
    
    def get_train_test_split(
        self,
        table_name: str = "customers",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/test split of a table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.tables[table_name]
        
        return self.analytics_enhancer.create_train_test_split(
            df,
            self.config.analytics.train_test_split,
            self.config.analytics.time_based_split,
            self.config.analytics.stratified_split,
            self.config.analytics.target_column,
        )
