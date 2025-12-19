"""
Base generator class with common functionality.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    DataQualityConfig,
    Industry,
    Geography,
)


class BaseGenerator(ABC):
    """Base class for all data generators."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        self.business_context = business_context
        self.business_size = business_size
        self.data_quality = data_quality
        self.seed = seed
        
        # Initialize random generators
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize Faker with appropriate locales
        self.faker = self._initialize_faker()
        
        # Calculate date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30 * business_context.time_span_months)
        
        # Industry-specific configurations
        self.industry_config = self._get_industry_config()
    
    def _initialize_faker(self) -> Faker:
        """Initialize Faker with appropriate locales based on geography."""
        locale_map = {
            "US": "en_US",
            "UK": "en_GB",
            "DE": "de_DE",
            "FR": "fr_FR",
            "ES": "es_ES",
            "IT": "it_IT",
            "JP": "ja_JP",
            "CN": "zh_CN",
            "IN": "en_IN",
            "BR": "pt_BR",
            "MX": "es_MX",
            "CA": "en_CA",
            "AU": "en_AU",
        }
        
        locales = []
        for country in self.business_context.countries:
            if country in locale_map:
                locales.append(locale_map[country])
            else:
                locales.append("en_US")
        
        if not locales:
            locales = ["en_US"]
        
        faker = Faker(locales)
        if self.seed is not None:
            Faker.seed(self.seed)
        
        return faker
    
    def _get_industry_config(self) -> Dict[str, Any]:
        """Get industry-specific configuration values."""
        configs = {
            Industry.RETAIL: {
                "product_categories": [
                    "Electronics", "Clothing", "Home & Garden", "Sports", 
                    "Beauty", "Toys", "Books", "Food & Beverages"
                ],
                "avg_order_value": (25, 150),
                "payment_methods": ["Credit Card", "Debit Card", "Cash", "Gift Card"],
                "channels": ["In-Store", "Online", "Mobile App"],
            },
            Industry.ECOMMERCE: {
                "product_categories": [
                    "Electronics", "Fashion", "Home & Living", "Health & Beauty",
                    "Sports & Outdoors", "Books & Media", "Toys & Games", "Grocery"
                ],
                "avg_order_value": (30, 200),
                "payment_methods": ["Credit Card", "PayPal", "Apple Pay", "Google Pay", "Buy Now Pay Later"],
                "channels": ["Website", "Mobile App", "Marketplace"],
            },
            Industry.FINTECH: {
                "transaction_types": [
                    "Transfer", "Payment", "Withdrawal", "Deposit",
                    "Investment", "Loan Payment", "Fee", "Refund"
                ],
                "avg_transaction_value": (50, 5000),
                "payment_methods": ["Bank Transfer", "Card", "Wallet", "Crypto"],
                "risk_levels": ["Low", "Medium", "High"],
            },
            Industry.HEALTHCARE: {
                "service_categories": [
                    "Consultation", "Diagnostic", "Treatment", "Surgery",
                    "Pharmacy", "Lab Test", "Imaging", "Therapy"
                ],
                "departments": [
                    "General Medicine", "Cardiology", "Orthopedics", "Pediatrics",
                    "Neurology", "Dermatology", "Oncology", "Emergency"
                ],
                "insurance_types": ["Private", "Medicare", "Medicaid", "Self-Pay"],
            },
            Industry.SAAS: {
                "plan_types": ["Free", "Starter", "Professional", "Enterprise"],
                "features": [
                    "Basic Analytics", "Advanced Analytics", "API Access",
                    "Custom Integrations", "Priority Support", "SSO"
                ],
                "usage_metrics": ["API Calls", "Storage", "Users", "Projects"],
                "billing_cycles": ["Monthly", "Annual"],
            },
            Industry.MANUFACTURING: {
                "product_lines": [
                    "Raw Materials", "Components", "Sub-assemblies",
                    "Finished Goods", "Spare Parts"
                ],
                "quality_grades": ["A", "B", "C", "Reject"],
                "production_stages": ["Procurement", "Fabrication", "Assembly", "QC", "Packaging"],
            },
            Industry.MARKETING: {
                "campaign_types": [
                    "Email", "Social Media", "PPC", "Display",
                    "Content", "Influencer", "SEO", "Affiliate"
                ],
                "channels": [
                    "Google", "Facebook", "Instagram", "LinkedIn",
                    "Twitter", "Email", "YouTube", "TikTok"
                ],
                "metrics": ["Impressions", "Clicks", "Conversions", "Revenue"],
            },
            Industry.LOGISTICS: {
                "shipment_types": ["Standard", "Express", "Overnight", "Freight"],
                "carriers": ["FedEx", "UPS", "DHL", "USPS", "Internal"],
                "statuses": [
                    "Pending", "Picked Up", "In Transit", "Out for Delivery",
                    "Delivered", "Failed", "Returned"
                ],
            },
        }
        
        return configs.get(self.business_context.industry, configs[Industry.ECOMMERCE])
    
    def random_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> datetime:
        """Generate a random date within the time range."""
        start = start or self.start_date
        end = end or self.end_date
        
        time_diff = end - start
        random_days = random.random() * time_diff.days
        return start + timedelta(days=random_days)
    
    def random_dates_sorted(
        self,
        n: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[datetime]:
        """Generate n sorted random dates."""
        dates = [self.random_date(start, end) for _ in range(n)]
        return sorted(dates)
    
    def weighted_choice(
        self,
        choices: List[Any],
        weights: Optional[List[float]] = None,
    ) -> Any:
        """Make a weighted random choice."""
        if weights is None:
            return random.choice(choices)
        return random.choices(choices, weights=weights, k=1)[0]
    
    def generate_id(self, prefix: str = "", length: int = 8) -> str:
        """Generate a unique ID with optional prefix."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        id_part = "".join(random.choices(chars, k=length))
        return f"{prefix}{id_part}" if prefix else id_part
    
    def generate_skewed_value(
        self,
        min_val: float,
        max_val: float,
        skew: float = 1.0,
    ) -> float:
        """Generate a value with configurable skew (higher skew = more low values)."""
        u = random.random()
        skewed = u ** skew
        return min_val + skewed * (max_val - min_val)
    
    def generate_normal_value(
        self,
        mean: float,
        std: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        """Generate a normally distributed value with optional bounds."""
        value = np.random.normal(mean, std)
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value
    
    def generate_seasonality_factor(self, date: datetime) -> float:
        """Generate a seasonality factor based on date."""
        # Monthly seasonality
        month_factors = {
            1: 0.8, 2: 0.75, 3: 0.9, 4: 0.95,
            5: 1.0, 6: 0.95, 7: 0.9, 8: 0.85,
            9: 1.0, 10: 1.1, 11: 1.3, 12: 1.4,  # Holiday bump
        }
        
        # Day of week seasonality
        dow_factors = {
            0: 1.1, 1: 1.0, 2: 1.0, 3: 1.0,
            4: 1.15, 5: 1.3, 6: 1.2,  # Weekend bump
        }
        
        month_factor = month_factors.get(date.month, 1.0)
        dow_factor = dow_factors.get(date.weekday(), 1.0)
        
        return month_factor * dow_factor
    
    def get_country_for_record(self) -> str:
        """Get a country for a record based on geography settings."""
        if self.business_context.geography == Geography.SINGLE_COUNTRY:
            return self.business_context.countries[0]
        else:
            return random.choice(self.business_context.countries)
    
    @abstractmethod
    def generate(self, num_rows: int, **kwargs) -> pd.DataFrame:
        """Generate the data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for this generator's output."""
        pass
    
    @abstractmethod
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        pass
