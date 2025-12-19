"""
Customer data generator.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    DataQualityConfig,
    Industry,
    BusinessModel,
    BusinessScale,
)
from synthdata.generators.base import BaseGenerator


class CustomerGenerator(BaseGenerator):
    """Generator for customer data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        # Customer segments based on industry
        self.segments = self._get_customer_segments()
        self.acquisition_channels = self._get_acquisition_channels()
    
    def _get_customer_segments(self) -> List[str]:
        """Get customer segments based on industry and business model."""
        if self.business_context.business_model == BusinessModel.B2B:
            return ["Enterprise", "Mid-Market", "SMB", "Startup"]
        
        industry_segments = {
            Industry.RETAIL: ["Budget", "Value", "Premium", "Luxury"],
            Industry.ECOMMERCE: ["Casual", "Regular", "Power User", "VIP"],
            Industry.FINTECH: ["Basic", "Standard", "Premium", "Private"],
            Industry.HEALTHCARE: ["Individual", "Family", "Senior", "Corporate"],
            Industry.SAAS: ["Free", "Starter", "Professional", "Enterprise"],
            Industry.MANUFACTURING: ["Distributor", "Retailer", "Direct", "OEM"],
            Industry.MARKETING: ["Small Business", "Agency", "Enterprise", "Consultant"],
            Industry.LOGISTICS: ["Occasional", "Regular", "High Volume", "Strategic"],
        }
        
        return industry_segments.get(
            self.business_context.industry,
            ["Standard", "Premium", "VIP"]
        )
    
    def _get_acquisition_channels(self) -> List[str]:
        """Get acquisition channels based on business model."""
        if self.business_context.business_model == BusinessModel.B2B:
            return ["Sales Team", "Referral", "Conference", "LinkedIn", "Cold Outreach", "Partner"]
        
        return [
            "Organic Search", "Paid Search", "Social Media", "Email",
            "Referral", "Direct", "Affiliate", "Content Marketing"
        ]
    
    def _generate_customer_value_score(self, segment: str) -> float:
        """Generate a customer lifetime value score."""
        segment_multipliers = {
            self.segments[0]: 0.5,   # Lowest tier
            self.segments[-1]: 2.0,  # Highest tier
        }
        
        base = random.gauss(50, 20)
        multiplier = segment_multipliers.get(segment, 1.0)
        
        return max(0, min(100, base * multiplier))
    
    def _generate_signup_date(self, index: int, total: int) -> datetime:
        """Generate signup dates with realistic growth patterns."""
        # Simulate growth - more recent customers
        # Use exponential distribution to favor recent dates
        progress = index / total
        
        # Adjust based on business scale
        if self.business_size.scale == BusinessScale.STARTUP:
            # Faster recent growth
            skew = 2.0
        elif self.business_size.scale == BusinessScale.ENTERPRISE:
            # More even distribution
            skew = 0.5
        else:
            skew = 1.0
        
        # Apply skew
        adjusted_progress = progress ** (1 / skew)
        
        days_range = (self.end_date - self.start_date).days
        days_offset = int(adjusted_progress * days_range)
        
        return self.start_date + timedelta(days=days_offset)
    
    def _generate_churn_date(
        self,
        signup_date: datetime,
        is_churned: bool,
    ) -> Optional[datetime]:
        """Generate churn date for churned customers."""
        if not is_churned:
            return None
        
        # Churn happens sometime after signup
        min_days = 30  # At least 30 days before churn
        max_days = (self.end_date - signup_date).days
        
        if max_days <= min_days:
            return None
        
        churn_days = random.randint(min_days, max_days)
        return signup_date + timedelta(days=churn_days)
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        include_churn: bool = True,
        churn_rate: float = 0.15,
        **kwargs
    ) -> pd.DataFrame:
        """Generate customer data."""
        if num_rows is None:
            num_rows = self.business_size.num_customers
        
        customers = []
        
        for i in range(num_rows):
            customer_id = self.generate_id("CUS", 8)
            country = self.get_country_for_record()
            
            # Basic info - faker is already initialized with appropriate locales
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@{self.faker.free_email_domain()}"
            
            # Segment assignment with realistic distribution
            segment_weights = [0.4, 0.35, 0.2, 0.05][:len(self.segments)]
            segment = self.weighted_choice(self.segments, segment_weights)
            
            # Dates
            signup_date = self._generate_signup_date(i, num_rows)
            
            # Churn
            is_churned = random.random() < churn_rate if include_churn else False
            churn_date = self._generate_churn_date(signup_date, is_churned)
            
            # Acquisition
            channel_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02][:len(self.acquisition_channels)]
            acquisition_channel = self.weighted_choice(self.acquisition_channels, channel_weights)
            
            # Engagement metrics
            days_active = (self.end_date - signup_date).days
            if is_churned and churn_date:
                days_active = (churn_date - signup_date).days
            
            # B2B specific fields
            if self.business_context.business_model == BusinessModel.B2B:
                company_name = self.faker.company()
                company_size = random.choice(["1-10", "11-50", "51-200", "201-500", "500+"])
                industry = random.choice([
                    "Technology", "Finance", "Healthcare", "Retail",
                    "Manufacturing", "Education", "Government", "Other"
                ])
            else:
                company_name = None
                company_size = None
                industry = None
            
            customer = {
                "customer_id": customer_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": self.faker.phone_number() if random.random() > 0.2 else None,
                "country": country,
                "city": self.faker.city(),
                "state": self.faker.state() if country == "US" else None,
                "postal_code": self.faker.postcode(),
                "segment": segment,
                "acquisition_channel": acquisition_channel,
                "signup_date": signup_date,
                "is_active": not is_churned,
                "churn_date": churn_date,
                "lifetime_value_score": round(self._generate_customer_value_score(segment), 2),
                "total_orders": max(0, int(self.generate_normal_value(
                    mean=10 * (self.segments.index(segment) + 1),
                    std=5,
                    min_val=0
                ))),
                "total_spent": 0.0,  # Will be calculated from transactions
                "avg_order_value": 0.0,  # Will be calculated from transactions
                "days_since_signup": days_active,
                "last_activity_date": None,  # Will be set from transactions
                "email_opt_in": random.random() > 0.3,
                "sms_opt_in": random.random() > 0.7,
            }
            
            # Add B2B fields if applicable
            if self.business_context.business_model == BusinessModel.B2B:
                customer["company_name"] = company_name
                customer["company_size"] = company_size
                customer["company_industry"] = industry
                customer["account_manager"] = self.faker.name() if random.random() > 0.5 else None
                customer["contract_value"] = round(
                    self.generate_skewed_value(1000, 500000, skew=2.0), 2
                )
            
            # Add SaaS-specific fields
            if self.business_context.industry == Industry.SAAS:
                plans = ["Free", "Starter", "Professional", "Enterprise"]
                plan_weights = [0.5, 0.3, 0.15, 0.05]
                customer["subscription_plan"] = self.weighted_choice(plans, plan_weights)
                customer["monthly_recurring_revenue"] = self._calculate_mrr(customer["subscription_plan"])
                customer["billing_cycle"] = random.choice(["monthly", "annual"])
            
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        
        # Sort by signup date
        df = df.sort_values("signup_date").reset_index(drop=True)
        
        return df
    
    def _calculate_mrr(self, plan: str) -> float:
        """Calculate monthly recurring revenue based on plan."""
        mrr_by_plan = {
            "Free": 0.0,
            "Starter": random.uniform(9, 29),
            "Professional": random.uniform(49, 99),
            "Enterprise": random.uniform(199, 999),
        }
        return round(mrr_by_plan.get(plan, 0.0), 2)
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for customer data."""
        schema = {
            "customer_id": "string",
            "first_name": "string",
            "last_name": "string",
            "email": "string",
            "phone": "string",
            "country": "string",
            "city": "string",
            "state": "string",
            "postal_code": "string",
            "segment": "category",
            "acquisition_channel": "category",
            "signup_date": "datetime",
            "is_active": "boolean",
            "churn_date": "datetime",
            "lifetime_value_score": "float",
            "total_orders": "integer",
            "total_spent": "float",
            "avg_order_value": "float",
            "days_since_signup": "integer",
            "last_activity_date": "datetime",
            "email_opt_in": "boolean",
            "sms_opt_in": "boolean",
        }
        
        if self.business_context.business_model == BusinessModel.B2B:
            schema.update({
                "company_name": "string",
                "company_size": "category",
                "company_industry": "category",
                "account_manager": "string",
                "contract_value": "float",
            })
        
        if self.business_context.industry == Industry.SAAS:
            schema.update({
                "subscription_plan": "category",
                "monthly_recurring_revenue": "float",
                "billing_cycle": "category",
            })
        
        return schema
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        descriptions = {
            "customer_id": "Unique identifier for the customer",
            "first_name": "Customer's first name",
            "last_name": "Customer's last name",
            "email": "Customer's email address",
            "phone": "Customer's phone number",
            "country": "Customer's country of residence",
            "city": "Customer's city",
            "state": "Customer's state/province",
            "postal_code": "Customer's postal/zip code",
            "segment": "Customer segment classification",
            "acquisition_channel": "Channel through which customer was acquired",
            "signup_date": "Date when customer signed up",
            "is_active": "Whether the customer is currently active",
            "churn_date": "Date when customer churned (if applicable)",
            "lifetime_value_score": "Predicted lifetime value score (0-100)",
            "total_orders": "Total number of orders placed",
            "total_spent": "Total amount spent by customer",
            "avg_order_value": "Average order value",
            "days_since_signup": "Number of days since signup",
            "last_activity_date": "Date of last customer activity",
            "email_opt_in": "Whether customer opted in for email marketing",
            "sms_opt_in": "Whether customer opted in for SMS marketing",
        }
        
        if self.business_context.business_model == BusinessModel.B2B:
            descriptions.update({
                "company_name": "Name of the customer's company",
                "company_size": "Size category of the company",
                "company_industry": "Industry of the company",
                "account_manager": "Assigned account manager",
                "contract_value": "Total contract value",
            })
        
        if self.business_context.industry == Industry.SAAS:
            descriptions.update({
                "subscription_plan": "Current subscription plan",
                "monthly_recurring_revenue": "Monthly recurring revenue from this customer",
                "billing_cycle": "Billing cycle preference",
            })
        
        return descriptions
