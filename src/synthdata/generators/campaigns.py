"""
Marketing campaign data generator.
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
)
from synthdata.generators.base import BaseGenerator


class CampaignGenerator(BaseGenerator):
    """Generator for marketing campaign data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        self.campaign_types = self._get_campaign_types()
        self.channels = self._get_channels()
    
    def _get_campaign_types(self) -> List[str]:
        """Get campaign types."""
        return [
            "Brand Awareness", "Lead Generation", "Product Launch",
            "Seasonal Promotion", "Retargeting", "Loyalty Program",
            "Referral", "Flash Sale", "Content Marketing", "Event"
        ]
    
    def _get_channels(self) -> List[str]:
        """Get marketing channels."""
        return [
            "Email", "Facebook", "Instagram", "Google Ads", "LinkedIn",
            "Twitter", "YouTube", "TikTok", "Display Ads", "Affiliate",
            "SMS", "Push Notification", "Podcast", "Influencer"
        ]
    
    def _generate_campaign_name(self, campaign_type: str, channel: str) -> str:
        """Generate a campaign name."""
        themes = [
            "Summer", "Winter", "Spring", "Fall", "Holiday",
            "Black Friday", "Cyber Monday", "New Year",
            "Valentine's", "Back to School"
        ]
        
        actions = ["Boost", "Drive", "Launch", "Engage", "Convert", "Grow"]
        
        if random.random() > 0.5:
            return f"{random.choice(themes)} {campaign_type} - {channel}"
        else:
            return f"{random.choice(actions)} {campaign_type} Q{random.randint(1, 4)}"
    
    def _generate_campaign_metrics(
        self,
        budget: float,
        channel: str,
        campaign_type: str,
    ) -> Dict[str, Any]:
        """Generate campaign performance metrics."""
        # Channel-specific benchmarks (impressions per dollar)
        channel_efficiency = {
            "Email": (100, 500),
            "Facebook": (50, 200),
            "Instagram": (40, 180),
            "Google Ads": (20, 100),
            "LinkedIn": (10, 50),
            "Twitter": (30, 150),
            "YouTube": (15, 80),
            "TikTok": (60, 300),
            "Display Ads": (100, 400),
            "Affiliate": (5, 30),
            "SMS": (80, 200),
            "Push Notification": (200, 1000),
        }
        
        eff_range = channel_efficiency.get(channel, (50, 150))
        impressions_per_dollar = random.uniform(*eff_range)
        
        impressions = int(budget * impressions_per_dollar)
        
        # CTR varies by channel
        ctr_ranges = {
            "Email": (0.015, 0.05),
            "Facebook": (0.008, 0.025),
            "Instagram": (0.005, 0.02),
            "Google Ads": (0.02, 0.08),
            "LinkedIn": (0.003, 0.015),
            "Display Ads": (0.001, 0.005),
        }
        
        ctr = random.uniform(*ctr_ranges.get(channel, (0.005, 0.02)))
        clicks = int(impressions * ctr)
        
        # Conversion rate
        conv_rate = random.uniform(0.01, 0.08)
        conversions = int(clicks * conv_rate)
        
        # Revenue (some campaigns don't have direct revenue)
        if campaign_type in ["Brand Awareness", "Content Marketing"]:
            revenue = 0
        else:
            avg_order_value = random.uniform(50, 200)
            revenue = conversions * avg_order_value
        
        return {
            "impressions": impressions,
            "clicks": clicks,
            "ctr": round(ctr * 100, 2),  # As percentage
            "conversions": conversions,
            "conversion_rate": round(conv_rate * 100, 2),
            "revenue": round(revenue, 2),
            "cost_per_click": round(budget / max(clicks, 1), 2),
            "cost_per_conversion": round(budget / max(conversions, 1), 2),
            "roas": round(revenue / max(budget, 1), 2),  # Return on ad spend
        }
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Generate campaign data."""
        if num_rows is None:
            # Estimate campaigns based on business size
            monthly_campaigns = {
                "startup": 5,
                "sme": 15,
                "enterprise": 50,
            }
            campaigns_per_month = monthly_campaigns.get(
                self.business_size.scale.value, 10
            )
            num_rows = campaigns_per_month * self.business_context.time_span_months
        
        campaigns = []
        
        for i in range(num_rows):
            campaign_id = self.generate_id("CMP", 8)
            
            # Campaign details
            campaign_type = random.choice(self.campaign_types)
            channel = random.choice(self.channels)
            name = self._generate_campaign_name(campaign_type, channel)
            
            # Dates
            start_date = self.random_date()
            duration_days = random.choice([7, 14, 30, 60, 90])
            end_date = start_date + timedelta(days=duration_days)
            
            # Ensure end date is not in the future
            if end_date > self.end_date:
                end_date = self.end_date
            
            # Budget based on business size
            budget_ranges = {
                "startup": (100, 5000),
                "sme": (500, 25000),
                "enterprise": (5000, 500000),
            }
            budget_range = budget_ranges.get(
                self.business_size.scale.value, (500, 10000)
            )
            budget = round(self.generate_skewed_value(*budget_range, skew=1.5), 2)
            
            # Status
            if end_date < datetime.now():
                status = random.choice(["Completed", "Completed", "Completed", "Paused"])
            elif start_date > datetime.now():
                status = "Scheduled"
            else:
                status = random.choice(["Active", "Active", "Paused"])
            
            # Metrics (only for completed or active campaigns)
            if status in ["Completed", "Active"]:
                spent = budget * random.uniform(0.7, 1.0) if status == "Completed" else budget * random.uniform(0.3, 0.7)
                metrics = self._generate_campaign_metrics(spent, channel, campaign_type)
            else:
                spent = 0
                metrics = {
                    "impressions": 0, "clicks": 0, "ctr": 0,
                    "conversions": 0, "conversion_rate": 0,
                    "revenue": 0, "cost_per_click": 0,
                    "cost_per_conversion": 0, "roas": 0,
                }
            
            # Target audience
            audiences = [
                "All Users", "New Customers", "Returning Customers",
                "High Value", "Cart Abandoners", "Lookalike", "Retargeting"
            ]
            
            campaign = {
                "campaign_id": campaign_id,
                "name": name,
                "campaign_type": campaign_type,
                "channel": channel,
                "status": status,
                "start_date": start_date,
                "end_date": end_date,
                "budget": budget,
                "spent": round(spent, 2),
                "target_audience": random.choice(audiences),
                "geographic_target": self.get_country_for_record(),
                "objective": random.choice(["Awareness", "Consideration", "Conversion"]),
                **metrics,
                "a_b_test": random.random() < 0.3,
                "creative_count": random.randint(1, 10),
                "created_by": self.faker.name(),
                "created_date": start_date - timedelta(days=random.randint(1, 14)),
            }
            
            # Industry-specific fields
            if self.business_context.industry == Industry.SAAS:
                campaign["target_plan"] = random.choice(["Free", "Starter", "Pro", "Enterprise", "All"])
                campaign["trial_signups"] = int(metrics["conversions"] * 0.7) if metrics["conversions"] > 0 else 0
                campaign["paid_conversions"] = metrics["conversions"] - campaign["trial_signups"]
            
            campaigns.append(campaign)
        
        df = pd.DataFrame(campaigns)
        
        # Sort by start date
        df = df.sort_values("start_date").reset_index(drop=True)
        
        return df
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for campaign data."""
        return {
            "campaign_id": "string",
            "name": "string",
            "campaign_type": "category",
            "channel": "category",
            "status": "category",
            "start_date": "datetime",
            "end_date": "datetime",
            "budget": "float",
            "spent": "float",
            "target_audience": "category",
            "geographic_target": "category",
            "objective": "category",
            "impressions": "integer",
            "clicks": "integer",
            "ctr": "float",
            "conversions": "integer",
            "conversion_rate": "float",
            "revenue": "float",
            "cost_per_click": "float",
            "cost_per_conversion": "float",
            "roas": "float",
            "a_b_test": "boolean",
            "creative_count": "integer",
            "created_by": "string",
            "created_date": "datetime",
        }
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        return {
            "campaign_id": "Unique identifier for the campaign",
            "name": "Campaign name",
            "campaign_type": "Type of marketing campaign",
            "channel": "Marketing channel",
            "status": "Campaign status",
            "start_date": "Campaign start date",
            "end_date": "Campaign end date",
            "budget": "Allocated budget",
            "spent": "Amount spent",
            "target_audience": "Target audience segment",
            "geographic_target": "Geographic targeting",
            "objective": "Campaign objective",
            "impressions": "Number of impressions",
            "clicks": "Number of clicks",
            "ctr": "Click-through rate (%)",
            "conversions": "Number of conversions",
            "conversion_rate": "Conversion rate (%)",
            "revenue": "Revenue attributed to campaign",
            "cost_per_click": "Cost per click",
            "cost_per_conversion": "Cost per conversion",
            "roas": "Return on ad spend",
            "a_b_test": "Whether A/B testing was used",
            "creative_count": "Number of creatives used",
            "created_by": "Campaign creator",
            "created_date": "Date campaign was created",
        }
