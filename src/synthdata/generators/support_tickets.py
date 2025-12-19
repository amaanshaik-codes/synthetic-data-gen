"""
Support ticket data generator.
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


class SupportTicketGenerator(BaseGenerator):
    """Generator for customer support ticket data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        self.categories = self._get_ticket_categories()
        self.priorities = ["Low", "Medium", "High", "Urgent"]
        self.channels = ["Email", "Phone", "Chat", "Social Media", "Web Form", "In-App"]
    
    def _get_ticket_categories(self) -> Dict[str, List[str]]:
        """Get ticket categories and subcategories based on industry."""
        categories = {
            Industry.RETAIL: {
                "Order Issues": ["Missing Item", "Wrong Item", "Damaged", "Delayed"],
                "Returns": ["Return Request", "Refund Status", "Exchange"],
                "Product": ["Quality", "Availability", "Information"],
                "Payment": ["Billing Error", "Payment Failed", "Refund"],
                "Account": ["Login", "Password", "Profile Update"],
            },
            Industry.ECOMMERCE: {
                "Shipping": ["Tracking", "Delayed", "Lost Package", "Address Change"],
                "Orders": ["Cancellation", "Modification", "Status"],
                "Products": ["Defective", "Not as Described", "Missing Parts"],
                "Payments": ["Failed Transaction", "Double Charge", "Promo Code"],
                "Technical": ["Website Issue", "App Bug", "Checkout Error"],
            },
            Industry.FINTECH: {
                "Transactions": ["Failed", "Disputed", "Unauthorized", "Delayed"],
                "Account": ["Verification", "Limits", "Closure", "Security"],
                "Cards": ["Lost", "Stolen", "PIN Reset", "Replacement"],
                "Payments": ["Transfer Failed", "Wrong Amount", "Recipient Issue"],
                "Compliance": ["KYC", "Documentation", "Verification"],
            },
            Industry.SAAS: {
                "Technical": ["Bug Report", "Feature Request", "Integration Issue"],
                "Billing": ["Invoice", "Subscription", "Upgrade", "Downgrade"],
                "Access": ["Login Issue", "Permissions", "SSO", "API Access"],
                "Onboarding": ["Setup Help", "Training", "Documentation"],
                "Performance": ["Slow", "Downtime", "Data Loss"],
            },
            Industry.HEALTHCARE: {
                "Appointments": ["Scheduling", "Cancellation", "Rescheduling"],
                "Billing": ["Insurance", "Co-pay", "Statement"],
                "Records": ["Access", "Update", "Transfer"],
                "Prescriptions": ["Refill", "Questions", "Side Effects"],
                "General": ["Feedback", "Complaint", "Information"],
            },
        }
        
        return categories.get(
            self.business_context.industry,
            {
                "General": ["Question", "Complaint", "Feedback"],
                "Technical": ["Bug", "Error", "Help Needed"],
                "Billing": ["Payment", "Invoice", "Refund"],
            }
        )
    
    def _generate_ticket_subject(self, category: str, subcategory: str) -> str:
        """Generate a ticket subject."""
        templates = {
            "Order Issues": [
                f"Problem with my order - {subcategory}",
                f"Order {subcategory.lower()} issue",
                f"Need help with {subcategory.lower()} order",
            ],
            "Shipping": [
                f"Where is my package? - {subcategory}",
                f"{subcategory} shipping issue",
                f"Delivery problem - {subcategory}",
            ],
            "Technical": [
                f"Bug: {subcategory}",
                f"Issue: {subcategory}",
                f"Help needed: {subcategory}",
            ],
            "Billing": [
                f"Billing question - {subcategory}",
                f"{subcategory} issue with payment",
                f"Need clarification on {subcategory.lower()}",
            ],
        }
        
        template_list = templates.get(category, [f"{category} - {subcategory}"])
        return random.choice(template_list)
    
    def _generate_resolution_time(
        self,
        priority: str,
        category: str,
    ) -> timedelta:
        """Generate resolution time based on priority and category."""
        # Base resolution times in hours
        priority_times = {
            "Urgent": (1, 8),
            "High": (4, 24),
            "Medium": (24, 72),
            "Low": (48, 168),
        }
        
        min_hours, max_hours = priority_times.get(priority, (24, 72))
        hours = random.uniform(min_hours, max_hours)
        
        return timedelta(hours=hours)
    
    def _generate_sentiment_score(self, was_resolved: bool, resolution_time_hours: float) -> float:
        """Generate customer sentiment score."""
        if was_resolved:
            # Faster resolution = happier customer
            if resolution_time_hours < 4:
                base_score = random.uniform(0.7, 1.0)
            elif resolution_time_hours < 24:
                base_score = random.uniform(0.5, 0.9)
            elif resolution_time_hours < 72:
                base_score = random.uniform(0.3, 0.7)
            else:
                base_score = random.uniform(0.1, 0.5)
        else:
            base_score = random.uniform(-0.5, 0.3)
        
        return round(base_score, 2)
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        customer_ids: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Generate support ticket data."""
        if num_rows is None:
            # Estimate tickets based on customers and time span
            tickets_per_customer_per_month = 0.2
            num_rows = int(
                self.business_size.num_customers *
                tickets_per_customer_per_month *
                self.business_context.time_span_months
            )
        
        tickets = []
        
        for i in range(num_rows):
            ticket_id = self.generate_id("TKT", 8)
            
            # Customer
            if customer_ids:
                customer_id = random.choice(customer_ids)
            else:
                customer_id = self.generate_id("CUS", 8)
            
            # Category and subcategory
            category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[category])
            
            # Priority (weighted towards medium/low)
            priority_weights = [0.35, 0.40, 0.20, 0.05]
            priority = self.weighted_choice(self.priorities, priority_weights)
            
            # Channel
            channel_weights = [0.30, 0.15, 0.30, 0.10, 0.10, 0.05]
            channel = self.weighted_choice(self.channels, channel_weights)
            
            # Dates
            created_date = self.random_date()
            
            # Status and resolution
            status_weights = [0.05, 0.10, 0.10, 0.05, 0.65, 0.05]
            status = self.weighted_choice(
                ["New", "Open", "Pending", "On Hold", "Resolved", "Closed"],
                status_weights
            )
            
            # Resolution
            if status in ["Resolved", "Closed"]:
                resolution_time = self._generate_resolution_time(priority, category)
                resolved_date = created_date + resolution_time
                if resolved_date > self.end_date:
                    resolved_date = self.end_date
                first_response_time = timedelta(hours=random.uniform(0.5, min(24, resolution_time.total_seconds() / 3600)))
            else:
                resolved_date = None
                first_response_time = timedelta(hours=random.uniform(0.5, 24)) if status != "New" else None
            
            # Calculate times
            resolution_hours = (resolved_date - created_date).total_seconds() / 3600 if resolved_date else None
            first_response_hours = first_response_time.total_seconds() / 3600 if first_response_time else None
            
            # Agent assignment
            agent_id = self.generate_id("AGT", 6) if status != "New" else None
            
            # Sentiment
            was_resolved = status in ["Resolved", "Closed"]
            sentiment = self._generate_sentiment_score(was_resolved, resolution_hours or 999)
            
            ticket = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "subject": self._generate_ticket_subject(category, subcategory),
                "description": self.faker.paragraph(nb_sentences=random.randint(2, 5)),
                "category": category,
                "subcategory": subcategory,
                "priority": priority,
                "status": status,
                "channel": channel,
                "created_date": created_date,
                "first_response_date": created_date + first_response_time if first_response_time else None,
                "resolved_date": resolved_date,
                "first_response_hours": round(first_response_hours, 2) if first_response_hours else None,
                "resolution_hours": round(resolution_hours, 2) if resolution_hours else None,
                "agent_id": agent_id,
                "escalated": random.random() < 0.1,
                "reopened": random.random() < 0.05 and was_resolved,
                "num_interactions": random.randint(1, 10),
                "sentiment_score": sentiment,
                "satisfaction_rating": random.randint(1, 5) if was_resolved and random.random() > 0.3 else None,
                "tags": self._generate_tags(category, subcategory),
            }
            
            # Industry-specific fields
            if self.business_context.industry == Industry.ECOMMERCE:
                ticket["order_id"] = self.generate_id("ORD", 8) if category in ["Orders", "Shipping"] else None
                ticket["product_id"] = self.generate_id("PRD", 8) if category == "Products" else None
            
            elif self.business_context.industry == Industry.FINTECH:
                ticket["transaction_id"] = self.generate_id("TXN", 10) if category == "Transactions" else None
                ticket["amount_involved"] = round(random.uniform(10, 5000), 2) if category == "Transactions" else None
            
            elif self.business_context.industry == Industry.SAAS:
                ticket["feature_area"] = random.choice(["Dashboard", "Reports", "API", "Integrations", "Settings"])
                ticket["subscription_plan"] = random.choice(["Free", "Starter", "Pro", "Enterprise"])
            
            tickets.append(ticket)
        
        df = pd.DataFrame(tickets)
        
        # Sort by created date
        df = df.sort_values("created_date").reset_index(drop=True)
        
        return df
    
    def _generate_tags(self, category: str, subcategory: str) -> str:
        """Generate ticket tags."""
        base_tags = [category.lower().replace(" ", "_"), subcategory.lower().replace(" ", "_")]
        additional_tags = ["follow_up", "vip", "escalated", "feedback", "bug", "feature_request"]
        
        num_additional = random.randint(0, 2)
        selected = random.sample(additional_tags, num_additional)
        
        return ",".join(base_tags + selected)
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for support ticket data."""
        return {
            "ticket_id": "string",
            "customer_id": "string",
            "subject": "string",
            "description": "text",
            "category": "category",
            "subcategory": "category",
            "priority": "category",
            "status": "category",
            "channel": "category",
            "created_date": "datetime",
            "first_response_date": "datetime",
            "resolved_date": "datetime",
            "first_response_hours": "float",
            "resolution_hours": "float",
            "agent_id": "string",
            "escalated": "boolean",
            "reopened": "boolean",
            "num_interactions": "integer",
            "sentiment_score": "float",
            "satisfaction_rating": "integer",
            "tags": "string",
        }
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        return {
            "ticket_id": "Unique identifier for the support ticket",
            "customer_id": "Reference to the customer",
            "subject": "Ticket subject line",
            "description": "Detailed ticket description",
            "category": "Ticket category",
            "subcategory": "Ticket subcategory",
            "priority": "Ticket priority level",
            "status": "Current ticket status",
            "channel": "Channel through which ticket was submitted",
            "created_date": "Date ticket was created",
            "first_response_date": "Date of first response",
            "resolved_date": "Date ticket was resolved",
            "first_response_hours": "Hours until first response",
            "resolution_hours": "Total hours to resolution",
            "agent_id": "Assigned support agent",
            "escalated": "Whether ticket was escalated",
            "reopened": "Whether ticket was reopened",
            "num_interactions": "Number of interactions on ticket",
            "sentiment_score": "Customer sentiment score (-1 to 1)",
            "satisfaction_rating": "Customer satisfaction rating (1-5)",
            "tags": "Ticket tags",
        }
