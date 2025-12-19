"""
Transaction data generator.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    DataQualityConfig,
    Industry,
    BusinessModel,
    AnalyticsUseCase,
)
from synthdata.generators.base import BaseGenerator


class TransactionGenerator(BaseGenerator):
    """Generator for transaction/order data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        self.transaction_types = self._get_transaction_types()
        self.statuses = self._get_transaction_statuses()
        self.payment_methods = self.industry_config.get(
            "payment_methods",
            ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
        )
    
    def _get_transaction_types(self) -> List[str]:
        """Get transaction types based on industry."""
        types = {
            Industry.RETAIL: ["Purchase", "Return", "Exchange"],
            Industry.ECOMMERCE: ["Order", "Return", "Refund", "Subscription"],
            Industry.FINTECH: [
                "Transfer", "Payment", "Withdrawal", "Deposit",
                "Investment", "Loan Payment", "Fee", "Refund"
            ],
            Industry.HEALTHCARE: [
                "Consultation", "Procedure", "Lab Test", "Pharmacy",
                "Insurance Claim", "Co-pay", "Refund"
            ],
            Industry.SAAS: [
                "Subscription", "Upgrade", "Downgrade", "Add-on",
                "Renewal", "Cancellation", "Refund"
            ],
            Industry.MANUFACTURING: [
                "Raw Material", "Component", "Finished Good",
                "Spare Part", "Return", "Credit Note"
            ],
            Industry.MARKETING: [
                "Campaign Spend", "Ad Spend", "Creative Cost",
                "Platform Fee", "Refund"
            ],
            Industry.LOGISTICS: [
                "Shipment", "Express", "Freight", "Return Shipment",
                "Insurance", "Storage Fee"
            ],
        }
        return types.get(self.business_context.industry, ["Purchase", "Return", "Refund"])
    
    def _get_transaction_statuses(self) -> List[str]:
        """Get transaction statuses based on industry."""
        if self.business_context.industry == Industry.LOGISTICS:
            return [
                "Pending", "Picked Up", "In Transit", "Out for Delivery",
                "Delivered", "Failed", "Returned"
            ]
        elif self.business_context.industry == Industry.FINTECH:
            return ["Pending", "Processing", "Completed", "Failed", "Reversed"]
        elif self.business_context.industry == Industry.HEALTHCARE:
            return ["Scheduled", "In Progress", "Completed", "Cancelled", "No Show"]
        else:
            return ["Pending", "Processing", "Completed", "Shipped", "Delivered", "Cancelled", "Refunded"]
    
    def _generate_transaction_amount(
        self,
        customer_segment: Optional[str] = None,
        transaction_type: str = "Purchase",
    ) -> float:
        """Generate transaction amount based on context."""
        min_val, max_val = self.industry_config.get("avg_order_value", (20, 200))
        
        # Segment multiplier
        segment_multipliers = {
            "Budget": 0.5, "Value": 0.8, "Standard": 1.0,
            "Premium": 1.5, "VIP": 2.5, "Luxury": 4.0,
            "Enterprise": 3.0, "Free": 0.0, "Starter": 0.5,
            "Professional": 1.5,
        }
        multiplier = segment_multipliers.get(customer_segment, 1.0)
        
        # Type adjustment
        if transaction_type in ["Return", "Refund"]:
            multiplier *= -1
        elif transaction_type in ["Fee"]:
            min_val, max_val = 1, 50
        
        # Generate with right-skewed distribution (more low values)
        amount = self.generate_skewed_value(min_val, max_val, skew=1.5) * multiplier
        
        return round(amount, 2)
    
    def _generate_fraud_indicators(
        self,
        amount: float,
        is_fraud: bool,
    ) -> Dict[str, Any]:
        """Generate fraud-related features."""
        if is_fraud:
            # Fraudulent transactions often have unusual patterns
            return {
                "is_international": random.random() > 0.3,
                "is_night_transaction": random.random() > 0.4,
                "device_change": random.random() > 0.5,
                "ip_mismatch": random.random() > 0.4,
                "velocity_flag": random.random() > 0.3,
                "amount_deviation": random.uniform(2, 5),
            }
        else:
            return {
                "is_international": random.random() > 0.85,
                "is_night_transaction": random.random() > 0.9,
                "device_change": random.random() > 0.95,
                "ip_mismatch": random.random() > 0.98,
                "velocity_flag": random.random() > 0.98,
                "amount_deviation": random.uniform(0, 1.5),
            }
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        customer_ids: Optional[List[str]] = None,
        customer_segments: Optional[Dict[str, str]] = None,
        customer_signup_dates: Optional[Dict[str, datetime]] = None,
        product_ids: Optional[List[str]] = None,
        include_fraud_labels: bool = False,
        fraud_rate: float = 0.02,
        **kwargs
    ) -> pd.DataFrame:
        """Generate transaction data."""
        if num_rows is None:
            # Calculate based on daily transactions and time span
            total_days = self.business_context.time_span_months * 30
            num_rows = self.business_size.daily_transactions * total_days
        
        transactions = []
        
        for i in range(num_rows):
            transaction_id = self.generate_id("TXN", 10)
            
            # Select customer
            if customer_ids:
                customer_id = random.choice(customer_ids)
                customer_segment = customer_segments.get(customer_id) if customer_segments else None
                signup_date = customer_signup_dates.get(customer_id) if customer_signup_dates else None
            else:
                customer_id = self.generate_id("CUS", 8)
                customer_segment = None
                signup_date = None
            
            # Generate date (after customer signup if known)
            if signup_date:
                transaction_date = self.random_date(start=signup_date)
            else:
                transaction_date = self.random_date()
            
            # Apply seasonality
            seasonality = self.generate_seasonality_factor(transaction_date)
            
            # Transaction type with weights
            type_weights = [0.8] + [0.2 / (len(self.transaction_types) - 1)] * (len(self.transaction_types) - 1)
            transaction_type = self.weighted_choice(self.transaction_types, type_weights)
            
            # Amount
            amount = self._generate_transaction_amount(customer_segment, transaction_type)
            amount = amount * seasonality
            
            # Status with weights (most completed)
            status_weights = [0.05, 0.05, 0.75, 0.05, 0.05, 0.03, 0.02][:len(self.statuses)]
            status = self.weighted_choice(self.statuses, status_weights)
            
            # Payment method
            payment_method = random.choice(self.payment_methods)
            
            # Product (if provided)
            product_id = random.choice(product_ids) if product_ids else None
            quantity = random.randint(1, 5) if product_id else 1
            
            # Build transaction record
            transaction = {
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": transaction_date,
                "transaction_type": transaction_type,
                "amount": abs(round(amount, 2)),
                "currency": "USD",
                "status": status,
                "payment_method": payment_method,
                "quantity": quantity,
                "unit_price": round(abs(amount) / quantity, 2),
                "discount_amount": round(random.uniform(0, abs(amount) * 0.2), 2) if random.random() > 0.7 else 0.0,
                "tax_amount": round(abs(amount) * 0.08, 2),
                "shipping_amount": round(random.uniform(0, 15), 2) if random.random() > 0.3 else 0.0,
                "channel": random.choice(self.industry_config.get("channels", ["Online", "Mobile"])),
                "device_type": random.choice(["Desktop", "Mobile", "Tablet", "POS"]),
                "session_id": self.generate_id("SES", 12),
            }
            
            # Add product_id if available
            if product_id:
                transaction["product_id"] = product_id
            
            # Add fraud-related fields for fintech
            if include_fraud_labels or self.business_context.industry == Industry.FINTECH:
                is_fraud = random.random() < fraud_rate
                transaction["is_fraud"] = is_fraud
                fraud_indicators = self._generate_fraud_indicators(amount, is_fraud)
                transaction.update(fraud_indicators)
            
            # Add industry-specific fields
            if self.business_context.industry == Industry.ECOMMERCE:
                transaction["order_id"] = self.generate_id("ORD", 8)
                transaction["promo_code"] = self._generate_promo_code() if random.random() > 0.8 else None
                transaction["cart_abandonment"] = random.random() < 0.2 and status == "Pending"
            
            elif self.business_context.industry == Industry.SAAS:
                transaction["subscription_id"] = self.generate_id("SUB", 8)
                transaction["billing_period_start"] = transaction_date
                transaction["billing_period_end"] = transaction_date + timedelta(days=30)
                transaction["is_renewal"] = random.random() > 0.7
            
            elif self.business_context.industry == Industry.LOGISTICS:
                transaction["shipment_id"] = self.generate_id("SHP", 10)
                transaction["origin"] = self.faker.city()
                transaction["destination"] = self.faker.city()
                transaction["weight_kg"] = round(random.uniform(0.5, 50), 2)
                transaction["carrier"] = random.choice(
                    self.industry_config.get("carriers", ["Internal"])
                )
            
            elif self.business_context.industry == Industry.HEALTHCARE:
                transaction["visit_id"] = self.generate_id("VIS", 8)
                transaction["provider_id"] = self.generate_id("DOC", 6)
                transaction["department"] = random.choice(
                    self.industry_config.get("departments", ["General"])
                )
                transaction["insurance_covered"] = round(abs(amount) * random.uniform(0.5, 0.9), 2)
            
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Sort by date
        df = df.sort_values("transaction_date").reset_index(drop=True)
        
        return df
    
    def _generate_promo_code(self) -> str:
        """Generate a promo code."""
        prefixes = ["SAVE", "DEAL", "OFFER", "PROMO", "SALE", "VIP", "NEW"]
        return f"{random.choice(prefixes)}{random.randint(10, 50)}"
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for transaction data."""
        schema = {
            "transaction_id": "string",
            "customer_id": "string",
            "transaction_date": "datetime",
            "transaction_type": "category",
            "amount": "float",
            "currency": "category",
            "status": "category",
            "payment_method": "category",
            "quantity": "integer",
            "unit_price": "float",
            "discount_amount": "float",
            "tax_amount": "float",
            "shipping_amount": "float",
            "channel": "category",
            "device_type": "category",
            "session_id": "string",
        }
        
        if self.business_context.industry == Industry.FINTECH:
            schema.update({
                "is_fraud": "boolean",
                "is_international": "boolean",
                "is_night_transaction": "boolean",
                "device_change": "boolean",
                "ip_mismatch": "boolean",
                "velocity_flag": "boolean",
                "amount_deviation": "float",
            })
        
        return schema
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        return {
            "transaction_id": "Unique identifier for the transaction",
            "customer_id": "Reference to the customer who made the transaction",
            "transaction_date": "Date and time of the transaction",
            "transaction_type": "Type of transaction",
            "amount": "Transaction amount",
            "currency": "Currency of the transaction",
            "status": "Current status of the transaction",
            "payment_method": "Payment method used",
            "quantity": "Number of items in the transaction",
            "unit_price": "Price per unit",
            "discount_amount": "Discount applied to the transaction",
            "tax_amount": "Tax amount",
            "shipping_amount": "Shipping cost",
            "channel": "Channel through which the transaction was made",
            "device_type": "Type of device used",
            "session_id": "Session identifier",
            "is_fraud": "Whether the transaction is fraudulent (for ML training)",
            "is_international": "Whether the transaction crosses borders",
            "is_night_transaction": "Whether the transaction occurred at night",
            "device_change": "Whether a new device was used",
            "ip_mismatch": "Whether IP location mismatches profile",
            "velocity_flag": "Whether transaction velocity is unusual",
            "amount_deviation": "Standard deviations from customer's average",
        }
