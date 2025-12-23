"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  DATA GENERATORS - Creates realistic synthetic data for various industries    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from faker import Faker


class DataGenerator:
    """Generates realistic synthetic data for data science practice."""
    
    # Industry-specific configurations
    INDUSTRY_CONFIG = {
        "ecommerce": {
            "categories": ["Electronics", "Clothing", "Home & Garden", "Beauty", "Sports", "Books", "Toys"],
            "payment_methods": ["Credit Card", "PayPal", "Debit Card", "Apple Pay", "Google Pay", "Klarna"],
            "channels": ["Website", "Mobile App", "Marketplace"],
            "ticket_types": ["Shipping", "Return", "Payment", "Product Quality", "Account"],
        },
        "retail": {
            "categories": ["Groceries", "Beverages", "Household", "Personal Care", "Snacks", "Frozen", "Dairy"],
            "payment_methods": ["Cash", "Credit Card", "Debit Card", "Store Credit", "Gift Card"],
            "channels": ["In-Store", "Online", "Click & Collect"],
            "ticket_types": ["Product Issue", "Store Experience", "Refund", "Loyalty Program"],
        },
        "fintech": {
            "categories": ["Savings", "Investments", "Loans", "Insurance", "Crypto", "Transfers"],
            "payment_methods": ["Bank Transfer", "Direct Debit", "Card", "Crypto Wallet"],
            "channels": ["Mobile App", "Web Portal", "API"],
            "ticket_types": ["Transaction Issue", "Account Security", "Verification", "Fee Dispute"],
        },
        "healthcare": {
            "categories": ["Primary Care", "Specialist", "Lab Tests", "Imaging", "Pharmacy", "Therapy"],
            "payment_methods": ["Insurance", "Credit Card", "HSA/FSA", "Payment Plan", "Self-Pay"],
            "channels": ["In-Person", "Telehealth", "Portal"],
            "ticket_types": ["Billing", "Appointment", "Records Request", "Insurance Claim"],
        },
        "saas": {
            "categories": ["Basic", "Pro", "Enterprise", "Add-ons", "Training", "Support"],
            "payment_methods": ["Credit Card", "Invoice", "Bank Transfer", "PayPal"],
            "channels": ["Web App", "API", "Mobile", "Desktop"],
            "ticket_types": ["Bug Report", "Feature Request", "Billing", "Integration", "Access Issue"],
        },
        "logistics": {
            "categories": ["Standard", "Express", "Same-Day", "International", "Freight", "Return"],
            "payment_methods": ["Invoice", "Credit Card", "Account Credit", "COD"],
            "channels": ["API", "Portal", "Email", "Phone"],
            "ticket_types": ["Delayed Shipment", "Lost Package", "Damage", "Address Issue"],
        },
    }
    
    # Common data across industries
    COUNTRIES = ["USA", "UK", "Canada", "Germany", "France", "Australia", "Netherlands", "Spain"]
    STATUSES = ["active", "inactive", "churned", "pending"]
    SEGMENTS = ["Premium", "Standard", "Budget", "New", "VIP"]
    
    def __init__(
        self,
        industry: str = "ecommerce",
        num_customers: int = 1000,
        num_products: int = 100,
        daily_transactions: int = 100,
        months: int = 12,
        seed: int = 42,
    ):
        self.industry = industry
        self.num_customers = num_customers
        self.num_products = num_products
        self.daily_transactions = daily_transactions
        self.months = months
        self.seed = seed
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Get industry config
        self.config = self.INDUSTRY_CONFIG.get(industry, self.INDUSTRY_CONFIG["ecommerce"])
        
        # Calculate date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=months * 30)
    
    def _random_date(self, start: datetime = None, end: datetime = None) -> datetime:
        """Generate random date within range."""
        start = start or self.start_date
        end = end or self.end_date
        delta = end - start
        random_days = random.randint(0, max(0, delta.days))
        random_seconds = random.randint(0, 86400)
        return start + timedelta(days=random_days, seconds=random_seconds)
    
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer data."""
        data = []
        
        for i in range(self.num_customers):
            signup_date = self._random_date()
            
            # Determine if churned based on signup age
            days_since_signup = (self.end_date - signup_date).days
            churn_prob = 0.1 + (days_since_signup / 365) * 0.15
            is_churned = random.random() < churn_prob
            
            # Last active calculation
            if is_churned:
                # Ensure valid range for churn date
                max_churn_days = max(31, min(days_since_signup, 365))
                churn_date = signup_date + timedelta(days=random.randint(30, max_churn_days))
                last_active = churn_date
                status = "churned"
            else:
                last_active = self._random_date(signup_date, self.end_date)
                status = random.choice(["active", "active", "active", "inactive", "pending"])
            
            customer = {
                "customer_id": f"CUST{i+1:06d}",
                "email": self.fake.email(),
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "phone": self.fake.phone_number(),
                "country": random.choice(self.COUNTRIES),
                "city": self.fake.city(),
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "last_active": last_active.strftime("%Y-%m-%d"),
                "status": status,
                "segment": random.choices(
                    self.SEGMENTS,
                    weights=[0.1, 0.4, 0.3, 0.15, 0.05]
                )[0],
                "lifetime_value": round(np.random.exponential(500) + 50, 2),
                "total_orders": 0,  # Will be updated after transactions
                "age": random.randint(18, 75),
                "gender": random.choice(["M", "F", "Other", "Prefer not to say"]),
                "acquisition_channel": random.choice(["Organic", "Paid Search", "Social", "Referral", "Email"]),
            }
            data.append(customer)
        
        return pd.DataFrame(data)
    
    def generate_products(self) -> pd.DataFrame:
        """Generate product/service catalog."""
        data = []
        
        for i in range(self.num_products):
            category = random.choice(self.config["categories"])
            
            # Price varies by category (simulate realistic distribution)
            base_price = np.random.exponential(50) + 10
            price = round(base_price * random.uniform(0.5, 2), 2)
            
            # Cost is 40-80% of price
            cost = round(price * random.uniform(0.4, 0.8), 2)
            
            product = {
                "product_id": f"PROD{i+1:05d}",
                "name": f"{self.fake.word().title()} {category} {random.randint(100, 999)}",
                "category": category,
                "subcategory": f"{category} - {self.fake.word().title()}",
                "price": price,
                "cost": cost,
                "margin": round((price - cost) / price * 100, 1),
                "stock_quantity": random.randint(0, 1000),
                "supplier": self.fake.company(),
                "rating": round(random.uniform(1, 5), 1),
                "review_count": int(np.random.exponential(50)),
                "is_active": random.random() > 0.1,
                "created_date": self._random_date().strftime("%Y-%m-%d"),
                "weight_kg": round(random.uniform(0.1, 20), 2),
                "tags": ",".join(random.sample(["sale", "new", "popular", "limited", "eco", "premium"], random.randint(0, 3))),
            }
            data.append(product)
        
        return pd.DataFrame(data)
    
    def generate_transactions(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate transaction data."""
        data = []
        
        customer_ids = customers["customer_id"].tolist()
        customer_signups = dict(zip(customers["customer_id"], pd.to_datetime(customers["signup_date"])))
        customer_statuses = dict(zip(customers["customer_id"], customers["status"]))
        
        product_ids = products["product_id"].tolist()
        product_prices = dict(zip(products["product_id"], products["price"]))
        product_categories = dict(zip(products["product_id"], products["category"]))
        
        total_days = (self.end_date - self.start_date).days
        transaction_id = 0
        
        for day in range(total_days):
            current_date = self.start_date + timedelta(days=day)
            
            # Vary transactions by day of week (weekends = more)
            day_of_week = current_date.weekday()
            daily_mult = 1.3 if day_of_week >= 5 else 1.0
            
            # Add seasonality (more in Nov-Dec)
            month = current_date.month
            season_mult = 1.5 if month in [11, 12] else (0.8 if month in [1, 2] else 1.0)
            
            num_today = int(self.daily_transactions * daily_mult * season_mult * random.uniform(0.8, 1.2))
            
            for _ in range(num_today):
                transaction_id += 1
                
                # Select customer (weighted by activity)
                customer_id = random.choice(customer_ids)
                signup_date = customer_signups[customer_id]
                
                # Skip if transaction date is before signup
                if current_date < signup_date:
                    continue
                
                # Skip most transactions for churned customers after churn
                if customer_statuses[customer_id] == "churned" and random.random() > 0.1:
                    continue
                
                # Select products (1-5 items per transaction)
                num_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
                selected_products = random.sample(product_ids, min(num_items, len(product_ids)))
                
                for product_id in selected_products:
                    quantity = random.choices([1, 2, 3, 4, 5], weights=[0.6, 0.25, 0.1, 0.03, 0.02])[0]
                    unit_price = product_prices[product_id]
                    
                    # Apply random discount
                    discount_pct = random.choices(
                        [0, 5, 10, 15, 20, 25],
                        weights=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02]
                    )[0]
                    
                    total = round(unit_price * quantity * (1 - discount_pct / 100), 2)
                    
                    # Transaction status
                    status = random.choices(
                        ["completed", "completed", "completed", "refunded", "cancelled", "pending"],
                        weights=[0.85, 0.05, 0.05, 0.02, 0.02, 0.01]
                    )[0]
                    
                    transaction = {
                        "transaction_id": f"TXN{transaction_id:08d}",
                        "customer_id": customer_id,
                        "product_id": product_id,
                        "category": product_categories[product_id],
                        "transaction_date": current_date.strftime("%Y-%m-%d"),
                        "transaction_time": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_percent": discount_pct,
                        "total_amount": total,
                        "payment_method": random.choice(self.config["payment_methods"]),
                        "channel": random.choice(self.config["channels"]),
                        "status": status,
                        "shipping_cost": round(random.uniform(0, 15), 2) if random.random() > 0.3 else 0,
                        "tax_amount": round(total * 0.08, 2),
                    }
                    data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Update customer order counts
        if len(df) > 0:
            order_counts = df.groupby("customer_id")["transaction_id"].nunique()
            customers["total_orders"] = customers["customer_id"].map(order_counts).fillna(0).astype(int)
        
        return df
    
    def generate_support_tickets(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate support ticket data."""
        data = []
        
        customer_ids = customers["customer_id"].tolist()
        product_ids = products["product_id"].tolist()
        
        # About 10-20% of customers will have tickets
        customers_with_tickets = random.sample(
            customer_ids,
            int(len(customer_ids) * random.uniform(0.1, 0.2))
        )
        
        priorities = ["Low", "Medium", "High", "Critical"]
        statuses = ["Open", "In Progress", "Resolved", "Closed", "Escalated"]
        
        ticket_id = 0
        
        for customer_id in customers_with_tickets:
            # Each customer has 1-5 tickets
            num_tickets = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]
            
            for _ in range(num_tickets):
                ticket_id += 1
                
                created = self._random_date()
                
                # Resolution time varies by priority
                priority = random.choices(priorities, weights=[0.3, 0.4, 0.2, 0.1])[0]
                
                if priority == "Critical":
                    resolution_hours = random.randint(1, 24)
                elif priority == "High":
                    resolution_hours = random.randint(4, 72)
                elif priority == "Medium":
                    resolution_hours = random.randint(24, 168)
                else:
                    resolution_hours = random.randint(48, 336)
                
                resolved = created + timedelta(hours=resolution_hours)
                status = random.choices(statuses, weights=[0.1, 0.15, 0.4, 0.3, 0.05])[0]
                
                # If still open, no resolution
                if status in ["Open", "In Progress", "Escalated"]:
                    resolved = None
                    resolution_hours = None
                
                ticket = {
                    "ticket_id": f"TKT{ticket_id:06d}",
                    "customer_id": customer_id,
                    "product_id": random.choice(product_ids) if random.random() > 0.3 else None,
                    "ticket_type": random.choice(self.config["ticket_types"]),
                    "priority": priority,
                    "status": status,
                    "created_date": created.strftime("%Y-%m-%d %H:%M:%S"),
                    "resolved_date": resolved.strftime("%Y-%m-%d %H:%M:%S") if resolved else None,
                    "resolution_hours": resolution_hours,
                    "agent_id": f"AGENT{random.randint(1, 50):03d}",
                    "channel": random.choice(["Email", "Chat", "Phone", "Social Media"]),
                    "satisfaction_score": random.randint(1, 5) if status in ["Resolved", "Closed"] else None,
                    "first_response_minutes": random.randint(5, 120),
                    "num_interactions": random.randint(1, 10),
                    "tags": random.choice(self.config["ticket_types"]),
                }
                data.append(ticket)
        
        return pd.DataFrame(data)
