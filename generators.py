import random
from datetime import datetime, timedelta
from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from faker import Faker


class BaseGenerator(ABC):
    def __init__(self, months: int = 12, seed: int = 42):
        self.months = months
        self.seed = seed
        self.end_date = datetime.now().replace(day=1) - timedelta(days=1)
        self.start_date = self.end_date - timedelta(days=months * 30)
        
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        self.dim_date = None

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        if delta.days <= 0:
            return start
        return start + timedelta(days=random.randint(0, delta.days))

    def _get_quarter(self, month: int) -> str:
        return f"Q{(month - 1) // 3 + 1}"

    def _pareto_weights(self, n: int, ratio: float = 0.8) -> np.ndarray:
        top_n = max(1, int(n * 0.2))
        weights = np.zeros(n)
        weights[:top_n] = ratio / top_n
        weights[top_n:] = (1 - ratio) / max(1, n - top_n)
        np.random.shuffle(weights)
        return weights / weights.sum()

    def generate_dim_date(self) -> pd.DataFrame:
        dates = pd.date_range(self.start_date, self.end_date, freq="D")
        data = []
        for i, d in enumerate(dates):
            data.append({
                "date_key": i + 1,
                "full_date": d.strftime("%Y-%m-%d"),
                "year": d.year,
                "quarter": self._get_quarter(d.month),
                "month": d.month,
                "month_name": d.strftime("%B"),
                "week": d.isocalendar()[1],
                "day": d.day,
                "day_of_week": d.weekday(),
                "day_name": d.strftime("%A"),
                "is_weekend": d.weekday() >= 5,
                "is_holiday": d.month == 12 and d.day in [24, 25, 31],
            })
        self.dim_date = pd.DataFrame(data)
        return self.dim_date

    @abstractmethod
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        pass


class RetailGenerator(BaseGenerator):
    def __init__(self, n_customers: int = 500, n_products: int = 100, n_stores: int = 20, 
                 n_transactions: int = 5000, months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_stores = n_stores
        self.n_transactions = n_transactions

    def generate_dim_customer(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_customers):
            data.append({
                "customer_key": i + 1,
                "customer_id": f"CUST-{i+1:06d}",
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "age": random.randint(18, 75),
                "gender": random.choice(["M", "F"]),
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "loyalty_tier": random.choices(["Bronze", "Silver", "Gold", "Platinum"], weights=[0.5, 0.3, 0.15, 0.05])[0],
                "signup_date": self._random_date(self.start_date - timedelta(days=365), self.end_date).strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(data)

    def generate_dim_product(self) -> pd.DataFrame:
        categories = ["Electronics", "Clothing", "Grocery", "Home", "Sports", "Beauty"]
        data = []
        for i in range(self.n_products):
            cat = random.choice(categories)
            price = round(random.uniform(5, 500), 2)
            data.append({
                "product_key": i + 1,
                "sku": f"SKU-{i+1:06d}",
                "product_name": f"{self.fake.word().title()} {cat[:4]} {random.randint(100,999)}",
                "category": cat,
                "brand": self.fake.company().split()[0],
                "unit_price": price,
                "unit_cost": round(price * random.uniform(0.4, 0.7), 2),
            })
        return pd.DataFrame(data)

    def generate_dim_store(self) -> pd.DataFrame:
        formats = ["Superstore", "Express", "Outlet", "Mall"]
        data = []
        for i in range(self.n_stores):
            data.append({
                "store_key": i + 1,
                "store_id": f"STR-{i+1:04d}",
                "store_name": f"{self.fake.city()} {random.choice(formats)}",
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "region": random.choice(["North", "South", "East", "West"]),
                "format": random.choice(formats),
                "sqft": random.randint(5000, 50000),
                "open_date": self._random_date(self.start_date - timedelta(days=1825), self.start_date).strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(data)

    def generate_fact_sales(self, dim_customer, dim_product, dim_store) -> pd.DataFrame:
        cust_weights = self._pareto_weights(len(dim_customer))
        prod_weights = self._pareto_weights(len(dim_product))
        data = []
        
        for i in range(self.n_transactions):
            cust_idx = np.random.choice(len(dim_customer), p=cust_weights)
            prod_idx = np.random.choice(len(dim_product), p=prod_weights)
            store_idx = random.randint(0, len(dim_store) - 1)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            
            qty = random.randint(1, 10)
            price = dim_product.iloc[prod_idx]["unit_price"]
            discount = round(random.uniform(0, 0.2) * price * qty, 2)
            
            data.append({
                "sale_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "product_key": dim_product.iloc[prod_idx]["product_key"],
                "store_key": dim_store.iloc[store_idx]["store_key"],
                "quantity": qty,
                "unit_price": price,
                "discount": discount,
                "revenue": round(price * qty - discount, 2),
            })
        return pd.DataFrame(data)

    def generate_fact_inventory(self, dim_product, dim_store) -> pd.DataFrame:
        data = []
        for _, prod in dim_product.iterrows():
            for _, store in dim_store.iterrows():
                stock = random.randint(0, 500)
                reorder = random.randint(20, 100)
                data.append({
                    "product_key": prod["product_key"],
                    "store_key": store["store_key"],
                    "date_key": self.dim_date.iloc[-1]["date_key"],
                    "stock_on_hand": stock,
                    "reorder_level": reorder,
                    "is_low_stock": stock < reorder,
                })
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_customer = self.generate_dim_customer()
        dim_product = self.generate_dim_product()
        dim_store = self.generate_dim_store()
        fact_sales = self.generate_fact_sales(dim_customer, dim_product, dim_store)
        fact_inventory = self.generate_fact_inventory(dim_product, dim_store)
        
        return {
            "dim_date": dim_date,
            "dim_customer": dim_customer,
            "dim_product": dim_product,
            "dim_store": dim_store,
            "fact_sales": fact_sales,
            "fact_inventory": fact_inventory,
        }


class EcommerceGenerator(BaseGenerator):
    def __init__(self, n_customers: int = 500, n_products: int = 100, 
                 n_orders: int = 2000, months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_orders = n_orders

    def generate_dim_customer(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_customers):
            data.append({
                "customer_key": i + 1,
                "customer_id": f"CUST-{i+1:06d}",
                "email": self.fake.email(),
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "country": "USA",
                "signup_date": self._random_date(self.start_date - timedelta(days=365), self.end_date).strftime("%Y-%m-%d"),
                "acquisition_channel": random.choice(["Organic", "Paid Search", "Social", "Referral", "Email"]),
            })
        return pd.DataFrame(data)

    def generate_dim_product(self) -> pd.DataFrame:
        categories = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books"]
        data = []
        for i in range(self.n_products):
            cat = random.choice(categories)
            price = round(random.uniform(10, 1000), 2)
            data.append({
                "product_key": i + 1,
                "product_id": f"PROD-{i+1:06d}",
                "product_name": f"{self.fake.word().title()} {cat[:4]} {random.randint(100,999)}",
                "category": cat,
                "brand": self.fake.company().split()[0],
                "unit_price": price,
            })
        return pd.DataFrame(data)

    def generate_dim_channel(self) -> pd.DataFrame:
        channels = [
            {"channel_key": 1, "channel_name": "Web", "platform": "Desktop"},
            {"channel_key": 2, "channel_name": "Mobile App", "platform": "iOS"},
            {"channel_key": 3, "channel_name": "Mobile App", "platform": "Android"},
            {"channel_key": 4, "channel_name": "Marketplace", "platform": "Amazon"},
            {"channel_key": 5, "channel_name": "Social", "platform": "Instagram"},
        ]
        return pd.DataFrame(channels)

    def generate_fact_orders(self, dim_customer, dim_channel) -> pd.DataFrame:
        cust_weights = self._pareto_weights(len(dim_customer))
        data = []
        
        for i in range(self.n_orders):
            cust_idx = np.random.choice(len(dim_customer), p=cust_weights)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            channel_idx = random.randint(0, len(dim_channel) - 1)
            
            data.append({
                "order_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "channel_key": dim_channel.iloc[channel_idx]["channel_key"],
                "order_value": 0,
                "shipping_cost": round(random.uniform(0, 15), 2),
                "status": random.choices(["Delivered", "Shipped", "Processing", "Cancelled"], weights=[0.7, 0.15, 0.1, 0.05])[0],
            })
        return pd.DataFrame(data)

    def generate_fact_order_items(self, fact_orders, dim_product) -> pd.DataFrame:
        prod_weights = self._pareto_weights(len(dim_product))
        data = []
        item_id = 1
        
        for _, order in fact_orders.iterrows():
            n_items = random.randint(1, 5)
            order_total = 0
            
            for _ in range(n_items):
                prod_idx = np.random.choice(len(dim_product), p=prod_weights)
                qty = random.randint(1, 3)
                price = dim_product.iloc[prod_idx]["unit_price"]
                line_total = round(price * qty, 2)
                order_total += line_total
                
                data.append({
                    "item_id": item_id,
                    "order_id": order["order_id"],
                    "product_key": dim_product.iloc[prod_idx]["product_key"],
                    "quantity": qty,
                    "unit_price": price,
                    "line_total": line_total,
                })
                item_id += 1
            
            fact_orders.loc[fact_orders["order_id"] == order["order_id"], "order_value"] = order_total
        
        return pd.DataFrame(data)

    def generate_fact_web_events(self, dim_customer, dim_product, dim_channel) -> pd.DataFrame:
        n_events = self.n_orders * 10
        cust_weights = self._pareto_weights(len(dim_customer))
        prod_weights = self._pareto_weights(len(dim_product))
        events = ["page_view", "product_view", "add_to_cart", "remove_from_cart", "checkout_start", "purchase"]
        data = []
        
        for i in range(n_events):
            data.append({
                "event_id": i + 1,
                "date_key": self.dim_date.iloc[random.randint(0, len(self.dim_date) - 1)]["date_key"],
                "customer_key": dim_customer.iloc[np.random.choice(len(dim_customer), p=cust_weights)]["customer_key"],
                "product_key": dim_product.iloc[np.random.choice(len(dim_product), p=prod_weights)]["product_key"] if random.random() > 0.3 else None,
                "channel_key": dim_channel.iloc[random.randint(0, len(dim_channel) - 1)]["channel_key"],
                "event_type": random.choices(events, weights=[0.4, 0.25, 0.15, 0.05, 0.1, 0.05])[0],
                "session_id": f"SES-{random.randint(100000, 999999)}",
            })
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_customer = self.generate_dim_customer()
        dim_product = self.generate_dim_product()
        dim_channel = self.generate_dim_channel()
        fact_orders = self.generate_fact_orders(dim_customer, dim_channel)
        fact_order_items = self.generate_fact_order_items(fact_orders, dim_product)
        fact_web_events = self.generate_fact_web_events(dim_customer, dim_product, dim_channel)
        
        return {
            "dim_date": dim_date,
            "dim_customer": dim_customer,
            "dim_product": dim_product,
            "dim_channel": dim_channel,
            "fact_orders": fact_orders,
            "fact_order_items": fact_order_items,
            "fact_web_events": fact_web_events,
        }


class BankingGenerator(BaseGenerator):
    def __init__(self, n_customers: int = 500, n_accounts: int = 800, n_branches: int = 30,
                 n_transactions: int = 10000, months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_customers = n_customers
        self.n_accounts = n_accounts
        self.n_branches = n_branches
        self.n_transactions = n_transactions

    def generate_dim_customer(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_customers):
            data.append({
                "customer_key": i + 1,
                "customer_id": f"CIF-{i+1:08d}",
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "date_of_birth": self.fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d"),
                "ssn_last4": f"{random.randint(1000, 9999)}",
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "credit_score": random.randint(300, 850),
                "customer_since": self._random_date(self.start_date - timedelta(days=3650), self.start_date).strftime("%Y-%m-%d"),
                "segment": random.choices(["Mass", "Affluent", "Private", "Business"], weights=[0.6, 0.25, 0.1, 0.05])[0],
            })
        return pd.DataFrame(data)

    def generate_dim_account(self, dim_customer) -> pd.DataFrame:
        account_types = ["Checking", "Savings", "Money Market", "CD", "IRA"]
        data = []
        for i in range(self.n_accounts):
            cust_idx = random.randint(0, len(dim_customer) - 1)
            acct_type = random.choice(account_types)
            data.append({
                "account_key": i + 1,
                "account_id": f"ACCT-{i+1:010d}",
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "account_type": acct_type,
                "open_date": self._random_date(self.start_date - timedelta(days=1825), self.end_date).strftime("%Y-%m-%d"),
                "status": random.choices(["Active", "Dormant", "Closed"], weights=[0.85, 0.1, 0.05])[0],
                "current_balance": round(random.uniform(100, 100000), 2),
            })
        return pd.DataFrame(data)

    def generate_dim_branch(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_branches):
            data.append({
                "branch_key": i + 1,
                "branch_id": f"BR-{i+1:04d}",
                "branch_name": f"{self.fake.city()} Branch",
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "region": random.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"]),
            })
        return pd.DataFrame(data)

    def generate_fact_transactions(self, dim_account, dim_branch) -> pd.DataFrame:
        txn_types = ["Deposit", "Withdrawal", "Transfer", "Payment", "Fee", "Interest"]
        acct_weights = self._pareto_weights(len(dim_account))
        data = []
        
        for i in range(self.n_transactions):
            acct_idx = np.random.choice(len(dim_account), p=acct_weights)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            branch_idx = random.randint(0, len(dim_branch) - 1)
            txn_type = random.choices(txn_types, weights=[0.25, 0.2, 0.2, 0.2, 0.1, 0.05])[0]
            
            amount = round(random.uniform(10, 5000), 2)
            if txn_type in ["Withdrawal", "Payment", "Fee"]:
                amount = -amount
            
            data.append({
                "transaction_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "account_key": dim_account.iloc[acct_idx]["account_key"],
                "branch_key": dim_branch.iloc[branch_idx]["branch_key"],
                "transaction_type": txn_type,
                "amount": amount,
                "channel": random.choice(["Branch", "ATM", "Online", "Mobile", "Wire"]),
                "reference": f"REF-{random.randint(100000000, 999999999)}",
            })
        return pd.DataFrame(data)

    def generate_fact_loans(self, dim_customer, dim_branch) -> pd.DataFrame:
        n_loans = self.n_customers // 3
        loan_types = ["Mortgage", "Auto", "Personal", "Student", "Business"]
        data = []
        
        for i in range(n_loans):
            cust_idx = random.randint(0, len(dim_customer) - 1)
            branch_idx = random.randint(0, len(dim_branch) - 1)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            loan_type = random.choice(loan_types)
            
            if loan_type == "Mortgage":
                principal = round(random.uniform(100000, 500000), 2)
                rate = round(random.uniform(3, 7), 2)
                term = random.choice([180, 240, 360])
            elif loan_type == "Auto":
                principal = round(random.uniform(10000, 60000), 2)
                rate = round(random.uniform(4, 10), 2)
                term = random.choice([36, 48, 60, 72])
            else:
                principal = round(random.uniform(5000, 50000), 2)
                rate = round(random.uniform(6, 18), 2)
                term = random.choice([12, 24, 36, 48, 60])
            
            monthly_payment = round((principal * (rate/100/12)) / (1 - (1 + rate/100/12)**(-term)), 2)
            
            data.append({
                "loan_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "branch_key": dim_branch.iloc[branch_idx]["branch_key"],
                "loan_type": loan_type,
                "principal": principal,
                "interest_rate": rate,
                "term_months": term,
                "monthly_payment": monthly_payment,
                "status": random.choices(["Current", "Delinquent", "Default", "Paid Off"], weights=[0.8, 0.1, 0.02, 0.08])[0],
            })
        return pd.DataFrame(data)

    def generate_fact_cards(self, dim_customer) -> pd.DataFrame:
        n_card_txns = self.n_transactions // 2
        cust_weights = self._pareto_weights(len(dim_customer))
        merchants = ["Amazon", "Walmart", "Target", "Starbucks", "Shell", "Uber", "Netflix", "Restaurant", "Grocery", "Travel"]
        data = []
        
        for i in range(n_card_txns):
            cust_idx = np.random.choice(len(dim_customer), p=cust_weights)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            
            data.append({
                "card_txn_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "merchant_category": random.choice(merchants),
                "amount": round(random.uniform(5, 500), 2),
                "card_type": random.choice(["Debit", "Credit"]),
                "is_international": random.random() < 0.05,
            })
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_customer = self.generate_dim_customer()
        dim_account = self.generate_dim_account(dim_customer)
        dim_branch = self.generate_dim_branch()
        fact_transactions = self.generate_fact_transactions(dim_account, dim_branch)
        fact_loans = self.generate_fact_loans(dim_customer, dim_branch)
        fact_cards = self.generate_fact_cards(dim_customer)
        
        return {
            "dim_date": dim_date,
            "dim_customer": dim_customer,
            "dim_account": dim_account,
            "dim_branch": dim_branch,
            "fact_transactions": fact_transactions,
            "fact_loans": fact_loans,
            "fact_cards": fact_cards,
        }


class HealthcareGenerator(BaseGenerator):
    def __init__(self, n_patients: int = 500, n_doctors: int = 50, n_hospitals: int = 10,
                 n_encounters: int = 3000, months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_patients = n_patients
        self.n_doctors = n_doctors
        self.n_hospitals = n_hospitals
        self.n_encounters = n_encounters

    def generate_dim_patient(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_patients):
            dob = self.fake.date_of_birth(minimum_age=1, maximum_age=90)
            data.append({
                "patient_key": i + 1,
                "patient_id": f"MRN-{i+1:08d}",
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "date_of_birth": dob.strftime("%Y-%m-%d"),
                "gender": random.choice(["M", "F"]),
                "blood_type": random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "insurance_type": random.choice(["Commercial", "Medicare", "Medicaid", "Self-Pay"]),
            })
        return pd.DataFrame(data)

    def generate_dim_doctor(self) -> pd.DataFrame:
        specialties = ["Family Medicine", "Internal Medicine", "Cardiology", "Orthopedics", 
                       "Pediatrics", "Neurology", "Oncology", "Emergency Medicine"]
        data = []
        for i in range(self.n_doctors):
            data.append({
                "doctor_key": i + 1,
                "doctor_id": f"NPI-{random.randint(1000000000, 9999999999)}",
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "specialty": random.choice(specialties),
                "years_experience": random.randint(1, 35),
            })
        return pd.DataFrame(data)

    def generate_dim_hospital(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_hospitals):
            data.append({
                "hospital_key": i + 1,
                "hospital_id": f"HOSP-{i+1:04d}",
                "hospital_name": f"{self.fake.city()} {random.choice(['Medical Center', 'Hospital', 'Health System'])}",
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "bed_count": random.randint(50, 500),
                "hospital_type": random.choice(["General", "Teaching", "Specialty", "Community"]),
            })
        return pd.DataFrame(data)

    def generate_dim_diagnosis(self) -> pd.DataFrame:
        diagnoses = [
            ("J06.9", "Acute upper respiratory infection"),
            ("I10", "Essential hypertension"),
            ("E11.9", "Type 2 diabetes mellitus"),
            ("M54.5", "Low back pain"),
            ("J18.9", "Pneumonia"),
            ("K21.0", "GERD"),
            ("F32.9", "Major depressive disorder"),
            ("J45.909", "Asthma"),
            ("M79.3", "Panniculitis"),
            ("R10.9", "Abdominal pain"),
        ]
        data = []
        for i, (code, desc) in enumerate(diagnoses):
            data.append({
                "diagnosis_key": i + 1,
                "icd10_code": code,
                "diagnosis_description": desc,
                "category": desc.split()[0],
            })
        return pd.DataFrame(data)

    def generate_fact_encounters(self, dim_patient, dim_doctor, dim_hospital, dim_diagnosis) -> pd.DataFrame:
        visit_types = ["Outpatient", "Inpatient", "Emergency", "Observation"]
        patient_weights = self._pareto_weights(len(dim_patient))
        data = []
        
        for i in range(self.n_encounters):
            patient_idx = np.random.choice(len(dim_patient), p=patient_weights)
            doctor_idx = random.randint(0, len(dim_doctor) - 1)
            hospital_idx = random.randint(0, len(dim_hospital) - 1)
            diagnosis_idx = random.randint(0, len(dim_diagnosis) - 1)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            visit_type = random.choices(visit_types, weights=[0.6, 0.2, 0.15, 0.05])[0]
            
            los = 0 if visit_type == "Outpatient" else random.randint(1, 10)
            
            data.append({
                "encounter_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "patient_key": dim_patient.iloc[patient_idx]["patient_key"],
                "doctor_key": dim_doctor.iloc[doctor_idx]["doctor_key"],
                "hospital_key": dim_hospital.iloc[hospital_idx]["hospital_key"],
                "diagnosis_key": dim_diagnosis.iloc[diagnosis_idx]["diagnosis_key"],
                "visit_type": visit_type,
                "length_of_stay": los,
                "total_charges": round(random.uniform(200, 50000) if visit_type != "Outpatient" else random.uniform(100, 1000), 2),
            })
        return pd.DataFrame(data)

    def generate_fact_procedures(self, fact_encounters, dim_doctor) -> pd.DataFrame:
        procedures = [
            ("99213", "Office visit - established"),
            ("99214", "Office visit - detailed"),
            ("36415", "Blood draw"),
            ("71046", "Chest X-ray"),
            ("93000", "ECG"),
            ("80053", "Comprehensive metabolic panel"),
            ("85025", "Complete blood count"),
        ]
        data = []
        proc_id = 1
        
        for _, enc in fact_encounters.iterrows():
            n_procs = random.randint(1, 4)
            for _ in range(n_procs):
                proc = random.choice(procedures)
                data.append({
                    "procedure_id": proc_id,
                    "encounter_id": enc["encounter_id"],
                    "doctor_key": enc["doctor_key"],
                    "cpt_code": proc[0],
                    "procedure_description": proc[1],
                    "charge_amount": round(random.uniform(50, 2000), 2),
                })
                proc_id += 1
        return pd.DataFrame(data)

    def generate_fact_billing(self, fact_encounters) -> pd.DataFrame:
        data = []
        for _, enc in fact_encounters.iterrows():
            charges = enc["total_charges"]
            allowed = round(charges * random.uniform(0.3, 0.8), 2)
            paid = round(allowed * random.uniform(0.7, 1.0), 2)
            
            data.append({
                "billing_id": enc["encounter_id"],
                "encounter_id": enc["encounter_id"],
                "total_charges": charges,
                "allowed_amount": allowed,
                "paid_amount": paid,
                "patient_responsibility": round(charges - paid, 2),
                "claim_status": random.choices(["Paid", "Pending", "Denied", "Appealed"], weights=[0.7, 0.15, 0.1, 0.05])[0],
            })
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_patient = self.generate_dim_patient()
        dim_doctor = self.generate_dim_doctor()
        dim_hospital = self.generate_dim_hospital()
        dim_diagnosis = self.generate_dim_diagnosis()
        fact_encounters = self.generate_fact_encounters(dim_patient, dim_doctor, dim_hospital, dim_diagnosis)
        fact_procedures = self.generate_fact_procedures(fact_encounters, dim_doctor)
        fact_billing = self.generate_fact_billing(fact_encounters)
        
        return {
            "dim_date": dim_date,
            "dim_patient": dim_patient,
            "dim_doctor": dim_doctor,
            "dim_hospital": dim_hospital,
            "dim_diagnosis": dim_diagnosis,
            "fact_encounters": fact_encounters,
            "fact_procedures": fact_procedures,
            "fact_billing": fact_billing,
        }


class SaasGenerator(BaseGenerator):
    def __init__(self, n_customers: int = 300, n_plans: int = 5, 
                 months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_customers = n_customers
        self.n_plans = n_plans

    def generate_dim_customer(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_customers):
            data.append({
                "customer_key": i + 1,
                "customer_id": f"ORG-{i+1:06d}",
                "company_name": self.fake.company(),
                "industry": random.choice(["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]),
                "employee_count": random.choice([10, 50, 100, 500, 1000, 5000]),
                "country": random.choice(["USA", "UK", "Canada", "Germany", "Australia"]),
                "signup_date": self._random_date(self.start_date - timedelta(days=730), self.end_date).strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(data)

    def generate_dim_plan(self) -> pd.DataFrame:
        plans = [
            {"plan_key": 1, "plan_name": "Free", "monthly_price": 0, "annual_price": 0, "features": "Basic"},
            {"plan_key": 2, "plan_name": "Starter", "monthly_price": 29, "annual_price": 290, "features": "Core"},
            {"plan_key": 3, "plan_name": "Pro", "monthly_price": 99, "annual_price": 990, "features": "Advanced"},
            {"plan_key": 4, "plan_name": "Business", "monthly_price": 299, "annual_price": 2990, "features": "Full"},
            {"plan_key": 5, "plan_name": "Enterprise", "monthly_price": 999, "annual_price": 9990, "features": "Custom"},
        ]
        return pd.DataFrame(plans)

    def generate_dim_product(self) -> pd.DataFrame:
        products = [
            {"product_key": 1, "product_name": "Core Platform", "category": "Platform"},
            {"product_key": 2, "product_name": "Analytics Add-on", "category": "Add-on"},
            {"product_key": 3, "product_name": "API Access", "category": "Add-on"},
            {"product_key": 4, "product_name": "Premium Support", "category": "Service"},
            {"product_key": 5, "product_name": "Training", "category": "Service"},
        ]
        return pd.DataFrame(products)

    def generate_fact_subscriptions(self, dim_customer, dim_plan) -> pd.DataFrame:
        data = []
        for date_idx in range(len(self.dim_date)):
            if self.dim_date.iloc[date_idx]["day"] != 1:
                continue
            
            for _, cust in dim_customer.iterrows():
                signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")
                current = datetime.strptime(self.dim_date.iloc[date_idx]["full_date"], "%Y-%m-%d")
                
                if current < signup:
                    continue
                
                plan_idx = random.choices(range(len(dim_plan)), weights=[0.1, 0.3, 0.35, 0.2, 0.05])[0]
                plan = dim_plan.iloc[plan_idx]
                
                data.append({
                    "snapshot_date_key": self.dim_date.iloc[date_idx]["date_key"],
                    "customer_key": cust["customer_key"],
                    "plan_key": plan["plan_key"],
                    "mrr": plan["monthly_price"],
                    "arr": plan["monthly_price"] * 12,
                    "status": random.choices(["Active", "Churned", "Paused"], weights=[0.9, 0.07, 0.03])[0],
                    "seats": random.randint(1, 50) if plan["plan_key"] > 1 else 1,
                })
        return pd.DataFrame(data)

    def generate_fact_usage(self, dim_customer) -> pd.DataFrame:
        n_events = self.n_customers * len(self.dim_date) // 3
        cust_weights = self._pareto_weights(len(dim_customer))
        features = ["Dashboard", "Reports", "API", "Export", "Import", "Settings", "Users"]
        data = []
        
        for i in range(n_events):
            cust_idx = np.random.choice(len(dim_customer), p=cust_weights)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            
            data.append({
                "usage_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "customer_key": dim_customer.iloc[cust_idx]["customer_key"],
                "feature": random.choice(features),
                "logins": random.randint(1, 50),
                "api_calls": random.randint(0, 10000),
                "data_storage_mb": random.randint(10, 5000),
            })
        return pd.DataFrame(data)

    def generate_fact_churn(self, dim_customer, dim_plan) -> pd.DataFrame:
        n_churns = int(self.n_customers * 0.15)
        reasons = ["Price", "Features", "Competition", "No longer needed", "Poor support", "Other"]
        data = []
        
        churned_customers = random.sample(range(len(dim_customer)), n_churns)
        
        for i, cust_idx in enumerate(churned_customers):
            cust = dim_customer.iloc[cust_idx]
            signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")
            
            if signup >= self.end_date:
                continue
            
            churn_date = self._random_date(signup + timedelta(days=30), self.end_date)
            date_idx = (churn_date - self.start_date).days
            if date_idx < 0 or date_idx >= len(self.dim_date):
                continue
            
            plan_idx = random.randint(0, len(dim_plan) - 1)
            
            data.append({
                "churn_id": i + 1,
                "date_key": self.dim_date.iloc[min(date_idx, len(self.dim_date) - 1)]["date_key"],
                "customer_key": cust["customer_key"],
                "plan_key": dim_plan.iloc[plan_idx]["plan_key"],
                "mrr_lost": dim_plan.iloc[plan_idx]["monthly_price"],
                "reason": random.choice(reasons),
                "tenure_months": (churn_date - signup).days // 30,
            })
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_customer = self.generate_dim_customer()
        dim_plan = self.generate_dim_plan()
        dim_product = self.generate_dim_product()
        fact_subscriptions = self.generate_fact_subscriptions(dim_customer, dim_plan)
        fact_usage = self.generate_fact_usage(dim_customer)
        fact_churn = self.generate_fact_churn(dim_customer, dim_plan)
        
        return {
            "dim_date": dim_date,
            "dim_customer": dim_customer,
            "dim_plan": dim_plan,
            "dim_product": dim_product,
            "fact_subscriptions": fact_subscriptions,
            "fact_usage": fact_usage,
            "fact_churn": fact_churn,
        }


class LogisticsGenerator(BaseGenerator):
    def __init__(self, n_routes: int = 100, n_vehicles: int = 50, n_warehouses: int = 10,
                 n_shipments: int = 5000, months: int = 12, seed: int = 42):
        super().__init__(months, seed)
        self.n_routes = n_routes
        self.n_vehicles = n_vehicles
        self.n_warehouses = n_warehouses
        self.n_shipments = n_shipments

    def generate_dim_route(self) -> pd.DataFrame:
        data = []
        cities = [self.fake.city() for _ in range(30)]
        for i in range(self.n_routes):
            origin, dest = random.sample(cities, 2)
            data.append({
                "route_key": i + 1,
                "route_id": f"RT-{i+1:05d}",
                "origin_city": origin,
                "destination_city": dest,
                "distance_miles": random.randint(50, 3000),
                "transit_days": random.randint(1, 7),
                "route_type": random.choice(["Ground", "Air", "Ocean", "Rail"]),
            })
        return pd.DataFrame(data)

    def generate_dim_vehicle(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_vehicles):
            vtype = random.choice(["Truck", "Van", "Cargo Plane", "Container Ship", "Train"])
            data.append({
                "vehicle_key": i + 1,
                "vehicle_id": f"VH-{i+1:05d}",
                "vehicle_type": vtype,
                "capacity_lbs": random.randint(1000, 100000),
                "fuel_type": random.choice(["Diesel", "Gasoline", "Electric", "Hybrid"]),
                "year": random.randint(2015, 2024),
                "status": random.choices(["Active", "Maintenance", "Retired"], weights=[0.85, 0.1, 0.05])[0],
            })
        return pd.DataFrame(data)

    def generate_dim_warehouse(self) -> pd.DataFrame:
        data = []
        for i in range(self.n_warehouses):
            data.append({
                "warehouse_key": i + 1,
                "warehouse_id": f"WH-{i+1:04d}",
                "warehouse_name": f"{self.fake.city()} DC",
                "city": self.fake.city(),
                "state": self.fake.state_abbr(),
                "sqft": random.randint(50000, 500000),
                "warehouse_type": random.choice(["Distribution", "Fulfillment", "Cross-Dock", "Cold Storage"]),
            })
        return pd.DataFrame(data)

    def generate_fact_shipments(self, dim_route, dim_vehicle, dim_warehouse) -> pd.DataFrame:
        route_weights = self._pareto_weights(len(dim_route))
        data = []
        
        for i in range(self.n_shipments):
            route_idx = np.random.choice(len(dim_route), p=route_weights)
            vehicle_idx = random.randint(0, len(dim_vehicle) - 1)
            origin_wh = random.randint(0, len(dim_warehouse) - 1)
            dest_wh = random.randint(0, len(dim_warehouse) - 1)
            date_idx = random.randint(0, len(self.dim_date) - 1)
            
            route = dim_route.iloc[route_idx]
            weight = random.randint(100, 10000)
            cost_per_lb = random.uniform(0.5, 2.0)
            
            data.append({
                "shipment_id": i + 1,
                "date_key": self.dim_date.iloc[date_idx]["date_key"],
                "route_key": route["route_key"],
                "vehicle_key": dim_vehicle.iloc[vehicle_idx]["vehicle_key"],
                "origin_warehouse_key": dim_warehouse.iloc[origin_wh]["warehouse_key"],
                "dest_warehouse_key": dim_warehouse.iloc[dest_wh]["warehouse_key"],
                "weight_lbs": weight,
                "distance_miles": route["distance_miles"],
                "shipping_cost": round(weight * cost_per_lb, 2),
            })
        return pd.DataFrame(data)

    def generate_fact_deliveries(self, fact_shipments, dim_route) -> pd.DataFrame:
        statuses = ["Delivered", "In Transit", "Out for Delivery", "Delayed", "Failed"]
        data = []
        
        for _, ship in fact_shipments.iterrows():
            route = dim_route[dim_route["route_key"] == ship["route_key"]].iloc[0]
            expected_days = route["transit_days"]
            actual_days = expected_days + random.randint(-1, 3)
            status = random.choices(statuses, weights=[0.75, 0.1, 0.05, 0.07, 0.03])[0]
            
            data.append({
                "delivery_id": ship["shipment_id"],
                "shipment_id": ship["shipment_id"],
                "expected_transit_days": expected_days,
                "actual_transit_days": max(1, actual_days) if status == "Delivered" else None,
                "delivery_status": status,
                "on_time": actual_days <= expected_days if status == "Delivered" else None,
                "delivery_attempts": random.randint(1, 3) if status in ["Delivered", "Failed"] else 0,
            })
        return pd.DataFrame(data)

    def generate_fact_fleet_usage(self, dim_vehicle) -> pd.DataFrame:
        data = []
        usage_id = 1
        
        for date_idx in range(len(self.dim_date)):
            active_vehicles = dim_vehicle[dim_vehicle["status"] == "Active"]
            for _, vehicle in active_vehicles.iterrows():
                if random.random() > 0.7:
                    continue
                
                data.append({
                    "usage_id": usage_id,
                    "date_key": self.dim_date.iloc[date_idx]["date_key"],
                    "vehicle_key": vehicle["vehicle_key"],
                    "miles_driven": random.randint(50, 500),
                    "fuel_gallons": round(random.uniform(20, 100), 2),
                    "hours_operated": round(random.uniform(4, 12), 1),
                    "maintenance_flag": random.random() < 0.05,
                })
                usage_id += 1
        return pd.DataFrame(data)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        dim_date = self.generate_dim_date()
        dim_route = self.generate_dim_route()
        dim_vehicle = self.generate_dim_vehicle()
        dim_warehouse = self.generate_dim_warehouse()
        fact_shipments = self.generate_fact_shipments(dim_route, dim_vehicle, dim_warehouse)
        fact_deliveries = self.generate_fact_deliveries(fact_shipments, dim_route)
        fact_fleet_usage = self.generate_fact_fleet_usage(dim_vehicle)
        
        return {
            "dim_date": dim_date,
            "dim_route": dim_route,
            "dim_vehicle": dim_vehicle,
            "dim_warehouse": dim_warehouse,
            "fact_shipments": fact_shipments,
            "fact_deliveries": fact_deliveries,
            "fact_fleet_usage": fact_fleet_usage,
        }


GENERATORS = {
    "retail": RetailGenerator,
    "ecommerce": EcommerceGenerator,
    "banking": BankingGenerator,
    "healthcare": HealthcareGenerator,
    "saas": SaasGenerator,
    "logistics": LogisticsGenerator,
}
