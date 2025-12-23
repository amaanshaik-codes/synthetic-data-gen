# Synthetic Data Generator

Generate realistic data warehouse datasets for learning analytics, ETL, and data cleaning.

## Installation

```bash
pip install pandas numpy faker rich pyarrow
```

## Usage

```bash
python synthdata.py
```

Select industry, size, time period, quality level, and output format.

## Industries

### Retail
Star schema for brick-and-mortar stores.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_customer | Customer profiles with loyalty tiers |
| dim_product | SKU, category, brand, pricing |
| dim_store | Store locations, formats, regions |
| fact_sales | Transactions with revenue, quantity, discounts |
| fact_inventory | Stock levels by product and store |

### E-commerce
Star + event model for online retail.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_customer | Customer profiles with acquisition channel |
| dim_product | Product catalog |
| dim_channel | Web, mobile, marketplace platforms |
| fact_orders | Order headers with status, shipping |
| fact_order_items | Line items with quantities, prices |
| fact_web_events | Clickstream: views, cart, checkout |

### Banking
Snowflake-style for financial services.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_customer | Customer profiles with credit scores |
| dim_account | Account types, balances, status |
| dim_branch | Branch locations and regions |
| fact_transactions | Deposits, withdrawals, transfers |
| fact_loans | Loan details, EMI, interest rates |
| fact_cards | Card transactions by merchant |

### Healthcare
Fact constellation for patient care.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_patient | Patient demographics, insurance |
| dim_doctor | Physicians with specialties |
| dim_hospital | Facilities with bed counts |
| dim_diagnosis | ICD-10 codes and descriptions |
| fact_encounters | Patient visits with charges |
| fact_procedures | CPT codes and billing |
| fact_billing | Claims with allowed/paid amounts |

### SaaS
Star + snapshot for subscription business.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_customer | Company profiles, employee counts |
| dim_plan | Pricing tiers (Free to Enterprise) |
| dim_product | Platform and add-on products |
| fact_subscriptions | Monthly MRR/ARR snapshots |
| fact_usage | Feature usage, API calls, logins |
| fact_churn | Churned customers with reasons |

### Logistics
Accumulating snapshot for shipping.

| Table | Description |
|-------|-------------|
| dim_date | Calendar dimension |
| dim_route | Origin-destination pairs, distances |
| dim_vehicle | Fleet details, capacity, fuel type |
| dim_warehouse | Distribution centers |
| fact_shipments | Shipment costs and weights |
| fact_deliveries | Delivery status, on-time tracking |
| fact_fleet_usage | Daily vehicle utilization |

## Quality Levels

| Level | Rate | Use Case |
|-------|------|----------|
| clean | 0% | Schema learning |
| light | 2% | Basic cleaning practice |
| moderate | 5% | Realistic scenarios |
| heavy | 10% | Advanced cleaning challenges |

## Programmatic Usage

```python
from generators import RetailGenerator
from quality import QualityInjector

gen = RetailGenerator(n_customers=1000, n_transactions=10000, months=12)
tables = gen.generate_all()

injector = QualityInjector(quality_rate=0.05)
for name, df in tables.items():
    tables[name] = injector.inject_issues(df, is_dimension=name.startswith("dim_"))
    df.to_csv(f"{name}.csv", index=False)
```

## License

MIT
