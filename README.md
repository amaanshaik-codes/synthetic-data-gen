# Synthetic Data Generator

Generate realistic data warehouse datasets for learning analytics, ETL, and data cleaning.

## Features

- **6 Industries**: Retail, E-commerce, Banking, Healthcare, SaaS, Logistics
- **Multiple Formats**: CSV, Parquet, JSON, Excel, SQLite
- **Realistic Patterns**: Growth trends, seasonality, customer behavior
- **Quality Control**: Configurable data quality issues for practice
- **Detailed Reports**: Analysis guide with SQL queries and insights
- **Flexible Config**: Custom sizes, time periods, growth rates

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python synthdata.py
```

Interactive wizard guides you through:
1. Industry selection
2. Dataset size (small/medium/large)
3. Time period (months of data)
4. Quality level (clean to heavy issues)
5. Output format
6. Advanced options (seed, growth rate, reports)

## Output Formats

| Format | Use Case |
|--------|----------|
| CSV | Universal compatibility, easy to inspect |
| Parquet | Efficient storage, big data tools |
| JSON | API integration, NoSQL databases |
| Excel | Business users, quick analysis |
| SQLite | Single-file database, SQL practice |

## Generated Reports

Each run produces `ANALYSIS_REPORT.md` with:
- Dataset statistics and profiling
- Industry-specific insights and metrics
- Sample SQL queries for analysis
- Data quality assessment
- Recommended cleaning steps
- Python code templates
- Business questions to explore

## Installation

```bash
pip install pandas numpy faker rich pyarrow openpyxl
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
from report import generate_report
from pathlib import Path

# Generate data with growth trend
gen = RetailGenerator(
    n_customers=1000,
    n_transactions=10000,
    months=12,
    growth_rate=0.05,  # 5% monthly growth
    seed=42
)
tables = gen.generate_all()

# Inject quality issues
injector = QualityInjector(quality_rate=0.05)
for name, df in tables.items():
    tables[name] = injector.inject_issues(df, is_dimension=name.startswith("dim_"))

# Save in multiple formats
for name, df in tables.items():
    df.to_csv(f"{name}.csv", index=False)
    df.to_parquet(f"{name}.parquet", index=False)
    df.to_json(f"{name}.json", orient="records", lines=True)

# Generate analysis report
config = {
    "industry": "retail",
    "months": 12,
    "quality": "moderate",
    "quality_rate": 0.05,
    "growth_rate": 5
}
generate_report(tables, config, Path("ANALYSIS_REPORT.md"))
```

## License

MIT
