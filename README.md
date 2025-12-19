# Synthetic Data Generator

A CLI-based synthetic data generator designed for real-world analytics and data science practice. Generate messy, imperfect, and business-relevant datasets to practice EDA, data cleaning, feature engineering, modeling, and storytelling.

## Features

- **Business Context Simulation**: Generate data for retail, ecommerce, fintech, healthcare, SaaS, manufacturing, marketing, and logistics
- **Configurable Data Quality**: Control missing values, duplicates, inconsistencies, outliers, and noise
- **Multiple Difficulty Levels**: From clean datasets to chaotic real-world messes
- **Relationship Management**: One-to-one, one-to-many, and many-to-many table relationships
- **Analytics Focus**: Built-in support for churn prediction, fraud detection, time series, and more
- **Reproducible**: Seed control and config export for exact dataset regeneration

## Installation

```bash
# Clone the repository
cd synthetic-data-gen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Generate a simple retail dataset
synthdata generate --industry retail --size sme --difficulty medium

# Generate with custom config
synthdata generate --config my_config.yaml

# Generate for specific analytics use case
synthdata generate --industry fintech --use-case fraud-detection --difficulty hard

# Interactive mode
synthdata interactive
```

## CLI Commands

### Generate Dataset

```bash
synthdata generate [OPTIONS]

Options:
  --industry TEXT          Industry type: retail|ecommerce|fintech|healthcare|saas|manufacturing|marketing|logistics
  --business-model TEXT    Business model: b2b|b2c|d2c|marketplace|subscription
  --geography TEXT         Geography: single-country|multi-country|global
  --time-span TEXT         Time span (e.g., "2y" for 2 years, "6m" for 6 months)
  --size TEXT              Business size: startup|sme|enterprise
  --customers INTEGER      Number of customers
  --daily-transactions INT Transactions per day
  --revenue-scale TEXT     Revenue scale: low|medium|high
  --difficulty TEXT        Cleaning difficulty: easy|medium|hard|chaotic
  --use-case TEXT          Analytics use case
  --output-format TEXT     Output format: csv|parquet|json|sql
  --output-dir TEXT        Output directory
  --seed INTEGER           Random seed for reproducibility
  --config PATH            Load configuration from YAML/JSON file
  --save-config PATH       Save configuration to file
```

### List Presets

```bash
synthdata presets
```

### Validate Config

```bash
synthdata validate --config my_config.yaml
```

## Configuration File

Create a YAML configuration file for complex setups:

```yaml
business_context:
  industry: ecommerce
  business_model: b2c
  geography: multi-country
  time_span_months: 24

business_size:
  scale: sme
  num_customers: 50000
  daily_transactions: 1000
  revenue_scale: medium

tables:
  customers:
    enabled: true
    columns:
      - name: customer_id
        type: id
      - name: email
        type: email
      - name: signup_date
        type: datetime
      - name: country
        type: category
  transactions:
    enabled: true
    relationship:
      parent: customers
      type: one-to-many
      cardinality: [1, 50]

data_quality:
  missing_values:
    global_rate: 0.05
    per_column:
      email: 0.02
      phone: 0.15
  duplicates:
    rate: 0.01
  inconsistencies:
    date_formats: true
    currency_formats: true
    category_typos: true
  outliers:
    rate: 0.02
    magnitude: 3.0
  noise:
    label_noise: 0.05

difficulty: hard

analytics:
  use_case: churn-prediction
  target_variable: churned
  signal_to_noise: 0.7
  feature_leakage: false
  train_test_split: 0.8
  class_imbalance: 0.1

output:
  format: parquet
  directory: ./output
  include_metadata: true
  include_data_dictionary: true

reproducibility:
  seed: 42
```

## Difficulty Levels

| Level   | Description |
|---------|-------------|
| easy    | Minimal missing values, consistent schema, clean formats |
| medium  | Some nulls, duplicates, minor inconsistencies |
| hard    | Schema changes, corrupted values, mixed units, partial records |
| chaotic | Real-world mess with broken joins, invalid entries, logs |

## Analytics Use Cases

- **descriptive**: Summary statistics, distributions, aggregations
- **diagnostic**: Root cause analysis, correlation exploration
- **predictive**: Classification/regression modeling
- **time-series**: Forecasting with trends and seasonality
- **churn-prediction**: Customer retention analysis
- **fraud-detection**: Anomaly detection with rare events
- **recommendation**: User-item interaction data

## Output Structure

```
output/
├── customers.parquet
├── transactions.parquet
├── products.parquet
├── campaigns.parquet
├── support_tickets.parquet
├── operational_logs.parquet
├── metadata/
│   ├── config.yaml
│   ├── data_dictionary.json
│   ├── quality_report.json
│   └── suggested_questions.md
└── splits/
    ├── train.parquet
    └── test.parquet
```

## License

MIT License
