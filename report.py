from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime


def generate_report(tables: Dict[str, pd.DataFrame], config: dict, output_path: Path):
    industry = config["industry"]
    
    report = []
    report.append(f"# Data Analysis Report: {industry.title()}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Configuration\n")
    report.append(f"- **Industry**: {industry}")
    report.append(f"- **Size**: {config['size']}")
    report.append(f"- **Time Period**: {config['months']} months")
    report.append(f"- **Growth Rate**: {config.get('growth_rate', 5)}% per month")
    report.append(f"- **Quality Level**: {config['quality']} ({int(config['quality_rate']*100)}% issues)")
    report.append(f"- **Total Rows**: {sum(len(df) for df in tables.values()):,}")
    
    report.append(f"\n## Dataset Overview\n")
    for name, df in tables.items():
        report.append(f"### {name}")
        report.append(f"- Rows: {len(df):,}")
        report.append(f"- Columns: {len(df.columns)}")
        report.append(f"- Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            report.append(f"\n**Key Statistics:**")
            for col in numeric_cols[:3]:
                report.append(f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}")
        report.append("")
    
    report.append(f"\n## Data Quality Assessment\n")
    total_nulls = sum(df.isnull().sum().sum() for df in tables.values())
    total_cells = sum(df.shape[0] * df.shape[1] for df in tables.values())
    null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0
    report.append(f"- **Total Missing Values**: {total_nulls:,} ({null_pct:.2f}%)")
    
    if config['quality_rate'] > 0:
        report.append(f"- **Injected Issues**: {config['quality']} level")
        report.append(f"  - Case inconsistencies in text fields")
        report.append(f"  - Whitespace issues (leading/trailing)")
        report.append(f"  - Typos in names and addresses")
        report.append(f"  - Missing values in non-critical fields")
        if any(not name.startswith('dim_') for name in tables.keys()):
            report.append(f"  - Duplicate records in fact tables")
            report.append(f"  - Value anomalies (negative quantities, calculation errors)")
    
    report.append(f"\n## Industry-Specific Insights\n")
    
    if industry == "retail":
        fact_sales = tables.get("fact_sales")
        if fact_sales is not None:
            total_revenue = fact_sales["revenue"].sum()
            avg_transaction = fact_sales["revenue"].mean()
            report.append(f"### Sales Performance")
            report.append(f"- Total Revenue: ${total_revenue:,.2f}")
            report.append(f"- Average Transaction: ${avg_transaction:.2f}")
            report.append(f"- Total Transactions: {len(fact_sales):,}")
            
            top_products = fact_sales.groupby("product_key")["revenue"].sum().nlargest(5)
            report.append(f"\n**Top 5 Products by Revenue:**")
            for i, (prod_key, rev) in enumerate(top_products.items(), 1):
                report.append(f"{i}. Product {prod_key}: ${rev:,.2f}")
    
    elif industry == "ecommerce":
        fact_orders = tables.get("fact_orders")
        if fact_orders is not None:
            total_orders = len(fact_orders)
            total_gmv = fact_orders["order_value"].sum()
            avg_order = fact_orders["order_value"].mean()
            report.append(f"### E-commerce Metrics")
            report.append(f"- Total Orders: {total_orders:,}")
            report.append(f"- Gross Merchandise Value: ${total_gmv:,.2f}")
            report.append(f"- Average Order Value: ${avg_order:.2f}")
            
            status_dist = fact_orders["status"].value_counts()
            report.append(f"\n**Order Status Distribution:**")
            for status, count in status_dist.items():
                pct = count / len(fact_orders) * 100
                report.append(f"- {status}: {count:,} ({pct:.1f}%)")
    
    elif industry == "banking":
        fact_txns = tables.get("fact_transactions")
        if fact_txns is not None:
            deposits = fact_txns[fact_txns["amount"] > 0]["amount"].sum()
            withdrawals = abs(fact_txns[fact_txns["amount"] < 0]["amount"].sum())
            report.append(f"### Banking Activity")
            report.append(f"- Total Deposits: ${deposits:,.2f}")
            report.append(f"- Total Withdrawals: ${withdrawals:,.2f}")
            report.append(f"- Net Flow: ${deposits - withdrawals:,.2f}")
            
            txn_types = fact_txns["transaction_type"].value_counts()
            report.append(f"\n**Transaction Types:**")
            for txn_type, count in txn_types.items():
                report.append(f"- {txn_type}: {count:,}")
    
    elif industry == "healthcare":
        fact_enc = tables.get("fact_encounters")
        if fact_enc is not None:
            total_charges = fact_enc["total_charges"].sum()
            avg_charge = fact_enc["total_charges"].mean()
            report.append(f"### Healthcare Utilization")
            report.append(f"- Total Encounters: {len(fact_enc):,}")
            report.append(f"- Total Charges: ${total_charges:,.2f}")
            report.append(f"- Average Charge: ${avg_charge:.2f}")
            
            visit_types = fact_enc["visit_type"].value_counts()
            report.append(f"\n**Visit Types:**")
            for vtype, count in visit_types.items():
                pct = count / len(fact_enc) * 100
                report.append(f"- {vtype}: {count:,} ({pct:.1f}%)")
    
    elif industry == "saas":
        fact_subs = tables.get("fact_subscriptions")
        if fact_subs is not None:
            active_subs = fact_subs[fact_subs["status"] == "Active"]
            total_mrr = active_subs["mrr"].sum()
            total_arr = active_subs["arr"].sum()
            report.append(f"### SaaS Metrics")
            report.append(f"- Monthly Recurring Revenue: ${total_mrr:,.2f}")
            report.append(f"- Annual Recurring Revenue: ${total_arr:,.2f}")
            report.append(f"- Active Subscriptions: {len(active_subs):,}")
            
            plan_dist = active_subs.groupby("plan_key").size()
            report.append(f"\n**Plan Distribution:**")
            for plan_key, count in plan_dist.items():
                report.append(f"- Plan {plan_key}: {count:,}")
    
    elif industry == "logistics":
        fact_ship = tables.get("fact_shipments")
        fact_del = tables.get("fact_deliveries")
        if fact_ship is not None:
            total_weight = fact_ship["weight_lbs"].sum()
            total_miles = fact_ship["distance_miles"].sum()
            total_cost = fact_ship["shipping_cost"].sum()
            report.append(f"### Logistics Operations")
            report.append(f"- Total Shipments: {len(fact_ship):,}")
            report.append(f"- Total Weight: {total_weight:,.0f} lbs")
            report.append(f"- Total Miles: {total_miles:,.0f}")
            report.append(f"- Total Cost: ${total_cost:,.2f}")
            
            if fact_del is not None:
                on_time = fact_del[fact_del["on_time"] == True]
                on_time_pct = len(on_time) / len(fact_del) * 100 if len(fact_del) > 0 else 0
                report.append(f"- On-Time Delivery Rate: {on_time_pct:.1f}%")
    
    report.append(f"\n## Recommended Analytics\n")
    
    if industry == "retail":
        report.append("### Analysis Ideas")
        report.append("1. **Sales Trends**: Analyze daily/weekly/monthly sales patterns")
        report.append("2. **Store Performance**: Compare revenue across store locations")
        report.append("3. **Customer Segmentation**: Group customers by loyalty tier and purchase behavior")
        report.append("4. **Inventory Optimization**: Identify slow-moving vs. fast-moving products")
        report.append("5. **Seasonality**: Detect seasonal trends by product category")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- Top performing stores by revenue")
        report.append("SELECT s.store_name, s.city, SUM(f.revenue) as total_revenue")
        report.append("FROM fact_sales f")
        report.append("JOIN dim_store s ON f.store_key = s.store_key")
        report.append("GROUP BY s.store_key, s.store_name, s.city")
        report.append("ORDER BY total_revenue DESC")
        report.append("LIMIT 10;")
        report.append("```")
    
    elif industry == "ecommerce":
        report.append("### Analysis Ideas")
        report.append("1. **Conversion Funnel**: Analyze web events from view to purchase")
        report.append("2. **Customer Acquisition**: Track acquisition channels and CAC")
        report.append("3. **Cart Abandonment**: Identify drop-off points in checkout")
        report.append("4. **Product Affinity**: Find products frequently bought together")
        report.append("5. **Customer Lifetime Value**: Calculate LTV by cohort")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- Daily order trends with moving average")
        report.append("SELECT d.full_date,")
        report.append("       COUNT(o.order_id) as orders,")
        report.append("       SUM(o.order_value) as gmv,")
        report.append("       AVG(o.order_value) as aov")
        report.append("FROM fact_orders o")
        report.append("JOIN dim_date d ON o.date_key = d.date_key")
        report.append("GROUP BY d.full_date")
        report.append("ORDER BY d.full_date;")
        report.append("```")
    
    elif industry == "banking":
        report.append("### Analysis Ideas")
        report.append("1. **Fraud Detection**: Identify unusual transaction patterns")
        report.append("2. **Risk Assessment**: Analyze loan default rates by customer segment")
        report.append("3. **Branch Profitability**: Compare transaction volume across branches")
        report.append("4. **Customer Behavior**: Analyze transaction types and frequencies")
        report.append("5. **Portfolio Health**: Monitor account balances and loan performance")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- Customer transaction summary")
        report.append("SELECT c.customer_id,")
        report.append("       COUNT(t.transaction_id) as txn_count,")
        report.append("       SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) as deposits,")
        report.append("       SUM(CASE WHEN t.amount < 0 THEN ABS(t.amount) ELSE 0 END) as withdrawals")
        report.append("FROM fact_transactions t")
        report.append("JOIN dim_account a ON t.account_key = a.account_key")
        report.append("JOIN dim_customer c ON a.customer_key = c.customer_key")
        report.append("GROUP BY c.customer_id;")
        report.append("```")
    
    elif industry == "healthcare":
        report.append("### Analysis Ideas")
        report.append("1. **Utilization Analysis**: Track encounter types and volumes")
        report.append("2. **Cost Analysis**: Analyze charges by diagnosis and procedure")
        report.append("3. **Length of Stay**: Monitor LOS trends and outliers")
        report.append("4. **Provider Productivity**: Compare encounter volumes by doctor")
        report.append("5. **Claims Analysis**: Track claim denial rates and reasons")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- High-cost diagnoses")
        report.append("SELECT dx.diagnosis_description,")
        report.append("       COUNT(e.encounter_id) as encounter_count,")
        report.append("       AVG(e.total_charges) as avg_charges,")
        report.append("       SUM(e.total_charges) as total_charges")
        report.append("FROM fact_encounters e")
        report.append("JOIN dim_diagnosis dx ON e.diagnosis_key = dx.diagnosis_key")
        report.append("GROUP BY dx.diagnosis_key, dx.diagnosis_description")
        report.append("ORDER BY total_charges DESC")
        report.append("LIMIT 10;")
        report.append("```")
    
    elif industry == "saas":
        report.append("### Analysis Ideas")
        report.append("1. **MRR Growth**: Track monthly recurring revenue trends")
        report.append("2. **Churn Analysis**: Identify churn drivers and at-risk customers")
        report.append("3. **Cohort Analysis**: Analyze retention by signup cohort")
        report.append("4. **Usage Patterns**: Correlate feature usage with plan upgrades")
        report.append("5. **Customer Health**: Score accounts based on usage and engagement")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- Monthly MRR trend")
        report.append("SELECT d.year, d.month,")
        report.append("       SUM(s.mrr) as total_mrr,")
        report.append("       COUNT(DISTINCT s.customer_key) as active_customers")
        report.append("FROM fact_subscriptions s")
        report.append("JOIN dim_date d ON s.snapshot_date_key = d.date_key")
        report.append("WHERE s.status = 'Active'")
        report.append("GROUP BY d.year, d.month")
        report.append("ORDER BY d.year, d.month;")
        report.append("```")
    
    elif industry == "logistics":
        report.append("### Analysis Ideas")
        report.append("1. **Delivery Performance**: Track on-time delivery rates")
        report.append("2. **Route Optimization**: Identify cost-efficient routes")
        report.append("3. **Fleet Utilization**: Analyze vehicle usage patterns")
        report.append("4. **Cost Per Mile**: Calculate shipping economics")
        report.append("5. **Warehouse Efficiency**: Monitor throughput by facility")
        
        report.append("\n### Sample SQL Queries")
        report.append("```sql")
        report.append("-- Route performance analysis")
        report.append("SELECT r.origin_city, r.destination_city,")
        report.append("       COUNT(s.shipment_id) as shipment_count,")
        report.append("       AVG(s.shipping_cost) as avg_cost,")
        report.append("       AVG(d.actual_transit_days) as avg_days,")
        report.append("       SUM(CASE WHEN d.on_time THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as on_time_pct")
        report.append("FROM fact_shipments s")
        report.append("JOIN dim_route r ON s.route_key = r.route_key")
        report.append("JOIN fact_deliveries d ON s.shipment_id = d.shipment_id")
        report.append("GROUP BY r.route_key, r.origin_city, r.destination_city")
        report.append("ORDER BY shipment_count DESC;")
        report.append("```")
    
    report.append(f"\n## Data Quality Actions\n")
    
    if config['quality_rate'] > 0:
        report.append("### Recommended Cleaning Steps")
        report.append("1. **Standardize Text Fields**")
        report.append("   - Convert names and addresses to consistent casing")
        report.append("   - Trim leading/trailing whitespace")
        report.append("   - Standardize country names and abbreviations")
        
        report.append("\n2. **Handle Missing Values**")
        report.append("   - Decide on imputation strategy for each field")
        report.append("   - Consider dropping rows with missing critical fields")
        report.append("   - Document assumptions for filled values")
        
        report.append("\n3. **Remove Duplicates**")
        report.append("   - Identify duplicate records in fact tables")
        report.append("   - Determine deduplication logic (keep first, last, or merge)")
        report.append("   - Verify primary key uniqueness")
        
        report.append("\n4. **Validate Referential Integrity**")
        report.append("   - Check for orphaned foreign keys")
        report.append("   - Ensure all fact table keys exist in dimensions")
        report.append("   - Handle or document data quality exceptions")
        
        report.append("\n5. **Fix Data Anomalies**")
        report.append("   - Correct negative quantities where inappropriate")
        report.append("   - Validate calculated fields (e.g., total = qty * price)")
        report.append("   - Investigate and handle outliers")
        
        report.append("\n### Python Cleaning Template")
        report.append("```python")
        report.append("import pandas as pd")
        report.append("")
        report.append("# Load data")
        report.append("df = pd.read_csv('fact_sales.csv')")
        report.append("")
        report.append("# Standardize text")
        report.append("df['customer_name'] = df['customer_name'].str.strip().str.title()")
        report.append("")
        report.append("# Handle nulls")
        report.append("df['phone'].fillna('UNKNOWN', inplace=True)")
        report.append("")
        report.append("# Remove duplicates")
        report.append("df.drop_duplicates(subset=['transaction_id'], inplace=True)")
        report.append("")
        report.append("# Fix negative quantities")
        report.append("df.loc[df['quantity'] < 0, 'quantity'] = df.loc[df['quantity'] < 0, 'quantity'].abs()")
        report.append("```")
    else:
        report.append("### No Cleaning Required")
        report.append("This dataset was generated with **clean** quality level and contains no intentional data quality issues.")
        report.append("However, you may still want to:")
        report.append("- Validate data types and constraints")
        report.append("- Check for unexpected patterns or distributions")
        report.append("- Verify business logic consistency")
    
    report.append(f"\n## Next Steps\n")
    report.append("1. **Load into Database**: Import tables into your preferred database (PostgreSQL, MySQL, SQLite)")
    report.append("2. **Build ETL Pipeline**: Create data transformation workflows")
    report.append("3. **Create Dashboards**: Connect to BI tools (Tableau, Power BI, Metabase)")
    report.append("4. **Practice SQL**: Execute the sample queries and create your own")
    report.append("5. **Data Cleaning**: If quality issues were injected, practice cleaning techniques")
    report.append("6. **ML Experiments**: Use for predictive modeling and analytics projects")
    
    report.append(f"\n---")
    report.append(f"\n*Report generated by Synthetic Data Generator*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
