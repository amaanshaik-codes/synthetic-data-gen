"""
Quick start script for SynthData.
Run this script to generate a sample dataset and explore it.
"""

from synthdata import SyntheticDataGenerator, SynthDataConfig
from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    AnalyticsConfig,
    OutputConfig,
    Industry,
    BusinessModel,
    BusinessScale,
    Difficulty,
    AnalyticsUseCase,
    OutputFormat,
)


def main():
    """Generate a sample dataset."""
    print("=" * 60)
    print("SynthData - Quick Start Demo")
    print("=" * 60)
    
    # Create a configuration
    config = SynthDataConfig(
        name="quickstart_demo",
        description="A quick demo of the synthetic data generator",
        business_context=BusinessContextConfig(
            industry=Industry.ECOMMERCE,
            business_model=BusinessModel.B2C,
            time_span_months=6,
        ),
        business_size=BusinessSizeConfig(
            scale=BusinessScale.STARTUP,
            num_customers=1000,
            daily_transactions=50,
            num_products=100,
        ),
        difficulty=Difficulty.MEDIUM,
        analytics=AnalyticsConfig(
            use_case=AnalyticsUseCase.CHURN_PREDICTION,
            class_imbalance_ratio=0.15,
        ),
        output=OutputConfig(
            format=OutputFormat.CSV,
            directory="./demo_output",
        ),
    )
    
    print(f"\n📊 Generating dataset: {config.name}")
    print(f"   Industry: {config.business_context.industry.value}")
    print(f"   Customers: {config.business_size.num_customers:,}")
    print(f"   Difficulty: {config.difficulty.value}")
    print(f"   Use Case: {config.analytics.use_case.value}")
    
    # Generate the data
    print("\n⏳ Generating data...")
    generator = SyntheticDataGenerator(config)
    tables = generator.generate(show_progress=True)
    
    # Save the data
    print("\n💾 Saving data...")
    output_path = generator.save()
    
    # Display summary
    print("\n✅ Generation Complete!")
    print("-" * 40)
    
    for table_name, df in tables.items():
        print(f"\n📋 {table_name}:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Missing values: {df.isna().sum().sum():,} ({df.isna().mean().mean():.1%})")
    
    print(f"\n📁 Output saved to: {output_path}")
    print(f"🔑 Random seed: {config.reproducibility.seed}")
    
    # Show a sample of the customers table
    print("\n" + "=" * 60)
    print("Sample Data Preview (customers)")
    print("=" * 60)
    
    if "customers" in tables:
        print(tables["customers"].head(5).to_string())
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Explore the generated CSV files in ./demo_output/")
    print("2. Read the metadata in ./demo_output/metadata/")
    print("3. Try the suggested analytics questions")
    print("4. Clean the data and build models!")
    print("\nFor more options, run: synthdata --help")


if __name__ == "__main__":
    main()
