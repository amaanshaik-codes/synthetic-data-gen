"""
Tests for SynthData package.
"""

import pytest
import tempfile
from pathlib import Path

import pandas as pd

from synthdata import SynthDataConfig, SyntheticDataGenerator
from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    Industry,
    BusinessModel,
    BusinessScale,
    Difficulty,
    AnalyticsUseCase,
    OutputFormat,
)


class TestSynthDataConfig:
    """Tests for configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SynthDataConfig()
        assert config.name == "synthetic_dataset"
        assert config.difficulty == Difficulty.MEDIUM
    
    def test_config_with_options(self):
        """Test configuration with custom options."""
        config = SynthDataConfig(
            name="test_dataset",
            business_context=BusinessContextConfig(
                industry=Industry.FINTECH,
                business_model=BusinessModel.B2B,
            ),
            difficulty=Difficulty.HARD,
        )
        assert config.name == "test_dataset"
        assert config.business_context.industry == Industry.FINTECH
        assert config.difficulty == Difficulty.HARD
    
    def test_config_serialization(self):
        """Test config save and load."""
        config = SynthDataConfig(name="test_serialization")
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_file(f.name)
            loaded = SynthDataConfig.from_file(f.name)
        
        assert loaded.name == config.name
        assert loaded.difficulty == config.difficulty
    
    def test_seed_initialization(self):
        """Test seed initialization."""
        config = SynthDataConfig()
        seed = config.initialize_seed()
        assert config.reproducibility.seed == seed
        assert isinstance(seed, int)


class TestGenerators:
    """Tests for data generators."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for fast tests."""
        return SynthDataConfig(
            business_size=BusinessSizeConfig(
                scale=BusinessScale.STARTUP,
                num_customers=100,
                daily_transactions=10,
                num_products=20,
            ),
            business_context=BusinessContextConfig(
                time_span_months=1,
            ),
        )
    
    def test_generator_creation(self, small_config):
        """Test generator can be created."""
        generator = SyntheticDataGenerator(small_config)
        assert generator is not None
    
    def test_generate_customers(self, small_config):
        """Test customer generation."""
        generator = SyntheticDataGenerator(small_config)
        tables = generator.generate(tables=["customers"])
        
        assert "customers" in tables
        assert len(tables["customers"]) == small_config.business_size.num_customers
        assert "customer_id" in tables["customers"].columns
    
    def test_generate_products(self, small_config):
        """Test product generation."""
        generator = SyntheticDataGenerator(small_config)
        tables = generator.generate(tables=["products"])
        
        assert "products" in tables
        assert len(tables["products"]) == small_config.business_size.num_products
    
    def test_generate_all_tables(self, small_config):
        """Test generating all tables."""
        generator = SyntheticDataGenerator(small_config)
        tables = generator.generate()
        
        expected_tables = ["customers", "products", "transactions", "campaigns", "support_tickets"]
        for table in expected_tables:
            assert table in tables
    
    def test_reproducibility(self, small_config):
        """Test that same seed produces same data."""
        small_config.reproducibility.seed = 42
        
        gen1 = SyntheticDataGenerator(small_config)
        tables1 = gen1.generate(tables=["customers"])
        
        small_config.reproducibility.seed = 42
        gen2 = SyntheticDataGenerator(small_config)
        tables2 = gen2.generate(tables=["customers"])
        
        pd.testing.assert_frame_equal(tables1["customers"], tables2["customers"])


class TestDataQuality:
    """Tests for data quality injection."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator with quality issues."""
        config = SynthDataConfig(
            business_size=BusinessSizeConfig(
                num_customers=500,
                daily_transactions=50,
                num_products=50,
            ),
            business_context=BusinessContextConfig(
                time_span_months=1,
            ),
            difficulty=Difficulty.HARD,
        )
        return SyntheticDataGenerator(config)
    
    def test_missing_values_injected(self, generator):
        """Test that missing values are injected."""
        tables = generator.generate(tables=["customers"])
        customers = tables["customers"]
        
        # Should have some missing values
        total_missing = customers.isna().sum().sum()
        assert total_missing > 0
    
    def test_difficulty_affects_quality(self):
        """Test that difficulty level affects data quality."""
        easy_config = SynthDataConfig(
            business_size=BusinessSizeConfig(num_customers=200),
            business_context=BusinessContextConfig(time_span_months=1),
            difficulty=Difficulty.EASY,
        )
        hard_config = SynthDataConfig(
            business_size=BusinessSizeConfig(num_customers=200),
            business_context=BusinessContextConfig(time_span_months=1),
            difficulty=Difficulty.HARD,
        )
        
        easy_gen = SyntheticDataGenerator(easy_config)
        hard_gen = SyntheticDataGenerator(hard_config)
        
        easy_tables = easy_gen.generate(tables=["customers"])
        hard_tables = hard_gen.generate(tables=["customers"])
        
        easy_missing = easy_tables["customers"].isna().mean().mean()
        hard_missing = hard_tables["customers"].isna().mean().mean()
        
        # Hard should have more missing than easy
        assert hard_missing > easy_missing


class TestOutput:
    """Tests for output handling."""
    
    @pytest.fixture
    def generator(self):
        """Create a small generator."""
        config = SynthDataConfig(
            business_size=BusinessSizeConfig(
                num_customers=50,
                daily_transactions=10,
                num_products=10,
            ),
            business_context=BusinessContextConfig(
                time_span_months=1,
            ),
        )
        return SyntheticDataGenerator(config)
    
    def test_save_csv(self, generator):
        """Test saving as CSV."""
        tables = generator.generate(tables=["customers"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.config.output.format = OutputFormat.CSV
            output_path = generator.save(output_dir=tmpdir)
            
            csv_file = output_path / "customers.csv"
            assert csv_file.exists()
            
            loaded = pd.read_csv(csv_file)
            assert len(loaded) == len(tables["customers"])
    
    def test_save_parquet(self, generator):
        """Test saving as Parquet."""
        tables = generator.generate(tables=["customers"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.config.output.format = OutputFormat.PARQUET
            output_path = generator.save(output_dir=tmpdir)
            
            parquet_file = output_path / "customers.parquet"
            assert parquet_file.exists()
    
    def test_metadata_saved(self, generator):
        """Test that metadata files are saved."""
        generator.generate(tables=["customers"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generator.save(output_dir=tmpdir, include_metadata=True)
            
            metadata_dir = output_path / "metadata"
            assert metadata_dir.exists()
            assert (metadata_dir / "data_dictionary.json").exists()


class TestAnalytics:
    """Tests for analytics use cases."""
    
    def test_churn_prediction_features(self):
        """Test churn prediction adds relevant features."""
        config = SynthDataConfig(
            business_size=BusinessSizeConfig(num_customers=100),
            business_context=BusinessContextConfig(time_span_months=1),
            analytics=AnalyticsConfig(
                use_case=AnalyticsUseCase.CHURN_PREDICTION,
                class_imbalance_ratio=0.2,
            ),
        )
        
        generator = SyntheticDataGenerator(config)
        tables = generator.generate(tables=["customers"])
        
        customers = tables["customers"]
        assert "churned" in customers.columns or "is_active" in customers.columns
    
    def test_fraud_detection_features(self):
        """Test fraud detection adds fraud labels."""
        config = SynthDataConfig(
            business_size=BusinessSizeConfig(
                num_customers=100,
                daily_transactions=20,
            ),
            business_context=BusinessContextConfig(
                industry=Industry.FINTECH,
                time_span_months=1,
            ),
            analytics=AnalyticsConfig(
                use_case=AnalyticsUseCase.FRAUD_DETECTION,
            ),
        )
        
        generator = SyntheticDataGenerator(config)
        tables = generator.generate(tables=["customers", "transactions"])
        
        transactions = tables["transactions"]
        assert "is_fraud" in transactions.columns


from synthdata.config import AnalyticsConfig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
