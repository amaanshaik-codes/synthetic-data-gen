"""
Configuration schema and validation for SynthData.
Defines all configurable parameters for synthetic data generation.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Industry(str, Enum):
    """Supported industry types."""
    RETAIL = "retail"
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    SAAS = "saas"
    MANUFACTURING = "manufacturing"
    MARKETING = "marketing"
    LOGISTICS = "logistics"


class BusinessModel(str, Enum):
    """Supported business models."""
    B2B = "b2b"
    B2C = "b2c"
    D2C = "d2c"
    MARKETPLACE = "marketplace"
    SUBSCRIPTION = "subscription"


class Geography(str, Enum):
    """Geographic scope options."""
    SINGLE_COUNTRY = "single-country"
    MULTI_COUNTRY = "multi-country"
    GLOBAL = "global"


class BusinessScale(str, Enum):
    """Business size scale."""
    STARTUP = "startup"
    SME = "sme"
    ENTERPRISE = "enterprise"


class RevenueScale(str, Enum):
    """Revenue scale options."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Difficulty(str, Enum):
    """Data cleaning difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHAOTIC = "chaotic"


class AnalyticsUseCase(str, Enum):
    """Supported analytics use cases."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    TIME_SERIES = "time-series"
    CHURN_PREDICTION = "churn-prediction"
    FRAUD_DETECTION = "fraud-detection"
    RECOMMENDATION = "recommendation"
    SEGMENTATION = "segmentation"
    COHORT_ANALYSIS = "cohort-analysis"
    AB_TESTING = "ab-testing"


class OutputFormat(str, Enum):
    """Supported output formats."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    SQL = "sql"


class RelationshipType(str, Enum):
    """Table relationship types."""
    ONE_TO_ONE = "one-to-one"
    ONE_TO_MANY = "one-to-many"
    MANY_TO_MANY = "many-to-many"


class ColumnType(str, Enum):
    """Supported column types."""
    ID = "id"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    CATEGORY = "category"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    COMPANY = "company"
    URL = "url"
    IP_ADDRESS = "ip_address"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    TEXT = "text"
    JSON = "json"


# ============================================================================
# Configuration Models
# ============================================================================

class BusinessContextConfig(BaseModel):
    """Business context configuration."""
    industry: Industry = Industry.ECOMMERCE
    business_model: BusinessModel = BusinessModel.B2C
    geography: Geography = Geography.SINGLE_COUNTRY
    time_span_months: int = Field(default=12, ge=1, le=120)
    start_date: Optional[datetime] = None
    countries: List[str] = Field(default_factory=lambda: ["US"])
    
    @field_validator('countries', mode='before')
    @classmethod
    def validate_countries(cls, v):
        if isinstance(v, str):
            return [v]
        return v or ["US"]


class BusinessSizeConfig(BaseModel):
    """Business size configuration."""
    scale: BusinessScale = BusinessScale.SME
    num_customers: int = Field(default=10000, ge=100, le=10000000)
    daily_transactions: int = Field(default=500, ge=10, le=1000000)
    revenue_scale: RevenueScale = RevenueScale.MEDIUM
    num_products: int = Field(default=500, ge=10, le=100000)
    num_employees: int = Field(default=50, ge=1, le=100000)

    @model_validator(mode='after')
    def set_defaults_by_scale(self):
        """Set sensible defaults based on business scale."""
        scale_defaults = {
            BusinessScale.STARTUP: {
                "num_customers": 1000,
                "daily_transactions": 50,
                "num_products": 50,
                "num_employees": 10,
            },
            BusinessScale.SME: {
                "num_customers": 10000,
                "daily_transactions": 500,
                "num_products": 500,
                "num_employees": 100,
            },
            BusinessScale.ENTERPRISE: {
                "num_customers": 500000,
                "daily_transactions": 10000,
                "num_products": 5000,
                "num_employees": 5000,
            },
        }
        return self


class ColumnConfig(BaseModel):
    """Column configuration for a table."""
    name: str
    type: ColumnType
    nullable: bool = True
    unique: bool = False
    missing_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    categories: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    foreign_key: Optional[str] = None  # Format: "table.column"


class RelationshipConfig(BaseModel):
    """Relationship configuration between tables."""
    parent_table: str
    parent_column: str = "id"
    child_table: str
    child_column: str
    relationship_type: RelationshipType = RelationshipType.ONE_TO_MANY
    min_cardinality: int = Field(default=0, ge=0)
    max_cardinality: int = Field(default=10, ge=1)
    nullable: bool = True


class TableConfig(BaseModel):
    """Configuration for a single table."""
    name: str
    enabled: bool = True
    num_rows: Optional[int] = None  # If None, derived from relationships
    columns: List[ColumnConfig] = Field(default_factory=list)
    primary_key: str = "id"
    timestamps: bool = True  # Add created_at, updated_at


class MissingValuesConfig(BaseModel):
    """Missing values configuration."""
    global_rate: float = Field(default=0.05, ge=0.0, le=0.9)
    per_column: Dict[str, float] = Field(default_factory=dict)
    patterns: List[str] = Field(
        default_factory=lambda: ["MCAR", "MAR", "MNAR"]
    )  # Missing completely at random, Missing at random, Missing not at random
    null_representations: List[str] = Field(
        default_factory=lambda: ["", "NULL", "N/A", "None", "nan", "-", "?", "NA", "NaN"]
    )


class DuplicatesConfig(BaseModel):
    """Duplicate records configuration."""
    rate: float = Field(default=0.01, ge=0.0, le=0.5)
    exact_duplicates: bool = True
    near_duplicates: bool = False  # Slight variations
    near_duplicate_columns: List[str] = Field(default_factory=list)


class InconsistenciesConfig(BaseModel):
    """Data inconsistencies configuration."""
    date_formats: bool = True
    date_format_variations: List[str] = Field(
        default_factory=lambda: [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d-%m-%Y", "%B %d, %Y", "%d %b %Y", "%Y%m%d"
        ]
    )
    currency_formats: bool = True
    currency_variations: List[str] = Field(
        default_factory=lambda: ["$1,234.56", "1234.56", "$1234.56", "1,234.56 USD", "USD 1234.56"]
    )
    category_typos: bool = True
    typo_rate: float = Field(default=0.02, ge=0.0, le=0.3)
    case_inconsistencies: bool = True  # Mixed case in categorical values
    whitespace_issues: bool = True  # Leading/trailing spaces
    encoding_issues: bool = False  # Unicode/encoding problems


class OutliersConfig(BaseModel):
    """Outliers and anomalies configuration."""
    rate: float = Field(default=0.02, ge=0.0, le=0.2)
    magnitude: float = Field(default=3.0, ge=1.5, le=10.0)  # Standard deviations
    outlier_types: List[str] = Field(
        default_factory=lambda: ["extreme_high", "extreme_low", "impossible_values"]
    )
    per_column: Dict[str, float] = Field(default_factory=dict)


class NoiseConfig(BaseModel):
    """Noise and label noise configuration."""
    label_noise_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    feature_noise_rate: float = Field(default=0.0, ge=0.0, le=0.3)
    noise_type: str = Field(default="gaussian")  # gaussian, uniform, salt_pepper


class DataDriftConfig(BaseModel):
    """Data drift configuration."""
    enabled: bool = False
    drift_columns: List[str] = Field(default_factory=list)
    drift_start_percentage: float = Field(default=0.7, ge=0.0, le=1.0)  # When drift starts
    drift_magnitude: float = Field(default=0.2, ge=0.0, le=1.0)
    drift_type: str = Field(default="gradual")  # gradual, sudden, seasonal


class DataQualityConfig(BaseModel):
    """Overall data quality configuration."""
    missing_values: MissingValuesConfig = Field(default_factory=MissingValuesConfig)
    duplicates: DuplicatesConfig = Field(default_factory=DuplicatesConfig)
    inconsistencies: InconsistenciesConfig = Field(default_factory=InconsistenciesConfig)
    outliers: OutliersConfig = Field(default_factory=OutliersConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    data_drift: DataDriftConfig = Field(default_factory=DataDriftConfig)


class AnalyticsConfig(BaseModel):
    """Analytics and modeling configuration."""
    use_case: AnalyticsUseCase = AnalyticsUseCase.DESCRIPTIVE
    target_variable: Optional[str] = None
    target_column: Optional[str] = None
    signal_to_noise_ratio: float = Field(default=0.7, ge=0.1, le=1.0)
    feature_leakage: bool = False
    leakage_columns: List[str] = Field(default_factory=list)
    train_test_split: float = Field(default=0.8, ge=0.5, le=0.95)
    validation_split: Optional[float] = Field(default=None, ge=0.0, le=0.3)
    class_imbalance_ratio: Optional[float] = Field(default=None, ge=0.01, le=0.5)
    time_based_split: bool = False
    stratified_split: bool = True


class TransformationConfig(BaseModel):
    """Data transformation and preprocessing options."""
    output_mode: str = Field(default="raw")  # raw, semi-processed, feature-engineered
    scaling: Optional[str] = None  # standard, minmax, robust
    encoding: Optional[str] = None  # onehot, label, target
    aggregations: List[str] = Field(default_factory=list)  # sum, mean, count, etc.
    date_features: bool = False  # Extract day, month, year, etc.
    text_features: bool = False  # Basic text feature extraction


class OutputConfig(BaseModel):
    """Output configuration."""
    format: OutputFormat = OutputFormat.CSV
    directory: str = "./output"
    include_metadata: bool = True
    include_data_dictionary: bool = True
    include_quality_report: bool = True
    include_suggested_questions: bool = True
    compression: Optional[str] = None  # gzip, snappy, etc.
    single_file: bool = False  # Combine all tables into one file
    

class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""
    seed: Optional[int] = None
    save_config: bool = True
    config_format: str = Field(default="yaml")  # yaml, json
    version: str = "1.0.0"


class SynthDataConfig(BaseModel):
    """Main configuration for synthetic data generation."""
    name: str = Field(default="synthetic_dataset")
    description: str = Field(default="")
    business_context: BusinessContextConfig = Field(default_factory=BusinessContextConfig)
    business_size: BusinessSizeConfig = Field(default_factory=BusinessSizeConfig)
    tables: List[TableConfig] = Field(default_factory=list)
    relationships: List[RelationshipConfig] = Field(default_factory=list)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    difficulty: Difficulty = Difficulty.MEDIUM
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    transformation: TransformationConfig = Field(default_factory=TransformationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    
    @model_validator(mode='after')
    def apply_difficulty_presets(self):
        """Apply difficulty presets to data quality settings."""
        if not self.tables:  # Only apply if using default tables
            self._apply_difficulty_to_quality()
        return self
    
    def _apply_difficulty_to_quality(self):
        """Apply difficulty level presets to data quality configuration."""
        presets = {
            Difficulty.EASY: {
                "missing_rate": 0.02,
                "duplicate_rate": 0.005,
                "outlier_rate": 0.01,
                "inconsistencies": False,
                "noise": 0.0,
            },
            Difficulty.MEDIUM: {
                "missing_rate": 0.08,
                "duplicate_rate": 0.02,
                "outlier_rate": 0.03,
                "inconsistencies": True,
                "noise": 0.02,
            },
            Difficulty.HARD: {
                "missing_rate": 0.15,
                "duplicate_rate": 0.05,
                "outlier_rate": 0.05,
                "inconsistencies": True,
                "noise": 0.05,
            },
            Difficulty.CHAOTIC: {
                "missing_rate": 0.25,
                "duplicate_rate": 0.10,
                "outlier_rate": 0.08,
                "inconsistencies": True,
                "noise": 0.10,
            },
        }
        
        preset = presets.get(self.difficulty, presets[Difficulty.MEDIUM])
        
        self.data_quality.missing_values.global_rate = preset["missing_rate"]
        self.data_quality.duplicates.rate = preset["duplicate_rate"]
        self.data_quality.outliers.rate = preset["outlier_rate"]
        self.data_quality.inconsistencies.date_formats = preset["inconsistencies"]
        self.data_quality.inconsistencies.category_typos = preset["inconsistencies"]
        self.data_quality.noise.label_noise_rate = preset["noise"]
    
    def initialize_seed(self) -> int:
        """Initialize and return the random seed."""
        if self.reproducibility.seed is None:
            self.reproducibility.seed = random.randint(0, 2**32 - 1)
        return self.reproducibility.seed
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SynthDataConfig":
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.model_validate(data)
    
    def to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML or JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump(mode='json')
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif path.suffix == '.json':
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(mode='json')


# ============================================================================
# Preset Configurations
# ============================================================================

def get_preset_config(preset_name: str) -> SynthDataConfig:
    """Get a preset configuration by name."""
    presets = {
        "ecommerce-basic": SynthDataConfig(
            name="ecommerce_basic",
            description="Basic ecommerce dataset for beginners",
            business_context=BusinessContextConfig(
                industry=Industry.ECOMMERCE,
                business_model=BusinessModel.B2C,
            ),
            difficulty=Difficulty.EASY,
        ),
        "fintech-fraud": SynthDataConfig(
            name="fintech_fraud_detection",
            description="Financial transactions with fraud labels",
            business_context=BusinessContextConfig(
                industry=Industry.FINTECH,
                business_model=BusinessModel.B2C,
            ),
            analytics=AnalyticsConfig(
                use_case=AnalyticsUseCase.FRAUD_DETECTION,
                target_variable="is_fraud",
                class_imbalance_ratio=0.02,
            ),
            difficulty=Difficulty.HARD,
        ),
        "saas-churn": SynthDataConfig(
            name="saas_churn_prediction",
            description="SaaS subscription data for churn prediction",
            business_context=BusinessContextConfig(
                industry=Industry.SAAS,
                business_model=BusinessModel.SUBSCRIPTION,
            ),
            analytics=AnalyticsConfig(
                use_case=AnalyticsUseCase.CHURN_PREDICTION,
                target_variable="churned",
                class_imbalance_ratio=0.15,
            ),
            difficulty=Difficulty.MEDIUM,
        ),
        "retail-timeseries": SynthDataConfig(
            name="retail_timeseries",
            description="Retail sales data for time series forecasting",
            business_context=BusinessContextConfig(
                industry=Industry.RETAIL,
                business_model=BusinessModel.B2C,
                time_span_months=36,
            ),
            analytics=AnalyticsConfig(
                use_case=AnalyticsUseCase.TIME_SERIES,
                time_based_split=True,
            ),
            difficulty=Difficulty.MEDIUM,
        ),
        "healthcare-messy": SynthDataConfig(
            name="healthcare_messy",
            description="Messy healthcare data with many quality issues",
            business_context=BusinessContextConfig(
                industry=Industry.HEALTHCARE,
                business_model=BusinessModel.B2B,
            ),
            difficulty=Difficulty.CHAOTIC,
        ),
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return presets[preset_name]


def list_presets() -> List[Dict[str, str]]:
    """List all available presets."""
    return [
        {"name": "ecommerce-basic", "description": "Basic ecommerce dataset for beginners"},
        {"name": "fintech-fraud", "description": "Financial transactions with fraud labels"},
        {"name": "saas-churn", "description": "SaaS subscription data for churn prediction"},
        {"name": "retail-timeseries", "description": "Retail sales data for time series forecasting"},
        {"name": "healthcare-messy", "description": "Messy healthcare data with many quality issues"},
    ]
