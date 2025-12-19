"""
Output handling for synthetic data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from synthdata.config import OutputConfig, OutputFormat


class OutputHandler:
    """Handle saving generated data to various formats."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
    
    def save(
        self,
        tables: Dict[str, pd.DataFrame],
        metadata: Dict[str, Any],
        output_dir: str,
        include_metadata: bool = True,
    ) -> Path:
        """
        Save generated data to files.
        
        Args:
            tables: Dictionary of table names to DataFrames
            metadata: Metadata dictionary
            output_dir: Output directory path
            include_metadata: Whether to save metadata files
        
        Returns:
            Path to the output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each table
        for table_name, df in tables.items():
            self._save_table(df, table_name, output_path)
        
        # Save metadata
        if include_metadata:
            self._save_metadata(metadata, output_path)
        
        return output_path
    
    def _save_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        output_path: Path,
    ) -> Path:
        """Save a single table to file."""
        file_path = self._get_file_path(output_path, table_name)
        
        if self.config.format == OutputFormat.CSV:
            df.to_csv(file_path, index=False)
        
        elif self.config.format == OutputFormat.PARQUET:
            df.to_parquet(
                file_path,
                index=False,
                compression=self.config.compression or "snappy",
            )
        
        elif self.config.format == OutputFormat.JSON:
            df.to_json(
                file_path,
                orient="records",
                date_format="iso",
                indent=2,
            )
        
        elif self.config.format == OutputFormat.SQL:
            self._save_as_sql(df, table_name, file_path)
        
        return file_path
    
    def _get_file_path(self, output_path: Path, table_name: str) -> Path:
        """Get the file path for a table."""
        extensions = {
            OutputFormat.CSV: ".csv",
            OutputFormat.PARQUET: ".parquet",
            OutputFormat.JSON: ".json",
            OutputFormat.SQL: ".sql",
        }
        
        ext = extensions.get(self.config.format, ".csv")
        return output_path / f"{table_name}{ext}"
    
    def _save_as_sql(
        self,
        df: pd.DataFrame,
        table_name: str,
        file_path: Path,
    ):
        """Save DataFrame as SQL INSERT statements."""
        lines = []
        
        # Create table statement
        columns = []
        for col, dtype in df.dtypes.items():
            sql_type = self._pandas_to_sql_type(dtype)
            columns.append(f"    {col} {sql_type}")
        
        lines.append(f"CREATE TABLE {table_name} (")
        lines.append(",\n".join(columns))
        lines.append(");")
        lines.append("")
        
        # Insert statements
        for _, row in df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("NULL")
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped = val.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(val, (datetime, pd.Timestamp)):
                    values.append(f"'{val}'")
                elif isinstance(val, bool):
                    values.append("TRUE" if val else "FALSE")
                else:
                    values.append(str(val))
            
            values_str = ", ".join(values)
            lines.append(f"INSERT INTO {table_name} VALUES ({values_str});")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    
    def _pandas_to_sql_type(self, dtype) -> str:
        """Convert pandas dtype to SQL type."""
        dtype_str = str(dtype)
        
        if "int" in dtype_str:
            return "INTEGER"
        elif "float" in dtype_str:
            return "REAL"
        elif "bool" in dtype_str:
            return "BOOLEAN"
        elif "datetime" in dtype_str:
            return "TIMESTAMP"
        else:
            return "TEXT"
    
    def _save_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: Path,
    ):
        """Save metadata files."""
        metadata_dir = output_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Save main config
        config_path = metadata_dir / f"config.{self.config.format.value}"
        if self.config.format in [OutputFormat.JSON, OutputFormat.SQL]:
            with open(metadata_dir / "config.json", 'w') as f:
                json.dump(metadata.get("config", {}), f, indent=2, default=str)
        else:
            with open(metadata_dir / "config.yaml", 'w') as f:
                yaml.dump(metadata.get("config", {}), f, default_flow_style=False)
        
        # Save data dictionary
        if self.config.include_data_dictionary:
            with open(metadata_dir / "data_dictionary.json", 'w') as f:
                json.dump(metadata.get("data_dictionary", {}), f, indent=2)
        
        # Save quality report
        if self.config.include_quality_report:
            with open(metadata_dir / "quality_report.json", 'w') as f:
                json.dump(metadata.get("quality_issues", {}), f, indent=2)
        
        # Save suggested questions
        if self.config.include_suggested_questions:
            self._save_suggested_questions(
                metadata.get("suggested_questions", []),
                metadata_dir,
            )
        
        # Save relationships
        with open(metadata_dir / "relationships.json", 'w') as f:
            json.dump(metadata.get("relationships", []), f, indent=2)
        
        # Save table summaries
        with open(metadata_dir / "table_info.json", 'w') as f:
            json.dump(metadata.get("tables", {}), f, indent=2)
    
    def _save_suggested_questions(
        self,
        questions: List[str],
        metadata_dir: Path,
    ):
        """Save suggested analytics questions as markdown."""
        lines = [
            "# Suggested Analytics Questions",
            "",
            "Use these questions to guide your exploration and analysis of this dataset.",
            "",
        ]
        
        for i, question in enumerate(questions, 1):
            lines.append(f"{i}. {question}")
        
        lines.extend([
            "",
            "## Data Quality Challenges",
            "",
            "As you explore the data, you may encounter:",
            "",
            "- Missing values in various columns",
            "- Duplicate records",
            "- Inconsistent formats (dates, currencies, etc.)",
            "- Outliers and anomalies",
            "- Data quality issues that require cleaning",
            "",
            "## Getting Started",
            "",
            "1. Load the data files",
            "2. Explore the data dictionary in `metadata/data_dictionary.json`",
            "3. Check the quality report in `metadata/quality_report.json`",
            "4. Start with exploratory data analysis (EDA)",
            "5. Address data quality issues",
            "6. Proceed with your analysis or modeling",
        ])
        
        with open(metadata_dir / "suggested_questions.md", 'w') as f:
            f.write("\n".join(lines))
