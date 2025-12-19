"""
Command Line Interface for SynthData.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from synthdata import __version__
from synthdata.config import (
    SynthDataConfig,
    BusinessContextConfig,
    BusinessSizeConfig,
    AnalyticsConfig,
    OutputConfig,
    Industry,
    BusinessModel,
    Geography,
    BusinessScale,
    RevenueScale,
    Difficulty,
    AnalyticsUseCase,
    OutputFormat,
    get_preset_config,
    list_presets,
)
from synthdata.generator import SyntheticDataGenerator

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="synthdata")
def main():
    """
    SynthData - Advanced Synthetic Data Generator for Analytics and Data Science Practice
    
    Generate realistic, messy, business-relevant datasets for practicing EDA,
    data cleaning, feature engineering, modeling, and storytelling.
    """
    pass


@main.command()
@click.option(
    "--industry", "-i",
    type=click.Choice([e.value for e in Industry]),
    default="ecommerce",
    help="Industry type for the synthetic data"
)
@click.option(
    "--business-model", "-b",
    type=click.Choice([e.value for e in BusinessModel]),
    default="b2c",
    help="Business model"
)
@click.option(
    "--geography", "-g",
    type=click.Choice([e.value for e in Geography]),
    default="single-country",
    help="Geographic scope"
)
@click.option(
    "--time-span", "-t",
    type=int,
    default=12,
    help="Time span in months"
)
@click.option(
    "--size", "-s",
    type=click.Choice([e.value for e in BusinessScale]),
    default="sme",
    help="Business size scale"
)
@click.option(
    "--customers", "-c",
    type=int,
    default=None,
    help="Number of customers (overrides size default)"
)
@click.option(
    "--daily-transactions",
    type=int,
    default=None,
    help="Transactions per day (overrides size default)"
)
@click.option(
    "--products",
    type=int,
    default=None,
    help="Number of products"
)
@click.option(
    "--difficulty", "-d",
    type=click.Choice([e.value for e in Difficulty]),
    default="medium",
    help="Data cleaning difficulty level"
)
@click.option(
    "--use-case", "-u",
    type=click.Choice([e.value for e in AnalyticsUseCase]),
    default="descriptive",
    help="Analytics use case"
)
@click.option(
    "--output-format", "-f",
    type=click.Choice([e.value for e in OutputFormat]),
    default="csv",
    help="Output file format"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="./output",
    help="Output directory"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Load configuration from YAML/JSON file"
)
@click.option(
    "--save-config",
    type=click.Path(),
    default=None,
    help="Save configuration to file"
)
@click.option(
    "--preset", "-p",
    type=str,
    default=None,
    help="Use a preset configuration (use 'synthdata presets' to list)"
)
@click.option(
    "--name", "-n",
    type=str,
    default="synthetic_dataset",
    help="Name for the dataset"
)
@click.option(
    "--tables",
    type=str,
    default=None,
    help="Comma-separated list of tables to generate"
)
@click.option(
    "--no-metadata",
    is_flag=True,
    default=False,
    help="Skip generating metadata files"
)
def generate(
    industry: str,
    business_model: str,
    geography: str,
    time_span: int,
    size: str,
    customers: Optional[int],
    daily_transactions: Optional[int],
    products: Optional[int],
    difficulty: str,
    use_case: str,
    output_format: str,
    output_dir: str,
    seed: Optional[int],
    config: Optional[str],
    save_config: Optional[str],
    preset: Optional[str],
    name: str,
    tables: Optional[str],
    no_metadata: bool,
):
    """Generate synthetic data with the specified configuration."""
    console.print(Panel.fit(
        "[bold blue]SynthData[/bold blue] - Synthetic Data Generator",
        subtitle=f"v{__version__}"
    ))
    
    try:
        # Load configuration
        if config:
            console.print(f"[dim]Loading configuration from {config}...[/dim]")
            synth_config = SynthDataConfig.from_file(config)
        elif preset:
            console.print(f"[dim]Using preset: {preset}[/dim]")
            synth_config = get_preset_config(preset)
        else:
            # Build configuration from CLI options
            synth_config = _build_config_from_options(
                industry=industry,
                business_model=business_model,
                geography=geography,
                time_span=time_span,
                size=size,
                customers=customers,
                daily_transactions=daily_transactions,
                products=products,
                difficulty=difficulty,
                use_case=use_case,
                output_format=output_format,
                output_dir=output_dir,
                seed=seed,
                name=name,
            )
        
        # Override output settings if specified
        if output_dir != "./output":
            synth_config.output.directory = output_dir
        if output_format:
            synth_config.output.format = OutputFormat(output_format)
        synth_config.output.include_metadata = not no_metadata
        
        # Save config if requested
        if save_config:
            synth_config.to_file(save_config)
            console.print(f"[green]✓[/green] Configuration saved to {save_config}")
        
        # Display configuration summary
        _display_config_summary(synth_config)
        
        # Parse tables to generate
        tables_to_generate = None
        if tables:
            tables_to_generate = [t.strip() for t in tables.split(",")]
        
        # Generate data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating synthetic data...", total=None)
            
            generator = SyntheticDataGenerator(synth_config)
            generated_tables = generator.generate(
                tables=tables_to_generate,
                show_progress=False,
            )
            
            progress.update(task, description="Saving data...")
            output_path = generator.save(include_metadata=not no_metadata)
        
        # Display summary
        _display_generation_summary(generated_tables, output_path, synth_config)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise click.Abort()


@main.command()
def presets():
    """List available preset configurations."""
    console.print(Panel.fit(
        "[bold blue]Available Presets[/bold blue]",
    ))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    
    for preset in list_presets():
        table.add_row(preset["name"], preset["description"])
    
    console.print(table)
    console.print("\n[dim]Use with: synthdata generate --preset <name>[/dim]")


@main.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str):
    """Validate a configuration file."""
    console.print(f"[dim]Validating {config_file}...[/dim]")
    
    try:
        config = SynthDataConfig.from_file(config_file)
        console.print("[green]✓[/green] Configuration is valid!")
        _display_config_summary(config)
    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {str(e)}")
        raise click.Abort()


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="synthdata_config.yaml",
    help="Output file path"
)
def init(output: str):
    """Create a sample configuration file."""
    config = SynthDataConfig(
        name="my_synthetic_dataset",
        description="A synthetic dataset for analytics practice",
    )
    
    config.to_file(output)
    console.print(f"[green]✓[/green] Created sample configuration at {output}")
    console.print("[dim]Edit the file and run: synthdata generate --config <file>[/dim]")


@main.command()
def interactive():
    """Interactive mode for generating data."""
    console.print(Panel.fit(
        "[bold blue]SynthData Interactive Mode[/bold blue]",
        subtitle="Answer the prompts to configure your dataset"
    ))
    
    # Industry
    console.print("\n[bold]1. Industry Type[/bold]")
    for i, ind in enumerate(Industry, 1):
        console.print(f"   {i}. {ind.value}")
    industry_idx = click.prompt("Select industry", type=int, default=2)
    industry = list(Industry)[industry_idx - 1]
    
    # Business Model
    console.print("\n[bold]2. Business Model[/bold]")
    for i, bm in enumerate(BusinessModel, 1):
        console.print(f"   {i}. {bm.value}")
    bm_idx = click.prompt("Select business model", type=int, default=2)
    business_model = list(BusinessModel)[bm_idx - 1]
    
    # Size
    console.print("\n[bold]3. Business Size[/bold]")
    for i, s in enumerate(BusinessScale, 1):
        console.print(f"   {i}. {s.value}")
    size_idx = click.prompt("Select size", type=int, default=2)
    size = list(BusinessScale)[size_idx - 1]
    
    # Difficulty
    console.print("\n[bold]4. Difficulty Level[/bold]")
    for i, d in enumerate(Difficulty, 1):
        console.print(f"   {i}. {d.value}")
    diff_idx = click.prompt("Select difficulty", type=int, default=2)
    difficulty = list(Difficulty)[diff_idx - 1]
    
    # Use Case
    console.print("\n[bold]5. Analytics Use Case[/bold]")
    for i, uc in enumerate(AnalyticsUseCase, 1):
        console.print(f"   {i}. {uc.value}")
    uc_idx = click.prompt("Select use case", type=int, default=1)
    use_case = list(AnalyticsUseCase)[uc_idx - 1]
    
    # Output
    output_dir = click.prompt("\n[bold]6. Output directory[/bold]", default="./output")
    
    console.print("\n[bold]7. Output Format[/bold]")
    for i, f in enumerate(OutputFormat, 1):
        console.print(f"   {i}. {f.value}")
    fmt_idx = click.prompt("Select format", type=int, default=1)
    output_format = list(OutputFormat)[fmt_idx - 1]
    
    # Build config
    config = SynthDataConfig(
        name="interactive_dataset",
        business_context=BusinessContextConfig(
            industry=industry,
            business_model=business_model,
        ),
        business_size=BusinessSizeConfig(scale=size),
        difficulty=difficulty,
        analytics=AnalyticsConfig(use_case=use_case),
        output=OutputConfig(
            format=output_format,
            directory=output_dir,
        ),
    )
    
    # Confirm
    console.print("\n")
    _display_config_summary(config)
    
    if click.confirm("\nGenerate with these settings?", default=True):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating...", total=None)
            
            generator = SyntheticDataGenerator(config)
            tables = generator.generate(show_progress=False)
            output_path = generator.save()
        
        _display_generation_summary(tables, output_path, config)


def _build_config_from_options(
    industry: str,
    business_model: str,
    geography: str,
    time_span: int,
    size: str,
    customers: Optional[int],
    daily_transactions: Optional[int],
    products: Optional[int],
    difficulty: str,
    use_case: str,
    output_format: str,
    output_dir: str,
    seed: Optional[int],
    name: str,
) -> SynthDataConfig:
    """Build configuration from CLI options."""
    business_context = BusinessContextConfig(
        industry=Industry(industry),
        business_model=BusinessModel(business_model),
        geography=Geography(geography),
        time_span_months=time_span,
    )
    
    business_size = BusinessSizeConfig(scale=BusinessScale(size))
    
    if customers is not None:
        business_size.num_customers = customers
    if daily_transactions is not None:
        business_size.daily_transactions = daily_transactions
    if products is not None:
        business_size.num_products = products
    
    analytics = AnalyticsConfig(use_case=AnalyticsUseCase(use_case))
    
    output = OutputConfig(
        format=OutputFormat(output_format),
        directory=output_dir,
    )
    
    from synthdata.config import ReproducibilityConfig
    reproducibility = ReproducibilityConfig(seed=seed)
    
    return SynthDataConfig(
        name=name,
        business_context=business_context,
        business_size=business_size,
        difficulty=Difficulty(difficulty),
        analytics=analytics,
        output=output,
        reproducibility=reproducibility,
    )


def _display_config_summary(config: SynthDataConfig):
    """Display configuration summary."""
    tree = Tree("[bold]Configuration Summary[/bold]")
    
    # Business context
    context = tree.add("[cyan]Business Context[/cyan]")
    context.add(f"Industry: {config.business_context.industry.value}")
    context.add(f"Model: {config.business_context.business_model.value}")
    context.add(f"Geography: {config.business_context.geography.value}")
    context.add(f"Time Span: {config.business_context.time_span_months} months")
    
    # Business size
    size = tree.add("[cyan]Business Size[/cyan]")
    size.add(f"Scale: {config.business_size.scale.value}")
    size.add(f"Customers: {config.business_size.num_customers:,}")
    size.add(f"Daily Transactions: {config.business_size.daily_transactions:,}")
    size.add(f"Products: {config.business_size.num_products:,}")
    
    # Analytics
    analytics = tree.add("[cyan]Analytics[/cyan]")
    analytics.add(f"Use Case: {config.analytics.use_case.value}")
    analytics.add(f"Difficulty: {config.difficulty.value}")
    
    # Output
    output = tree.add("[cyan]Output[/cyan]")
    output.add(f"Format: {config.output.format.value}")
    output.add(f"Directory: {config.output.directory}")
    
    console.print(tree)


def _display_generation_summary(
    tables: dict,
    output_path: Path,
    config: SynthDataConfig,
):
    """Display generation summary."""
    console.print("\n[green]✓[/green] [bold]Generation Complete![/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    
    total_rows = 0
    for name, df in tables.items():
        table.add_row(name, f"{len(df):,}", str(len(df.columns)))
        total_rows += len(df)
    
    table.add_row("[bold]Total[/bold]", f"[bold]{total_rows:,}[/bold]", "")
    
    console.print(table)
    
    console.print(f"\n[dim]Files saved to: {output_path}[/dim]")
    console.print(f"[dim]Seed: {config.reproducibility.seed}[/dim]")
    
    if config.output.include_metadata:
        console.print("\n[dim]Metadata files:[/dim]")
        console.print(f"  - {output_path}/metadata/config.yaml")
        console.print(f"  - {output_path}/metadata/data_dictionary.json")
        console.print(f"  - {output_path}/metadata/quality_report.json")
        console.print(f"  - {output_path}/metadata/suggested_questions.md")


if __name__ == "__main__":
    main()
