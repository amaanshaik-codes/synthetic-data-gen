import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import openpyxl
except ImportError:
    openpyxl = None

from generators import GENERATORS
from quality import QualityInjector, LogicalIssueInjector


console = Console()

INDUSTRIES = list(GENERATORS.keys())

SIZES = {
    "small": {"scale": 0.2},
    "medium": {"scale": 1.0},
    "large": {"scale": 5.0},
}

QUALITY_LEVELS = {
    "clean": 0.0,
    "light": 0.02,
    "moderate": 0.05,
    "heavy": 0.10,
}


def show_menu(title: str, options: list) -> int:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(justify="right", style="cyan", width=4)
    table.add_column(style="white")
    
    for i, opt in enumerate(options, 1):
        table.add_row(f"[{i}]", opt)
    
    console.print(Panel(table, title=title, border_style="blue"))
    
    while True:
        try:
            choice = IntPrompt.ask("Select", default=1)
            if 1 <= choice <= len(options):
                return choice
            console.print(f"[red]Enter 1-{len(options)}[/red]")
        except KeyboardInterrupt:
            sys.exit(0)


def collect_config() -> Optional[dict]:
    config = {}
    
    console.print("\n[bold]1. Industry[/bold]")
    idx = show_menu("Industry", INDUSTRIES)
    config["industry"] = INDUSTRIES[idx - 1]
    
    console.print("\n[bold]2. Data Size[/bold]")
    size_opts = list(SIZES.keys())
    idx = show_menu("Size", size_opts)
    config["size"] = size_opts[idx - 1]
    config["scale"] = SIZES[config["size"]]["scale"]
    
    console.print("\n[bold]3. Time Period[/bold]")
    config["months"] = IntPrompt.ask("Months of data", default=12)
    if config["months"] < 1:
        config["months"] = 1
    if config["months"] > 60:
        config["months"] = 60
    
    console.print("\n[bold]4. Data Quality[/bold]")
    quality_opts = [f"{k} ({int(v*100)}% issues)" for k, v in QUALITY_LEVELS.items()]
    idx = show_menu("Quality", quality_opts)
    quality_name = list(QUALITY_LEVELS.keys())[idx - 1]
    config["quality"] = quality_name
    config["quality_rate"] = QUALITY_LEVELS[quality_name]
    
    console.print("\n[bold]5. Output[/bold]")
    formats = ["csv", "parquet", "json", "excel", "sqlite", "all"]
    idx = show_menu("Format", formats)
    config["format"] = formats[idx - 1]
    config["output_dir"] = Prompt.ask("Output folder", default="./output")
    
    console.print("\n[bold]6. Advanced Options[/bold]")
    if Confirm.ask("Customize advanced settings?", default=False):
        config["seed"] = IntPrompt.ask("Random seed", default=42)
        config["growth_rate"] = IntPrompt.ask("Monthly growth % (0-50)", default=5)
        if Confirm.ask("Generate analysis report?", default=True):
            config["generate_report"] = True
    else:
        config["seed"] = 42
        config["growth_rate"] = 5
        config["generate_report"] = True
    
    console.print()
    summary = Table(show_header=False, box=None)
    summary.add_column(style="cyan", width=18)
    summary.add_column(style="white")
    summary.add_row("Industry", config["industry"])
    summary.add_row("Size", config["size"])
    summary.add_row("Months", str(config["months"]))
    summary.add_row("Quality", config["quality"])
    summary.add_row("Format", config["format"])
    summary.add_row("Output", config["output_dir"])
    summary.add_row("Growth Rate", f"{config.get('growth_rate', 5)}%/month")
    summary.add_row("Generate Report", "Yes" if config.get("generate_report", True) else "No")
    console.print(Panel(summary, title="Configuration", border_style="green"))
    
    if not Confirm.ask("\nGenerate?", default=True):
        return None
    return config


def generate(config: dict) -> Dict[str, pd.DataFrame]:
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Generating...", total=None)
        
        GeneratorClass = GENERATORS[config["industry"]]
        scale = config["scale"]
        
        seed = config.get("seed", 42)
        growth = config.get("growth_rate", 5) / 100
        
        if config["industry"] == "retail":
            gen = GeneratorClass(
                n_customers=int(500 * scale),
                n_products=int(100 * scale),
                n_stores=int(20 * scale),
                n_transactions=int(5000 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        elif config["industry"] == "ecommerce":
            gen = GeneratorClass(
                n_customers=int(500 * scale),
                n_products=int(100 * scale),
                n_orders=int(2000 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        elif config["industry"] == "banking":
            gen = GeneratorClass(
                n_customers=int(500 * scale),
                n_accounts=int(800 * scale),
                n_branches=int(30 * scale),
                n_transactions=int(10000 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        elif config["industry"] == "healthcare":
            gen = GeneratorClass(
                n_patients=int(500 * scale),
                n_doctors=int(50 * scale),
                n_hospitals=int(10 * scale),
                n_encounters=int(3000 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        elif config["industry"] == "saas":
            gen = GeneratorClass(
                n_customers=int(300 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        elif config["industry"] == "logistics":
            gen = GeneratorClass(
                n_routes=int(100 * scale),
                n_vehicles=int(50 * scale),
                n_warehouses=int(10 * scale),
                n_shipments=int(5000 * scale),
                months=config["months"],
                seed=seed,
                growth_rate=growth
            )
        
        progress.update(task, description="Generating tables...")
        tables = gen.generate_all()
        
        if config["quality_rate"] > 0:
            progress.update(task, description="Injecting quality issues...")
            injector = QualityInjector(quality_rate=config["quality_rate"])
            
            for name, df in tables.items():
                is_dim = name.startswith("dim_")
                tables[name] = injector.inject_issues(df, is_dimension=is_dim)
        
        progress.update(task, description="Saving files...")
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fmt = config["format"]
        for name, df in tables.items():
            if fmt in ["csv", "all"]:
                df.to_csv(output_dir / f"{name}.csv", index=False)
            if fmt in ["parquet", "all"]:
                df.to_parquet(output_dir / f"{name}.parquet", index=False)
            if fmt in ["json", "all"]:
                df.to_json(output_dir / f"{name}.json", orient="records", lines=True)
            if fmt in ["excel", "all"]:
                df.to_excel(output_dir / f"{name}.xlsx", index=False, engine="openpyxl")
        
        if fmt in ["sqlite", "all"]:
            import sqlite3
            db_path = output_dir / "data.db"
            conn = sqlite3.connect(db_path)
            for name, df in tables.items():
                df.to_sql(name, conn, if_exists="replace", index=False)
            conn.close()
        
        if config.get("generate_report", True):
            progress.update(task, description="Generating report...")
            from report import generate_report
            report_path = output_dir / "ANALYSIS_REPORT.md"
            generate_report(tables, config, report_path)
    
    return tables


def show_results(tables: Dict[str, pd.DataFrame], output_dir: str):
    console.print("\n[bold green]Done[/bold green]\n")
    
    result = Table(title="Generated Tables")
    result.add_column("Table", style="cyan")
    result.add_column("Rows", justify="right")
    result.add_column("Columns", justify="right")
    
    total = 0
    for name, df in tables.items():
        result.add_row(name, f"{len(df):,}", str(len(df.columns)))
        total += len(df)
    
    result.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]", "")
    console.print(result)
    console.print(f"\nSaved to: [cyan]{output_dir}[/cyan]")


def main():
    console.print("\n[bold]Synthetic Data Generator[/bold]\n")
    
    try:
        config = collect_config()
        if not config:
            console.print("[yellow]Cancelled[/yellow]")
            return
        
        tables = generate(config)
        show_results(tables, config["output_dir"])
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
