#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SYNTHDATA - Synthetic Data Generator for Data Science Practice              ║
║  Just run: python synthdata.py                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

# Local modules
from generators import DataGenerator
from quality import QualityInjector

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

INDUSTRIES = {
    "1": ("E-Commerce", "ecommerce", "🛒"),
    "2": ("Retail", "retail", "🏪"),
    "3": ("Fintech", "fintech", "💳"),
    "4": ("Healthcare", "healthcare", "🏥"),
    "5": ("SaaS", "saas", "☁️"),
    "6": ("Logistics", "logistics", "🚚"),
}

SIZES = {
    "1": ("Small (1K customers)", 1000, 50, 100),      # customers, products, daily_txn
    "2": ("Medium (10K customers)", 10000, 200, 500),
    "3": ("Large (50K customers)", 50000, 500, 2000),
    "4": ("Massive (100K customers)", 100000, 1000, 5000),
}

DIFFICULTIES = {
    "1": ("Clean", 0.02, "Perfect for learning basics"),
    "2": ("Messy", 0.08, "Real-world quality issues"),
    "3": ("Dirty", 0.15, "Challenging data cleaning"),
    "4": ("Chaotic", 0.25, "Nightmare mode 💀"),
}

FORMATS = {
    "1": ("CSV", "csv"),
    "2": ("Parquet", "parquet"),
    "3": ("JSON", "json"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def show_banner():
    """Display the main banner."""
    banner = """
[bold cyan]
███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗██████╗  █████╗ ████████╗ █████╗ 
██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║██║  ██║███████║   ██║   ███████║
╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██║  ██║██╔══██║   ██║   ██╔══██║
███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██████╔╝██║  ██║   ██║   ██║  ██║
╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
[/bold cyan]
[dim]Synthetic Data Generator for Data Science Practice[/dim]
    """
    console.print(banner)


def show_menu(title: str, options: dict, show_emoji: bool = True) -> str:
    """Display a menu and get user selection."""
    console.print(f"\n[bold yellow]▶ {title}[/bold yellow]\n")
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan", width=4)
    table.add_column("Option", style="white")
    
    for key, value in options.items():
        if isinstance(value, tuple):
            if show_emoji and len(value) >= 3:
                display = f"{value[2]} {value[0]}"
            else:
                display = value[0]
            # Add description if available
            if len(value) >= 3 and isinstance(value[2], str) and not value[2].startswith(("🛒", "🏪", "💳", "🏥", "☁️", "🚚")):
                display = f"{value[0]} [dim]- {value[2]}[/dim]"
        else:
            display = value
        table.add_row(f"[{key}]", display)
    
    console.print(table)
    
    while True:
        choice = Prompt.ask("\n[bold]Select option[/bold]", default="1")
        if choice in options:
            return choice
        console.print("[red]Invalid choice. Try again.[/red]")


def show_summary(config: dict):
    """Display configuration summary."""
    console.print("\n")
    table = Table(title="📋 Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Industry", config["industry_name"])
    table.add_row("Dataset Size", config["size_name"])
    table.add_row("Customers", f"{config['num_customers']:,}")
    table.add_row("Products", f"{config['num_products']:,}")
    table.add_row("Daily Transactions", f"{config['daily_transactions']:,}")
    table.add_row("Time Period", f"{config['months']} months")
    table.add_row("Data Quality", config["difficulty_name"])
    table.add_row("Quality Rate", f"{config['quality_rate']*100:.0f}% issues")
    table.add_row("Output Format", config["format"].upper())
    table.add_row("Output Directory", config["output_dir"])
    table.add_row("Seed", str(config["seed"]))
    
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def collect_config() -> dict:
    """Interactive configuration collection."""
    config = {}
    
    # Step 1: Industry
    console.print(Panel("[bold]Step 1 of 6[/bold]: Choose your industry", style="blue"))
    choice = show_menu("Industry", INDUSTRIES)
    industry_name, industry_key, _ = INDUSTRIES[choice]
    config["industry_name"] = industry_name
    config["industry"] = industry_key
    
    # Step 2: Size
    console.print(Panel("[bold]Step 2 of 6[/bold]: Dataset size", style="blue"))
    choice = show_menu("Size", SIZES, show_emoji=False)
    size_name, customers, products, daily_txn = SIZES[choice]
    config["size_name"] = size_name
    config["num_customers"] = customers
    config["num_products"] = products
    config["daily_transactions"] = daily_txn
    
    # Step 3: Time period
    console.print(Panel("[bold]Step 3 of 6[/bold]: Time period", style="blue"))
    months = IntPrompt.ask(
        "[bold]How many months of data?[/bold]",
        default=12,
        show_default=True
    )
    config["months"] = max(1, min(60, months))  # Clamp between 1-60
    
    # Step 4: Data quality
    console.print(Panel("[bold]Step 4 of 6[/bold]: Data quality (difficulty)", style="blue"))
    console.print("[dim]Higher difficulty = more missing values, duplicates, and noise[/dim]")
    choice = show_menu("Difficulty", {
        "1": ("Clean", 0.02, "Perfect for learning basics"),
        "2": ("Messy", 0.08, "Real-world quality issues"),
        "3": ("Dirty", 0.15, "Challenging data cleaning"),
        "4": ("Chaotic", 0.25, "Nightmare mode 💀"),
    })
    difficulty_name, quality_rate, _ = DIFFICULTIES[choice]
    config["difficulty_name"] = difficulty_name
    config["quality_rate"] = quality_rate
    
    # Step 5: Output format
    console.print(Panel("[bold]Step 5 of 6[/bold]: Output format", style="blue"))
    choice = show_menu("Format", FORMATS, show_emoji=False)
    config["format"] = FORMATS[choice][1]
    
    # Step 6: Output directory
    console.print(Panel("[bold]Step 6 of 6[/bold]: Output location", style="blue"))
    default_dir = f"./data_{config['industry']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Prompt.ask(
        "[bold]Output directory[/bold]",
        default=default_dir
    )
    config["output_dir"] = output_dir
    
    # Seed for reproducibility
    config["seed"] = int(datetime.now().timestamp()) % 100000
    
    return config


def quick_generate() -> dict:
    """Quick generation with smart defaults."""
    console.print("\n[bold green]⚡ Quick Generate Mode[/bold green]")
    console.print("[dim]Generating e-commerce dataset with sensible defaults...[/dim]\n")
    
    return {
        "industry_name": "E-Commerce",
        "industry": "ecommerce",
        "size_name": "Medium (10K customers)",
        "num_customers": 10000,
        "num_products": 200,
        "daily_transactions": 500,
        "months": 12,
        "difficulty_name": "Messy",
        "quality_rate": 0.08,
        "format": "csv",
        "output_dir": f"./data_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "seed": int(datetime.now().timestamp()) % 100000,
    }


def generate_data(config: dict):
    """Generate the synthetic data."""
    console.print("\n")
    
    # Create output directory
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        
        # Initialize generator
        task = progress.add_task("[cyan]Initializing...", total=100)
        generator = DataGenerator(
            industry=config["industry"],
            num_customers=config["num_customers"],
            num_products=config["num_products"],
            daily_transactions=config["daily_transactions"],
            months=config["months"],
            seed=config["seed"],
        )
        progress.update(task, advance=10)
        
        # Generate tables
        progress.update(task, description="[cyan]Generating customers...")
        customers = generator.generate_customers()
        progress.update(task, advance=15)
        
        progress.update(task, description="[cyan]Generating products...")
        products = generator.generate_products()
        progress.update(task, advance=15)
        
        progress.update(task, description="[cyan]Generating transactions...")
        transactions = generator.generate_transactions(customers, products)
        progress.update(task, advance=20)
        
        progress.update(task, description="[cyan]Generating support tickets...")
        tickets = generator.generate_support_tickets(customers, products)
        progress.update(task, advance=10)
        
        # Inject quality issues
        progress.update(task, description="[yellow]Injecting quality issues...")
        injector = QualityInjector(config["quality_rate"], config["seed"])
        
        customers = injector.inject(customers, "customers")
        products = injector.inject(products, "products")
        transactions = injector.inject(transactions, "transactions")
        tickets = injector.inject(tickets, "tickets")
        progress.update(task, advance=15)
        
        # Save files
        progress.update(task, description="[green]Saving files...")
        tables = {
            "customers": customers,
            "products": products,
            "transactions": transactions,
            "support_tickets": tickets,
        }
        
        for name, df in tables.items():
            file_path = output_path / f"{name}.{config['format']}"
            if config["format"] == "csv":
                df.to_csv(file_path, index=False)
            elif config["format"] == "parquet":
                df.to_parquet(file_path, index=False)
            elif config["format"] == "json":
                df.to_json(file_path, orient="records", indent=2)
        
        progress.update(task, advance=15, description="[bold green]✓ Complete!")
    
    # Show results
    console.print("\n")
    results_table = Table(title="📊 Generated Data", show_header=True, header_style="bold green")
    results_table.add_column("Table", style="cyan")
    results_table.add_column("Rows", justify="right", style="yellow")
    results_table.add_column("Columns", justify="right")
    results_table.add_column("File", style="dim")
    
    total_rows = 0
    for name, df in tables.items():
        file_name = f"{name}.{config['format']}"
        results_table.add_row(name, f"{len(df):,}", str(len(df.columns)), file_name)
        total_rows += len(df)
    
    results_table.add_row("", "", "", "", style="dim")
    results_table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_rows:,}[/bold]", "", "")
    
    console.print(results_table)
    
    console.print(f"\n[bold green]✓[/bold green] Data saved to: [cyan]{output_path.absolute()}[/cyan]")
    console.print(f"[dim]Seed: {config['seed']} (use this to reproduce the exact dataset)[/dim]")
    
    # Quality summary
    console.print(f"\n[bold yellow]📋 Quality Issues Injected:[/bold yellow]")
    console.print(f"   • Missing values: ~{config['quality_rate']*100:.0f}% of cells")
    console.print(f"   • Duplicates: ~{config['quality_rate']*50:.1f}% of rows")
    console.print(f"   • Outliers: ~{config['quality_rate']*30:.1f}% extreme values")
    console.print(f"   • Inconsistencies: Date formats, typos, case variations")


def main():
    """Main entry point."""
    os.system('cls' if os.name == 'nt' else 'clear')
    show_banner()
    
    console.print(Panel(
        "[bold]Welcome![/bold] Generate realistic, messy datasets for data science practice.\n\n"
        "[dim]This tool creates interconnected tables with configurable quality issues:\n"
        "missing values, duplicates, outliers, and inconsistencies.[/dim]",
        title="🚀 SynthData",
        border_style="cyan"
    ))
    
    # Main menu
    main_options = {
        "1": "⚡ Quick Generate (recommended for first-timers)",
        "2": "🎛️  Custom Generate (full control)",
        "3": "❓ Help",
        "4": "🚪 Exit",
    }
    
    choice = show_menu("What would you like to do?", main_options, show_emoji=False)
    
    if choice == "1":
        config = quick_generate()
        show_summary(config)
        if Confirm.ask("\n[bold]Proceed with generation?[/bold]", default=True):
            generate_data(config)
        else:
            console.print("[yellow]Cancelled.[/yellow]")
            
    elif choice == "2":
        config = collect_config()
        show_summary(config)
        if Confirm.ask("\n[bold]Proceed with generation?[/bold]", default=True):
            generate_data(config)
        else:
            console.print("[yellow]Cancelled.[/yellow]")
            
    elif choice == "3":
        console.print(Panel(
            "[bold]How to use SynthData:[/bold]\n\n"
            "1. [cyan]Quick Generate[/cyan] - One-click dataset with smart defaults\n"
            "2. [cyan]Custom Generate[/cyan] - Choose industry, size, quality level\n\n"
            "[bold]Generated Tables:[/bold]\n"
            "• customers - Customer profiles with demographics\n"
            "• products - Product catalog with categories & prices\n"
            "• transactions - Purchase history linked to customers\n"
            "• support_tickets - Customer support interactions\n\n"
            "[bold]Quality Issues:[/bold]\n"
            "• Missing values (NaN, empty strings, nulls)\n"
            "• Duplicate rows (exact and near-duplicates)\n"
            "• Outliers (extreme values in numeric columns)\n"
            "• Inconsistencies (date formats, typos, case)\n\n"
            "[dim]Perfect for practicing: EDA, data cleaning, feature engineering, ML[/dim]",
            title="❓ Help",
            border_style="yellow"
        ))
        
        if Confirm.ask("\n[bold]Start generating?[/bold]", default=True):
            config = quick_generate()
            generate_data(config)
    
    elif choice == "4":
        console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
        sys.exit(0)
    
    console.print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user.[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print("[dim]Please report this issue if it persists.[/dim]\n")
        sys.exit(1)
