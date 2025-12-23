# 📊 SynthData

**Generate realistic, messy datasets for data science practice.**

```
███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗██████╗  █████╗ ████████╗ █████╗ 
██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║██║  ██║███████║   ██║   ███████║
╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██║  ██║██╔══██║   ██║   ██╔══██║
███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██████╔╝██║  ██║   ██║   ██║  ██║
╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
```

## 🚀 Just Run It

```bash
python synthdata.py
```

That's it. Follow the menus.

## 📦 Install

```bash
pip install -r requirements.txt
python synthdata.py
```

## 📁 Files

```
synthdata.py     ← Run this!
generators.py    ← Data generation logic
quality.py       ← Quality issues injection
requirements.txt ← Dependencies
```

## 🎯 What You Get

| Table | Description |
|-------|-------------|
| `customers.csv` | Customer profiles with demographics, LTV, status |
| `products.csv` | Product catalog with categories, prices, margins |
| `transactions.csv` | Purchase history with payments, discounts |
| `support_tickets.csv` | Customer support interactions |

## 🔧 Quality Issues Included

The generated data includes realistic problems for you to clean:

- **Missing Values** - NaN, empty strings, "N/A", "null"
- **Duplicates** - Exact and near-duplicate rows
- **Outliers** - Extreme values in numeric columns
- **Typos** - Character swaps, doubled letters
- **Inconsistent Formats** - Multiple date formats, case variations
- **Whitespace Issues** - Leading/trailing spaces

## 🏭 Industries

- E-Commerce
- Retail
- Fintech
- Healthcare
- SaaS
- Logistics

## 📈 Difficulty Levels

| Level | Quality Rate | Description |
|-------|-------------|-------------|
| Clean | 2% | Perfect for learning basics |
| Messy | 8% | Real-world quality issues |
| Dirty | 15% | Challenging data cleaning |
| Chaotic | 25% | Nightmare mode 💀 |

## License

MIT
