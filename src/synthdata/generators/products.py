"""
Product data generator.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    DataQualityConfig,
    Industry,
)
from synthdata.generators.base import BaseGenerator


class ProductGenerator(BaseGenerator):
    """Generator for product/service data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        self.categories = self._get_categories()
        self.brands = self._generate_brands()
    
    def _get_categories(self) -> Dict[str, List[str]]:
        """Get product categories and subcategories based on industry."""
        categories = {
            Industry.RETAIL: {
                "Electronics": ["Smartphones", "Laptops", "Tablets", "Accessories", "Audio"],
                "Clothing": ["Men's Wear", "Women's Wear", "Kids", "Sportswear", "Formal"],
                "Home & Garden": ["Furniture", "Decor", "Kitchen", "Garden", "Bedding"],
                "Sports": ["Equipment", "Apparel", "Footwear", "Accessories"],
                "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance"],
            },
            Industry.ECOMMERCE: {
                "Electronics": ["Phones", "Computers", "Gaming", "Cameras", "Wearables"],
                "Fashion": ["Clothing", "Shoes", "Bags", "Jewelry", "Watches"],
                "Home": ["Furniture", "Kitchen", "Bedding", "Storage", "Lighting"],
                "Health": ["Supplements", "Personal Care", "Fitness", "Medical"],
                "Books & Media": ["Books", "Music", "Movies", "Games"],
            },
            Industry.SAAS: {
                "Productivity": ["Project Management", "Documentation", "Collaboration"],
                "Marketing": ["Email Marketing", "SEO Tools", "Social Media", "Analytics"],
                "Sales": ["CRM", "Sales Automation", "Lead Generation"],
                "Development": ["IDE", "Version Control", "CI/CD", "Monitoring"],
                "Finance": ["Accounting", "Invoicing", "Payroll", "Expense Management"],
            },
            Industry.HEALTHCARE: {
                "Pharmaceuticals": ["Prescription", "OTC", "Supplements", "Medical Devices"],
                "Services": ["Consultation", "Diagnostics", "Treatment", "Therapy"],
                "Equipment": ["Imaging", "Lab Equipment", "Surgical", "Patient Care"],
            },
            Industry.MANUFACTURING: {
                "Raw Materials": ["Metals", "Plastics", "Chemicals", "Textiles"],
                "Components": ["Electronic", "Mechanical", "Structural"],
                "Equipment": ["Machinery", "Tools", "Safety", "Packaging"],
            },
        }
        
        return categories.get(
            self.business_context.industry,
            {"General": ["Product A", "Product B", "Product C"]}
        )
    
    def _generate_brands(self) -> List[str]:
        """Generate brand names."""
        prefixes = ["Pro", "Ultra", "Max", "Prime", "Elite", "Core", "Neo", "Zen"]
        suffixes = ["Tech", "Labs", "Co", "Works", "Hub", "Plus", "X", "One"]
        
        brands = []
        for _ in range(50):
            if random.random() > 0.5:
                brand = f"{random.choice(prefixes)}{self.faker.last_name()}"
            else:
                brand = f"{self.faker.last_name()}{random.choice(suffixes)}"
            brands.append(brand)
        
        # Add some well-known style brands
        brands.extend(["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne"])
        
        return brands
    
    def _generate_product_name(self, category: str, subcategory: str) -> str:
        """Generate a product name."""
        adjectives = [
            "Premium", "Essential", "Professional", "Classic", "Modern",
            "Advanced", "Smart", "Eco", "Ultra", "Deluxe"
        ]
        
        product_types = {
            "Electronics": ["Device", "System", "Unit", "Module", "Kit"],
            "Clothing": ["Collection", "Series", "Line", "Edition"],
            "Home": ["Set", "Collection", "System", "Kit"],
            "Default": ["Item", "Product", "Solution"],
        }
        
        types = product_types.get(category, product_types["Default"])
        
        return f"{random.choice(adjectives)} {subcategory} {random.choice(types)}"
    
    def _generate_price(self, category: str) -> Tuple[float, float]:
        """Generate base price and cost for a product."""
        # Price ranges by category
        price_ranges = {
            "Electronics": (50, 2000),
            "Clothing": (20, 500),
            "Home": (30, 1500),
            "Home & Garden": (30, 1500),
            "Fashion": (25, 800),
            "Sports": (20, 500),
            "Beauty": (10, 200),
            "Health": (15, 300),
            "Books & Media": (5, 50),
            "Pharmaceuticals": (5, 500),
            "Services": (50, 5000),
            "Raw Materials": (10, 10000),
            "Components": (1, 500),
            "Equipment": (100, 50000),
            "Productivity": (10, 100),
            "Marketing": (20, 500),
            "Sales": (50, 300),
            "Development": (10, 200),
            "Finance": (30, 200),
        }
        
        min_price, max_price = price_ranges.get(category, (10, 500))
        base_price = round(self.generate_skewed_value(min_price, max_price, skew=1.5), 2)
        
        # Cost is typically 40-70% of price
        margin = random.uniform(0.3, 0.6)
        cost = round(base_price * (1 - margin), 2)
        
        return base_price, cost
    
    def _generate_sku(self, category: str) -> str:
        """Generate a SKU."""
        category_prefix = category[:3].upper()
        return f"{category_prefix}-{self.generate_id('', 6)}"
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Generate product data."""
        if num_rows is None:
            num_rows = self.business_size.num_products
        
        products = []
        
        for i in range(num_rows):
            product_id = self.generate_id("PRD", 8)
            
            # Select category and subcategory
            category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[category])
            
            # Generate product details
            name = self._generate_product_name(category, subcategory)
            sku = self._generate_sku(category)
            brand = random.choice(self.brands)
            base_price, cost = self._generate_price(category)
            
            # Stock and inventory
            initial_stock = random.randint(0, 1000)
            reorder_point = random.randint(10, 100)
            
            # Ratings (slightly skewed towards positive)
            avg_rating = min(5.0, max(1.0, random.gauss(4.0, 0.8)))
            num_reviews = int(self.generate_skewed_value(0, 500, skew=2.0))
            
            # Dates
            created_date = self.random_date(
                self.start_date,
                self.end_date - timedelta(days=30)
            )
            
            # Status
            status_weights = [0.7, 0.15, 0.1, 0.05]
            status = self.weighted_choice(
                ["Active", "Low Stock", "Out of Stock", "Discontinued"],
                status_weights
            )
            
            product = {
                "product_id": product_id,
                "sku": sku,
                "name": name,
                "description": self.faker.paragraph(nb_sentences=3),
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "base_price": base_price,
                "cost": cost,
                "current_price": base_price * random.uniform(0.8, 1.2),
                "currency": "USD",
                "stock_quantity": initial_stock,
                "reorder_point": reorder_point,
                "supplier_id": self.generate_id("SUP", 6),
                "weight_kg": round(random.uniform(0.1, 20), 2),
                "dimensions": f"{random.randint(5, 100)}x{random.randint(5, 100)}x{random.randint(5, 50)}",
                "avg_rating": round(avg_rating, 2),
                "num_reviews": num_reviews,
                "is_featured": random.random() < 0.1,
                "is_bestseller": random.random() < 0.15,
                "status": status,
                "created_date": created_date,
                "last_updated": self.random_date(created_date, self.end_date),
                "tags": self._generate_tags(category, subcategory),
            }
            
            # Industry-specific fields
            if self.business_context.industry == Industry.SAAS:
                product["plan_type"] = random.choice(["Free", "Starter", "Pro", "Enterprise"])
                product["billing_cycle"] = random.choice(["monthly", "annual"])
                product["trial_days"] = random.choice([0, 7, 14, 30])
                product["max_users"] = random.choice([1, 5, 10, 25, 100, -1])  # -1 = unlimited
                product["storage_gb"] = random.choice([1, 5, 10, 50, 100, -1])
            
            elif self.business_context.industry == Industry.HEALTHCARE:
                product["requires_prescription"] = random.random() < 0.3
                product["controlled_substance"] = random.random() < 0.05
                product["dosage_form"] = random.choice(["Tablet", "Capsule", "Liquid", "Injection", "Topical", "N/A"])
                product["manufacturer"] = self.faker.company()
            
            elif self.business_context.industry == Industry.MANUFACTURING:
                product["material"] = random.choice(["Steel", "Aluminum", "Plastic", "Composite", "Mixed"])
                product["lead_time_days"] = random.randint(1, 90)
                product["minimum_order_qty"] = random.choice([1, 10, 50, 100, 500])
                product["quality_grade"] = random.choice(["A", "B", "C"])
            
            products.append(product)
        
        df = pd.DataFrame(products)
        
        # Sort by created date
        df = df.sort_values("created_date").reset_index(drop=True)
        
        return df
    
    def _generate_tags(self, category: str, subcategory: str) -> str:
        """Generate product tags."""
        general_tags = ["new", "popular", "sale", "limited", "exclusive", "eco-friendly"]
        category_tags = [category.lower().replace(" ", "-"), subcategory.lower().replace(" ", "-")]
        
        num_tags = random.randint(2, 5)
        selected_tags = random.sample(general_tags, min(num_tags - 1, len(general_tags)))
        selected_tags.extend(random.sample(category_tags, min(2, len(category_tags))))
        
        return ",".join(selected_tags[:num_tags])
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for product data."""
        schema = {
            "product_id": "string",
            "sku": "string",
            "name": "string",
            "description": "text",
            "category": "category",
            "subcategory": "category",
            "brand": "category",
            "base_price": "float",
            "cost": "float",
            "current_price": "float",
            "currency": "category",
            "stock_quantity": "integer",
            "reorder_point": "integer",
            "supplier_id": "string",
            "weight_kg": "float",
            "dimensions": "string",
            "avg_rating": "float",
            "num_reviews": "integer",
            "is_featured": "boolean",
            "is_bestseller": "boolean",
            "status": "category",
            "created_date": "datetime",
            "last_updated": "datetime",
            "tags": "string",
        }
        
        return schema
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        return {
            "product_id": "Unique identifier for the product",
            "sku": "Stock Keeping Unit - unique product code",
            "name": "Product name",
            "description": "Product description",
            "category": "Main product category",
            "subcategory": "Product subcategory",
            "brand": "Brand name",
            "base_price": "Original base price",
            "cost": "Product cost (COGS)",
            "current_price": "Current selling price",
            "currency": "Price currency",
            "stock_quantity": "Current stock quantity",
            "reorder_point": "Stock level that triggers reorder",
            "supplier_id": "Reference to supplier",
            "weight_kg": "Product weight in kilograms",
            "dimensions": "Product dimensions (LxWxH in cm)",
            "avg_rating": "Average customer rating (1-5)",
            "num_reviews": "Number of customer reviews",
            "is_featured": "Whether product is featured",
            "is_bestseller": "Whether product is a bestseller",
            "status": "Product availability status",
            "created_date": "Date product was added",
            "last_updated": "Date of last update",
            "tags": "Product tags for search/categorization",
        }


# Type hint for tuple
from typing import Tuple
