"""Paths and file names used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

# Short name -> CSV file name as published on Kaggle
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

FEATURES_FILE = PROCESSED_DIR / "orders_features.parquet"

# Months held out at the end of the timeline for the out-of-time test
TEST_MONTHS = 3
