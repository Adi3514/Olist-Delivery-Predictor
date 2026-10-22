"""Read the raw Olist CSVs with the right types."""
from pathlib import Path

import pandas as pd

from olist.config import RAW_DIR, RAW_FILES

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

ZIP_COLUMNS = ["customer_zip_code_prefix", "seller_zip_code_prefix", "geolocation_zip_code_prefix"]


def load_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m olist.download` first.")
    # Zip prefixes have leading zeros, so keep them as strings
    dtypes = {col: str for col in ZIP_COLUMNS}
    return pd.read_csv(path, parse_dates=DATE_COLUMNS.get(name, []), dtype=dtypes)


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {name: load_table(name, raw_dir) for name in RAW_FILES}
