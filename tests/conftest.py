"""A small fake Olist dataset with the same columns as the real one."""
import numpy as np
import pandas as pd
import pytest

from olist.config import RAW_FILES

STATES = {"SP": (-23.5, -46.6), "RJ": (-22.9, -43.2), "MG": (-19.9, -43.9), "BA": (-12.9, -38.5), "RS": (-30.0, -51.2)}


def make_tables(n_orders: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    states = list(STATES)

    zips = [f"{i:05d}" for i in range(50)]
    zip_state = {z: states[i % len(states)] for i, z in enumerate(zips)}
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": zips,
        "geolocation_lat": [STATES[zip_state[z]][0] + rng.normal(0, 0.1) for z in zips],
        "geolocation_lng": [STATES[zip_state[z]][1] + rng.normal(0, 0.1) for z in zips],
        "geolocation_city": "city",
        "geolocation_state": [zip_state[z] for z in zips],
    })

    customer_zips = rng.choice(zips, n_orders)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n_orders)],
        "customer_unique_id": [f"u{i}" for i in range(n_orders)],
        "customer_zip_code_prefix": customer_zips,
        "customer_city": "city",
        "customer_state": [zip_state[z] for z in customer_zips],
    })

    seller_zips = rng.choice(zips, 30)
    sellers = pd.DataFrame({
        "seller_id": [f"s{i}" for i in range(30)],
        "seller_zip_code_prefix": seller_zips,
        "seller_city": "city",
        "seller_state": [zip_state[z] for z in seller_zips],
    })

    categories = pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "beleza_saude", "esporte_lazer"],
        "product_category_name_english": ["bed_bath_table", "health_beauty", "sports_leisure"],
    })
    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(100)],
        "product_category_name": rng.choice(categories["product_category_name"], 100),
        "product_name_lenght": 40,
        "product_description_lenght": 500,
        "product_photos_qty": rng.integers(1, 5, 100),
        "product_weight_g": rng.integers(100, 20000, 100),
        "product_length_cm": rng.integers(10, 60, 100),
        "product_height_cm": rng.integers(2, 40, 100),
        "product_width_cm": rng.integers(10, 50, 100),
    })

    start = pd.Timestamp("2017-01-01")
    purchase = start + pd.to_timedelta(np.sort(rng.uniform(0, 600, n_orders)), unit="D")
    promised = rng.integers(10, 40, n_orders)
    estimated = (purchase + pd.to_timedelta(promised, unit="D")).normalize()
    seller_idx = rng.integers(0, 30, n_orders)
    slow_seller = seller_idx < 5  # a few sellers are late much more often
    actual = rng.gamma(4, 3, n_orders) + np.where(slow_seller, 15, 0)
    delivered = purchase + pd.to_timedelta(actual, unit="D")

    status = np.full(n_orders, "delivered", dtype=object)
    status[rng.random(n_orders) < 0.01] = "canceled"
    status[rng.random(n_orders) < 0.01] = "shipped"
    delivered = pd.Series(delivered).where(status == "delivered")
    snapshot = purchase.max()
    delivered = delivered.where(delivered <= snapshot)
    status = np.where((status == "delivered") & delivered.isna(), "shipped", status)

    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n_orders)],
        "customer_id": customers["customer_id"],
        "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": purchase + pd.Timedelta(hours=1),
        "order_delivered_carrier_date": purchase + pd.Timedelta(days=2),
        "order_delivered_customer_date": delivered.values,
        "order_estimated_delivery_date": estimated,
    })

    items = pd.DataFrame({
        "order_id": orders["order_id"],
        "order_item_id": 1,
        "product_id": rng.choice(products["product_id"], n_orders),
        "seller_id": [f"s{i}" for i in seller_idx],
        "shipping_limit_date": purchase + pd.Timedelta(days=5),
        "price": rng.uniform(10, 500, n_orders).round(2),
        "freight_value": rng.uniform(5, 60, n_orders).round(2),
    })
    payments = pd.DataFrame({
        "order_id": orders["order_id"],
        "payment_sequential": 1,
        "payment_type": rng.choice(["credit_card", "boleto"], n_orders),
        "payment_installments": rng.integers(1, 10, n_orders),
        "payment_value": items["price"] + items["freight_value"],
    })
    reviews = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(n_orders)],
        "order_id": orders["order_id"],
        "review_score": rng.integers(1, 6, n_orders),
        "review_comment_title": None,
        "review_comment_message": None,
        "review_creation_date": purchase + pd.Timedelta(days=20),
        "review_answer_timestamp": purchase + pd.Timedelta(days=21),
    })

    return {
        "orders": orders, "items": items, "customers": customers, "sellers": sellers,
        "products": products, "payments": payments, "reviews": reviews,
        "geolocation": geolocation, "categories": categories,
    }


@pytest.fixture(scope="session")
def tables():
    return make_tables()


@pytest.fixture
def raw_dir(tmp_path, tables):
    for name, df in tables.items():
        df.to_csv(tmp_path / RAW_FILES[name], index=False)
    return tmp_path
