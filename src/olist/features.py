"""Build one row per order with features known at the moment of purchase.

History features (a seller's past late rate, and so on) only count orders whose
outcome was already known before this order was placed, so nothing leaks from
the future.

Usage: python -m olist.features [--label-mode corrected|corrected_all|naive]
"""
import argparse

import numpy as np
import pandas as pd

from olist.config import RAW_DIR, features_file
from olist.labels import LABEL_MODES, build_labels
from olist.load import load_all

CATEGORICAL = ["customer_state", "seller_state", "product_category", "payment_type"]
HISTORY_PRIOR_WEIGHT = 10  # pseudo-orders that pull small-sample rates toward the overall rate
BACKLOG_DAYS = 7


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    return (
        geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def item_features(items: pd.DataFrame, products: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    products = products.merge(categories, on="product_category_name", how="left")
    products["volume_cm3"] = products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    items = items.merge(
        products[["product_id", "product_category_name_english", "product_weight_g", "volume_cm3", "product_photos_qty"]],
        on="product_id",
        how="left",
    )
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        shipping_limit=("shipping_limit_date", "max"),
    )
    agg["freight_ratio"] = agg["total_freight"] / agg["total_price"].replace(0, np.nan)
    # The most expensive item decides the order's seller and category
    main = items.sort_values("price", ascending=False).drop_duplicates("order_id").set_index("order_id")
    agg["seller_id"] = main["seller_id"]
    agg["product_category"] = main["product_category_name_english"]
    return agg.reset_index()


def payment_features(payments: pd.DataFrame) -> pd.DataFrame:
    main = payments.sort_values("payment_value", ascending=False).drop_duplicates("order_id").set_index("order_id")
    agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    )
    agg["payment_type"] = main["payment_type"]
    return agg.reset_index()


def past_late_rate(events: pd.DataFrame, queries: pd.DataFrame, key: str | None, name: str) -> pd.DataFrame:
    """For each query order, the late rate of `key`'s orders whose outcome was known before purchase.

    events: key, known_at, is_late (one row per labeled order)
    queries: order_id, key, order_purchase_timestamp
    """
    by = [key] if key else []
    ev = events.dropna(subset=by + ["known_at"]).sort_values("known_at").copy()
    grouped = ev.groupby(key) if key else ev
    ev["n_known"] = grouped.cumcount() + 1 if key else np.arange(1, len(ev) + 1)
    ev["n_late"] = grouped["is_late"].cumsum()

    q = queries.dropna(subset=by).sort_values("order_purchase_timestamp")
    merged = pd.merge_asof(
        q,
        ev[by + ["known_at", "n_known", "n_late"]],
        left_on="order_purchase_timestamp",
        right_on="known_at",
        by=key,
        allow_exact_matches=False,
    )
    merged[["n_known", "n_late"]] = merged[["n_known", "n_late"]].fillna(0)
    out = merged[["order_id", "n_known", "n_late"]].rename(
        columns={"n_known": f"{name}_past_orders", "n_late": f"{name}_past_late"}
    )
    return out


def add_history_features(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    events = df[["order_id", "seller_id", "customer_state", "route", "order_purchase_timestamp"]].merge(
        labels[["order_id", "is_late", "known_at"]], on="order_id"
    )
    queries = df[["order_id", "seller_id", "customer_state", "route", "order_purchase_timestamp"]]

    overall = past_late_rate(events, queries, None, "overall")
    overall["prior"] = overall["overall_past_late"] / overall["overall_past_orders"].replace(0, np.nan)
    prior = overall.set_index("order_id")["prior"]

    for key, name in [("seller_id", "seller"), ("customer_state", "customer_state"), ("route", "route")]:
        hist = past_late_rate(events, queries, key, name).set_index("order_id")
        n, late = hist[f"{name}_past_orders"], hist[f"{name}_past_late"]
        p = prior.reindex(hist.index)
        hist[f"{name}_past_late_rate"] = (late + HISTORY_PRIOR_WEIGHT * p) / (n + HISTORY_PRIOR_WEIGHT)
        df = df.merge(hist[[f"{name}_past_orders", f"{name}_past_late_rate"]], on="order_id", how="left")
    return df


def seller_backlog(df: pd.DataFrame) -> pd.Series:
    """Orders the same seller received in the BACKLOG_DAYS before this one."""
    window = pd.Timedelta(days=BACKLOG_DAYS)
    out = pd.Series(0, index=df.index, dtype=float)
    for _, grp in df.dropna(subset=["seller_id"]).groupby("seller_id"):
        ts = grp["order_purchase_timestamp"].sort_values()
        values = ts.values
        start = np.searchsorted(values, values - window, side="left")
        end = np.searchsorted(values, values, side="left")
        out.loc[ts.index] = end - start
    return out


def build_features(tables: dict[str, pd.DataFrame], label_mode: str = "corrected") -> pd.DataFrame:
    orders = tables["orders"]
    labels = build_labels(orders, mode=label_mode)
    labels = labels.merge(
        orders[["order_id", "order_delivered_customer_date", "order_estimated_delivery_date"]], on="order_id"
    )
    # When the outcome became visible: the delivery day, or the promise day for never-delivered orders
    labels["known_at"] = labels["order_delivered_customer_date"].fillna(labels["order_estimated_delivery_date"])

    df = orders[["order_id", "customer_id", "order_purchase_timestamp", "order_estimated_delivery_date"]].copy()
    df = df.merge(tables["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]], on="customer_id", how="left")
    df = df.merge(item_features(tables["items"], tables["products"], tables["categories"]), on="order_id", how="left")
    df = df.merge(tables["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")
    df = df.merge(payment_features(tables["payments"]), on="order_id", how="left")

    coords = zip_coordinates(tables["geolocation"])
    cust = coords.reindex(df["customer_zip_code_prefix"]).to_numpy()
    sell = coords.reindex(df["seller_zip_code_prefix"]).to_numpy()
    df["customer_lat"], df["customer_lng"] = cust[:, 0], cust[:, 1]
    df["seller_lat"], df["seller_lng"] = sell[:, 0], sell[:, 1]
    df["distance_km"] = haversine_km(df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["route"] = df["seller_state"] + "->" + df["customer_state"]

    purchase = df["order_purchase_timestamp"]
    df["purchase_month"] = purchase.dt.month
    df["purchase_dow"] = purchase.dt.day_of_week
    df["purchase_hour"] = purchase.dt.hour
    df["promised_days"] = (df["order_estimated_delivery_date"] - purchase).dt.total_seconds() / 86400
    df["shipping_limit_days"] = (df["shipping_limit"] - purchase).dt.total_seconds() / 86400
    df["seller_backlog_7d"] = seller_backlog(df)

    df = add_history_features(df, labels)
    df = df.merge(labels[["order_id", "is_late", "label_source"]], on="order_id", how="inner")

    df = df.drop(columns=["customer_id", "customer_zip_code_prefix", "seller_zip_code_prefix", "shipping_limit", "route"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


ID_COLUMNS = ["order_id", "seller_id", "order_purchase_timestamp", "order_estimated_delivery_date", "label_source"]
TARGET = "is_late"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="corrected",
                        help="How to label orders that never arrived (see olist/labels.py)")
    args = parser.parse_args()

    df = build_features(load_all(RAW_DIR), label_mode=args.label_mode)
    path = features_file(args.label_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sources = df.loc[df[TARGET] == 1, "label_source"].value_counts()
    print(
        f"Saved {len(df):,} orders x {len(feature_columns(df))} features to {path}\n"
        f"Late rate {df[TARGET].mean():.1%}. Late orders by source:"
    )
    for source, count in sources.items():
        print(f"  {source:16s} {count:,}")


if __name__ == "__main__":
    main()
