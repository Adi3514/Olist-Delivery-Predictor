import numpy as np
import pandas as pd

from olist.features import build_features, feature_columns, haversine_km
from olist.load import load_all


def test_haversine_sao_paulo_to_rio():
    assert 340 < haversine_km(-23.55, -46.63, -22.91, -43.17) < 370


def test_one_row_per_labeled_order(tables):
    df = build_features(tables)
    assert df["order_id"].is_unique
    assert set(df["is_late"].unique()) <= {0, 1}
    assert len(feature_columns(df)) >= 25


def test_history_uses_only_the_past(tables):
    df = build_features(tables)
    first = df.iloc[0]
    assert first["seller_past_orders"] == 0
    # Past counts never exceed the number of orders placed earlier
    earlier = np.arange(len(df))
    assert (df["seller_past_orders"].to_numpy() <= earlier).all()


def test_slow_sellers_get_higher_past_late_rate(tables):
    df = build_features(tables)
    recent = df[df["order_purchase_timestamp"] > df["order_purchase_timestamp"].quantile(0.8)]
    slow = recent["seller_id"].isin([f"s{i}" for i in range(5)])
    assert recent.loc[slow, "seller_past_late_rate"].mean() > recent.loc[~slow, "seller_past_late_rate"].mean()


def test_loads_from_csv(raw_dir):
    tables = load_all(raw_dir)
    assert tables["customers"]["customer_zip_code_prefix"].str.len().eq(5).all()
    assert len(build_features(tables)) > 0


def test_label_modes_change_which_orders_are_kept(tables):
    default = build_features(tables)
    everything = build_features(tables, label_mode="corrected_all")
    assert "never_shipped" not in set(default["label_source"])
    assert "never_shipped" in set(everything["label_source"])
    assert len(everything) > len(default)


def test_future_outcomes_do_not_change_past_features(tables):
    cutoff = tables["orders"]["order_purchase_timestamp"].quantile(0.6)
    changed = {name: df.copy() for name, df in tables.items()}
    orders = changed["orders"]
    later = orders["order_purchase_timestamp"] >= cutoff
    # Make every later order very late; nothing about earlier orders may change
    orders.loc[later, "order_delivered_customer_date"] = orders.loc[later, "order_estimated_delivery_date"] + pd.Timedelta(days=30)
    orders.loc[later, "order_status"] = "delivered"

    before = build_features(tables).set_index("order_id")
    after = build_features(changed).set_index("order_id")
    early = before.index[before["order_purchase_timestamp"] < cutoff]
    cols = feature_columns(before)
    pd.testing.assert_frame_equal(before.loc[early, cols], after.loc[early, cols])


def test_recent_features_are_valid(tables):
    df = build_features(tables)
    assert df["overall_recent_late_rate"].dropna().between(0, 1).all()
    assert df["promise_slack_days"].notna().mean() > 0.9
