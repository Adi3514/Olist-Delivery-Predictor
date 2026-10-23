import numpy as np

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
