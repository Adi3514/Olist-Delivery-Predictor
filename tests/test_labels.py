import pandas as pd
import pytest

from olist.labels import build_labels


def make_orders(rows):
    df = pd.DataFrame(rows, columns=["order_id", "order_status", "purchase", "carrier", "delivered", "estimated"])
    return pd.DataFrame({
        "order_id": df["order_id"],
        "order_status": df["order_status"],
        "order_purchase_timestamp": pd.to_datetime(df["purchase"], format="ISO8601"),
        "order_delivered_carrier_date": pd.to_datetime(df["carrier"], format="ISO8601"),
        "order_delivered_customer_date": pd.to_datetime(df["delivered"], format="ISO8601"),
        "order_estimated_delivery_date": pd.to_datetime(df["estimated"], format="ISO8601"),
    })


ORDERS = make_orders([
    ("on_time", "delivered", "2018-01-01", "2018-01-03", "2018-01-10 15:00", "2018-01-10"),
    ("late", "delivered", "2018-01-01", "2018-01-03", "2018-01-11 09:00", "2018-01-10"),
    ("lost_in_transit", "shipped", "2018-01-01", "2018-01-03", None, "2018-01-20"),
    ("never_shipped", "processing", "2018-01-01", None, None, "2018-01-20"),
    ("canceled", "canceled", "2018-01-01", None, None, "2018-01-20"),
    ("still_open", "shipped", "2018-03-01", "2018-03-03", None, "2018-04-01"),
    ("fast_recent", "delivered", "2018-03-01", "2018-03-02", "2018-03-05", "2018-04-01"),
    ("last", "delivered", "2018-02-01", "2018-02-03", "2018-03-10", "2018-03-01"),
])


def labels_by_id(**kwargs):
    return build_labels(ORDERS, **kwargs).set_index("order_id")


def test_same_day_delivery_is_on_time():
    labels = labels_by_id()
    assert labels.loc["on_time", "is_late"] == 0
    assert labels.loc["late", "is_late"] == 1


def test_shipped_but_never_delivered_counts_as_late():
    labels = labels_by_id()
    assert labels.loc["lost_in_transit", "is_late"] == 1
    assert labels.loc["lost_in_transit", "label_source"] == "lost_in_transit"


def test_never_shipped_orders_are_dropped_by_default():
    assert "never_shipped" not in labels_by_id().index


def test_corrected_all_counts_never_shipped_as_late():
    labels = labels_by_id(mode="corrected_all")
    assert labels.loc["never_shipped", "is_late"] == 1
    assert labels.loc["never_shipped", "label_source"] == "never_shipped"


def test_canceled_and_unresolved_orders_are_dropped():
    labels = labels_by_id(mode="corrected_all")
    assert "canceled" not in labels.index
    # Promise date after the end of the data: outcome not known yet, for slow and fast orders alike
    assert "still_open" not in labels.index
    assert "fast_recent" not in labels.index


def test_naive_labels_keep_only_delivered_orders():
    labels = labels_by_id(mode="naive")
    assert set(labels.index) == {"on_time", "late", "fast_recent", "last"}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_labels(ORDERS, mode="strict")
