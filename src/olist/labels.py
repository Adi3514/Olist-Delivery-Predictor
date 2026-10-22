"""Build the late-delivery label for each order.

An order is late when it reaches the customer on a later calendar day than the
promised (estimated) delivery date.

The naive label keeps only orders with a delivery date. That silently drops late
orders that were never delivered, and near the end of the data it keeps the fast
deliveries while the slow ones from the same weeks have not arrived yet. Both make
recent months look better than they were. The corrected label:

* counts an order that was never delivered, and whose promise date has already
  passed when the data ends, as late;
* keeps only orders whose promise date falls before the end of the data, so every
  kept order has a known outcome;
* drops canceled and unavailable orders, which never had a delivery to be late.
"""
import pandas as pd

NO_DELIVERY_STATUSES = {"canceled", "unavailable"}


def snapshot_date(orders: pd.DataFrame) -> pd.Timestamp:
    """The moment the data was exported: the latest timestamp we can see."""
    return max(
        orders["order_purchase_timestamp"].max(),
        orders["order_delivered_customer_date"].max(),
    )


def build_labels(orders: pd.DataFrame, corrected: bool = True) -> pd.DataFrame:
    """Return order_id, is_late and label_source for orders with a usable outcome."""
    delivered_day = orders["order_delivered_customer_date"].dt.normalize()
    promised_day = orders["order_estimated_delivery_date"].dt.normalize()
    has_delivery = orders["order_delivered_customer_date"].notna()

    out = pd.DataFrame({"order_id": orders["order_id"]})
    out["is_late"] = (delivered_day > promised_day).astype(int)
    out["label_source"] = "delivered"

    if not corrected:
        return out[has_delivery].reset_index(drop=True)

    snapshot = snapshot_date(orders)
    no_delivery_expected = orders["order_status"].isin(NO_DELIVERY_STATUSES)
    observable = orders["order_estimated_delivery_date"] <= snapshot

    overdue = ~has_delivery & ~no_delivery_expected & observable
    out.loc[overdue, "is_late"] = 1
    out.loc[overdue, "label_source"] = "never_delivered"

    keep = observable & ~no_delivery_expected & (has_delivery | overdue)
    return out[keep].reset_index(drop=True)
