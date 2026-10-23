"""Build the late-delivery label for each order.

An order is late when it reaches the customer on a later calendar day than the
promised (estimated) delivery date.

The naive label keeps only orders with a delivery date. That silently drops late
orders that were never delivered, and near the end of the data it keeps the fast
deliveries while the slow ones from the same weeks have not arrived yet. Both make
recent months look better than they were. The corrected label:

* counts an order that was handed to the carrier but never delivered, and whose
  promise date has already passed when the data ends, as late;
* keeps only orders whose promise date falls before the end of the data, so every
  kept order has a known outcome;
* drops canceled and unavailable orders, which never had a delivery to be late;
* drops orders that never reached the carrier (stuck in "processing" or
  "invoiced"). Those are seller fulfilment failures, closer to cancellations than
  to slow deliveries. The "corrected_all" mode counts them as late instead.
"""
import pandas as pd

NO_DELIVERY_STATUSES = {"canceled", "unavailable"}
LABEL_MODES = ("corrected", "corrected_all", "naive")


def snapshot_date(orders: pd.DataFrame) -> pd.Timestamp:
    """The moment the data was exported: the latest timestamp we can see."""
    return max(
        orders["order_purchase_timestamp"].max(),
        orders["order_delivered_customer_date"].max(),
    )


def build_labels(orders: pd.DataFrame, mode: str = "corrected") -> pd.DataFrame:
    """Return order_id, is_late and label_source for orders with a usable outcome.

    label_source is "delivered", "lost_in_transit" (shipped, never delivered) or
    "never_shipped" (only kept in "corrected_all" mode).
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"mode must be one of {LABEL_MODES}, got {mode!r}")

    delivered_day = orders["order_delivered_customer_date"].dt.normalize()
    promised_day = orders["order_estimated_delivery_date"].dt.normalize()
    has_delivery = orders["order_delivered_customer_date"].notna()

    out = pd.DataFrame({"order_id": orders["order_id"]})
    out["is_late"] = (delivered_day > promised_day).astype(int)
    out["label_source"] = "delivered"

    if mode == "naive":
        return out[has_delivery].reset_index(drop=True)

    snapshot = snapshot_date(orders)
    no_delivery_expected = orders["order_status"].isin(NO_DELIVERY_STATUSES)
    observable = orders["order_estimated_delivery_date"] <= snapshot
    shipped = orders["order_delivered_carrier_date"].notna()

    overdue = ~has_delivery & ~no_delivery_expected & observable
    out.loc[overdue, "is_late"] = 1
    out.loc[overdue & shipped, "label_source"] = "lost_in_transit"
    out.loc[overdue & ~shipped, "label_source"] = "never_shipped"

    keep = observable & ~no_delivery_expected & (has_delivery | overdue)
    if mode == "corrected":
        keep &= out["label_source"] != "never_shipped"
    return out[keep].reset_index(drop=True)
