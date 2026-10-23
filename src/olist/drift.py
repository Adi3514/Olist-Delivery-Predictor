"""Show how the data shifts over time.

1. Late rate by month, to see the waves.
2. Adversarial validation: a model tries to tell old orders from the last
   TEST_MONTHS months using the features alone. An AUC near 0.5 means the two
   periods look alike; near 1.0 means they are easy to tell apart. The features it
   leans on most are the ones that drift.

Usage: python -m olist.drift [--label-mode corrected|corrected_all|naive]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from olist.config import REPORTS_DIR, TEST_MONTHS, features_file
from olist.features import CATEGORICAL, TARGET, feature_columns
from olist.labels import LABEL_MODES

SEED = 42
TOP_N = 12


def monthly_late_rate(df: pd.DataFrame) -> pd.DataFrame:
    month = df["order_purchase_timestamp"].dt.to_period("M")
    return df.groupby(month)[TARGET].agg(orders="size", late_rate="mean")


def adversarial_validation(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    """Return the AUC of telling recent orders from old ones, and per-feature drift."""
    month = df["order_purchase_timestamp"].dt.to_period("M")
    is_recent = (month >= month.max() - test_months + 1).astype(int)
    features = feature_columns(df)

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               random_state=SEED, verbose=-1, importance_type="gain")
    scores = cross_val_predict(model, df[features], is_recent, cv=3, method="predict_proba")[:, 1]
    auc = roc_auc_score(is_recent, scores)

    model.fit(df[features], is_recent)
    importance = pd.Series(model.feature_importances_, index=features)
    rows = []
    for feature in importance.sort_values(ascending=False).index[:TOP_N]:
        old, new = df.loc[is_recent == 0, feature], df.loc[is_recent == 1, feature]
        if feature in CATEGORICAL:
            old_value, new_value = old.mode().iloc[0], new.mode().iloc[0]
        else:
            old_value, new_value = round(float(old.median()), 3), round(float(new.median()), 3)
        rows.append({
            "feature": feature,
            "share_of_drift": round(float(importance[feature] / importance.sum()), 3),
            "old_typical": old_value,
            "recent_typical": new_value,
        })
    return auc, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="corrected")
    args = parser.parse_args()

    path = features_file(args.label_mode)
    if not path.exists():
        raise SystemExit(f"No feature table at {path}. Run `python -m olist.features` first.")
    df = pd.read_parquet(path)

    monthly = monthly_late_rate(df)
    print("Late rate by month of purchase:")
    for month, row in monthly.iterrows():
        bar = "#" * int(np.round(row["late_rate"] * 100))
        print(f"  {month}  {int(row['orders']):6,d} orders  {row['late_rate']:6.1%}  {bar}")

    auc, drift = adversarial_validation(df)
    print(f"\nOld vs last {TEST_MONTHS} months: a model tells them apart with AUC {auc:.3f}")
    print("(0.5 = the periods look the same, 1.0 = completely different)\n")
    print("Features that drift most:")
    print(drift.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"drift_{args.label_mode}_labels.json"
    out.write_text(json.dumps({
        "adversarial_auc": round(auc, 4),
        "monthly": {str(m): {"orders": int(r["orders"]), "late_rate": round(float(r["late_rate"]), 4)}
                    for m, r in monthly.iterrows()},
        "top_drifting_features": drift.astype(str).to_dict(orient="records"),
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
