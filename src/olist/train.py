"""Train and compare the models.

Each model is scored three ways:
* random split: train and test orders mixed from all months (optimistic);
* out-of-time: train on older months, test on the last TEST_MONTHS months (realistic);
* backtest: one test per month over the last BACKTEST_MONTHS months, each time
  training on everything known before that month. The mean is the headline number,
  and the spread shows how much a model swings from one wave to the next.

Training sets only contain orders whose outcome was known before the test period
starts, exactly as it would be in production.

Usage: python -m olist.train [--label-mode corrected|corrected_all|naive]
"""
import argparse
import json

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from olist.config import MODELS_DIR, RAW_DIR, REPORTS_DIR, TEST_MONTHS, features_file
from olist.features import CATEGORICAL, TARGET, build_features, feature_columns
from olist.labels import LABEL_MODES
from olist.load import load_all

SEED = 42
BACKTEST_MONTHS = 6
MIN_TEST_ORDERS = 500
RECENCY_HALF_LIFE_DAYS = 120

# Features that mostly track the calendar rather than the order: month of year (only two
# years of data), raw coordinates (stand-ins for state) and running order counts, which
# only ever grow. The drift-resistant model leaves them out.
DRIFT_PRONE = [
    "purchase_month",
    "customer_lat", "customer_lng", "seller_lat", "seller_lng",
    "seller_past_orders", "customer_state_past_orders", "route_past_orders",
]
MODEL_NAMES = ["logistic_regression", "lightgbm", "lightgbm_stable"]


def known_before(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    return df[df["known_at"] < start]


def time_split(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    month = df["order_purchase_timestamp"].dt.to_period("M")
    cutoff = month.max() - test_months + 1
    return known_before(df[month < cutoff], cutoff.start_time), df[month >= cutoff]


def random_split(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])


def backtest_months(df: pd.DataFrame, n_months: int = BACKTEST_MONTHS, min_orders: int = MIN_TEST_ORDERS):
    counts = df["order_purchase_timestamp"].dt.to_period("M").value_counts().sort_index()
    return list(counts[counts >= min_orders].index[-n_months:])


def logistic_model(features: list[str]) -> Pipeline:
    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), categorical),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=2000, class_weight="balanced"))


def lightgbm_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )


def lightgbm_stable_model() -> lgb.LGBMClassifier:
    """Smaller, more regularized trees, so the model can't memorise one period."""
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        reg_lambda=5.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        random_state=SEED,
        verbose=-1,
    )


def recency_weights(train: pd.DataFrame) -> pd.Series:
    """Halve an order's weight for every RECENCY_HALF_LIFE_DAYS it is older than the newest order."""
    age_days = (train["order_purchase_timestamp"].max() - train["order_purchase_timestamp"]).dt.days
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def model_features(model_name: str, df: pd.DataFrame) -> list[str]:
    features = feature_columns(df)
    if model_name == "lightgbm_stable":
        features = [f for f in features if f not in DRIFT_PRONE]
    return features


def as_lr_input(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL:
        if col in X:
            X[col] = X[col].astype(object)
    return X


def evaluate(model_name: str, train: pd.DataFrame, test: pd.DataFrame):
    features = model_features(model_name, train)
    if model_name == "logistic_regression":
        model = logistic_model(features)
        model.fit(as_lr_input(train[features]), train[TARGET])
        scores = model.predict_proba(as_lr_input(test[features]))[:, 1]
    elif model_name == "lightgbm":
        model = lightgbm_model()
        model.fit(train[features], train[TARGET])
        scores = model.predict_proba(test[features])[:, 1]
    else:
        model = lightgbm_stable_model()
        model.fit(train[features], train[TARGET], sample_weight=recency_weights(train))
        scores = model.predict_proba(test[features])[:, 1]
    metrics = {
        "auc": round(roc_auc_score(test[TARGET], scores), 4),
        "pr_auc": round(average_precision_score(test[TARGET], scores), 4),
        "test_late_rate": round(float(test[TARGET].mean()), 4),
        "n_train": len(train),
        "n_test": len(test),
    }
    return model, metrics


def backtest(df: pd.DataFrame, model_name: str, months) -> dict:
    period = df["order_purchase_timestamp"].dt.to_period("M")
    per_month = {}
    for month in months:
        train = known_before(df[period < month], month.start_time)
        test = df[period == month]
        if test[TARGET].nunique() < 2:
            continue
        _, metrics = evaluate(model_name, train, test)
        per_month[str(month)] = metrics
    aucs = pd.Series([m["auc"] for m in per_month.values()])
    pr_aucs = pd.Series([m["pr_auc"] for m in per_month.values()])
    return {
        "mean_auc": round(float(aucs.mean()), 4),
        "min_auc": round(float(aucs.min()), 4),
        "mean_pr_auc": round(float(pr_aucs.mean()), 4),
        "months": per_month,
    }


def run(df: pd.DataFrame, months=None) -> tuple[dict, lgb.LGBMClassifier]:
    months = backtest_months(df) if months is None else months
    results = {}
    final_model = None
    splits = {"random_split": random_split(df), "out_of_time": time_split(df)}
    for model_name in MODEL_NAMES:
        for split_name, (train, test) in splits.items():
            model, metrics = evaluate(model_name, train, test)
            results.setdefault(model_name, {})[split_name] = metrics
            if model_name == "lightgbm_stable" and split_name == "out_of_time":
                final_model = model
        results[model_name]["backtest"] = backtest(df, model_name, months)
    return results, final_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="corrected",
                        help="Which late label to train on (see olist/labels.py)")
    args = parser.parse_args()

    path = features_file(args.label_mode)
    if path.exists():
        df = pd.read_parquet(path)
    else:
        print(f"No feature table for '{args.label_mode}' labels yet, building it...")
        df = build_features(load_all(RAW_DIR), label_mode=args.label_mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    results, model = run(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"metrics_{args.label_mode}_labels.json"
    out.write_text(json.dumps(results, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / f"lightgbm_stable_{args.label_mode}.txt"))

    print(f"Labels: {args.label_mode}\n")
    print(f"{'model':20s} {'random AUC':>10s} {'last-3m AUC':>11s} {'backtest AUC':>12s} {'worst month':>11s} {'backtest PR-AUC':>15s}")
    for model_name, r in results.items():
        bt = r["backtest"]
        print(f"{model_name:20s} {r['random_split']['auc']:10.3f} {r['out_of_time']['auc']:11.3f} "
              f"{bt['mean_auc']:12.3f} {bt['min_auc']:11.3f} {bt['mean_pr_auc']:15.3f}")
    print("\nBacktest AUC by month:")
    months = list(next(iter(results.values()))["backtest"]["months"])
    print(f"{'month':10s}" + "".join(f"{m:>22s}" for m in results))
    for month in months:
        row = "".join(f"{results[m]['backtest']['months'][month]['auc']:22.3f}" for m in results)
        late = results["lightgbm"]["backtest"]["months"][month]["test_late_rate"]
        print(f"{month:10s}{row}   late rate {late:.1%}")
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
