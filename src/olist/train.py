"""Train and compare the baseline models.

Each model is scored two ways:
* random split: train and test orders mixed from all months (optimistic);
* out-of-time: train on older months, test on the last TEST_MONTHS months (realistic).

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


def time_split(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    month = df["order_purchase_timestamp"].dt.to_period("M")
    cutoff = month.max() - test_months + 1
    return df[month < cutoff], df[month >= cutoff]


def random_split(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])


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


def as_lr_input(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL:
        if col in X:
            X[col] = X[col].astype(object)
    return X


def evaluate(model_name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    if model_name == "logistic_regression":
        model = logistic_model(features)
        model.fit(as_lr_input(train[features]), train[TARGET])
        scores = model.predict_proba(as_lr_input(test[features]))[:, 1]
    else:
        model = lightgbm_model()
        model.fit(train[features], train[TARGET])
        scores = model.predict_proba(test[features])[:, 1]
    metrics = {
        "auc": round(roc_auc_score(test[TARGET], scores), 4),
        "pr_auc": round(average_precision_score(test[TARGET], scores), 4),
        "test_late_rate": round(float(test[TARGET].mean()), 4),
        "n_train": len(train),
        "n_test": len(test),
    }
    return model, metrics


def run(df: pd.DataFrame) -> tuple[dict, lgb.LGBMClassifier]:
    features = feature_columns(df)
    results = {}
    final_model = None
    for split_name, (train, test) in {"random_split": random_split(df), "out_of_time": time_split(df)}.items():
        for model_name in ["logistic_regression", "lightgbm"]:
            model, metrics = evaluate(model_name, train, test, features)
            results.setdefault(model_name, {})[split_name] = metrics
            if model_name == "lightgbm" and split_name == "out_of_time":
                final_model = model
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
    model.booster_.save_model(str(MODELS_DIR / f"lightgbm_out_of_time_{args.label_mode}.txt"))

    print(f"Labels: {args.label_mode}")
    for model_name, splits in results.items():
        for split_name, m in splits.items():
            print(f"  {model_name:20s} {split_name:13s} AUC {m['auc']:.3f}  PR-AUC {m['pr_auc']:.3f}")
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
