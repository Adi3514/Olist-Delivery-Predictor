from olist.features import build_features
from olist.train import run, time_split


def test_time_split_keeps_test_after_train(tables):
    df = build_features(tables)
    train, test = time_split(df, test_months=3)
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
    assert test["order_purchase_timestamp"].dt.to_period("M").nunique() == 3


def test_models_learn_the_slow_sellers(tables):
    results, model = run(build_features(tables))
    for model_name in ["logistic_regression", "lightgbm"]:
        for split in ["random_split", "out_of_time"]:
            assert results[model_name][split]["auc"] > 0.6
    assert model is not None
