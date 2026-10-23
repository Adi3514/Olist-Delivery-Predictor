from olist.features import build_features
from olist.train import MODEL_NAMES, backtest_months, run, time_split


def test_time_split_keeps_test_after_train(tables):
    df = build_features(tables)
    train, test = time_split(df, test_months=3)
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
    assert test["order_purchase_timestamp"].dt.to_period("M").nunique() == 3


def test_training_only_sees_outcomes_known_before_the_test_period(tables):
    df = build_features(tables)
    train, test = time_split(df, test_months=3)
    test_start = test["order_purchase_timestamp"].dt.to_period("M").min().start_time
    assert (train["known_at"] < test_start).all()


def test_models_learn_the_slow_sellers(tables):
    df = build_features(tables)
    months = backtest_months(df, n_months=3, min_orders=50)
    assert len(months) == 3
    results, model = run(df, months=months)
    for model_name in MODEL_NAMES:
        for split in ["random_split", "out_of_time"]:
            assert results[model_name][split]["auc"] > 0.6
        backtest = results[model_name]["backtest"]
        assert len(backtest["months"]) == 3
        assert backtest["min_auc"] <= backtest["mean_auc"]
    assert model is not None
