from olist.drift import adversarial_validation, monthly_late_rate
from olist.features import build_features


def test_drift_report_runs(tables):
    df = build_features(tables)
    monthly = monthly_late_rate(df)
    assert monthly["orders"].sum() == len(df)
    auc, drift = adversarial_validation(df)
    assert 0 <= auc <= 1
    assert len(drift) > 0
    assert drift["share_of_drift"].between(0, 1).all()
