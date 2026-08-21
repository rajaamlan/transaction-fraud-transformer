import pandas as pd

from fraud_model.features import FeaturePreprocessor


def test_preprocessor_encodes_unknown_category():
    frame = pd.DataFrame(
        [
            {
                "amount": 10,
                "hour": 10,
                "merchant_risk": 0.1,
                "customer_age_days": 100,
                "transactions_last_24h": 1,
                "country": "US",
                "merchant_category": "grocery",
                "payment_method": "card",
            }
        ]
    )
    preprocessor = FeaturePreprocessor.fit(frame)
    new_frame = frame.copy()
    new_frame.loc[0, "country"] = "ZZ"

    numeric, categorical = preprocessor.transform(new_frame)

    assert numeric.shape == (1, 5)
    assert categorical.shape == (1, 3)
    assert categorical[0, 0] == 0
