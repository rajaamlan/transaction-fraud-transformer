"""Submission checks run without GPU, torch, or a downloaded dataset."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("submission", Path(__file__).parents[1] / "scripts/kaggle_submission.py")
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame({"TransactionID": [10, 20], "isFraud": [0.5, 0.5]})

    def test_valid_roundtrip(self):
        submission.validate_submission(self.sample.copy(), self.sample)

    def test_rejects_bad_predictions_and_alignment(self):
        for values in ([np.nan, 0.5], [np.inf, 0.5], [-0.1, 0.5], [0.5, 1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                submission.validate_submission(self.sample.assign(isFraud=values), self.sample)
        for broken in (self.sample.iloc[::-1], self.sample.iloc[:1],
                       self.sample.assign(TransactionID=[10, 10]),
                       self.sample.rename(columns={"isFraud": "score"})):
            with self.assertRaises(ValueError):
                submission.validate_submission(broken, self.sample)

    def test_identity_rename_missing_values_and_unknown_category(self):
        class Scaler:
            def transform(self, frame):
                return frame.to_numpy()
        class Encoder:
            def transform(self, frame):
                return np.array([[0 if value == "known" else -1] for value in frame["id_30"]])
        pre = {"numeric_features": ["amount"], "categorical_features": ["id_30"],
               "numeric_medians": pd.Series({"amount": 2.0}), "scaler": Scaler(), "encoder": Encoder()}
        rows = pd.DataFrame({"TransactionID": [20, 10], "amount": [np.inf, 3.]})
        identity = pd.DataFrame({"TransactionID": [10, 20], "id-30": ["known", "new"]})
        xn, xc = submission.prepare_test_chunk(rows, identity, pre)
        np.testing.assert_array_equal(xn[:, 0], [2., 3.])
        np.testing.assert_array_equal(xc[:, 0], [0, 1])
        with self.assertRaises(ValueError):
            submission.prepare_test_chunk(rows.drop(columns="amount"), identity, pre)


if __name__ == "__main__":
    unittest.main()
