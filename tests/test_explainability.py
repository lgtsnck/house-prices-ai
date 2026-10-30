"""Тесты интерпретации, не требующие исходных данных Kaggle."""
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier

from house_prices_ai.preprocessing import HouseFeatures, _make_preprocessor
from house_prices_ai.explainability import (
    _xgb_contributions, _feature_importance, _select_regression_case,
)


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(22)
        n = 60
        cls.X = pd.DataFrame({
            "GrLivArea": rng.uniform(550, 3000, n),
            "OverallQual": rng.integers(3, 10, n),
            "LotArea": rng.uniform(2000, 12000, n),
            "Neighborhood": rng.choice(["Edwards", "NAmes", "CollgCr"], n),
            "PoolQC": rng.choice(["Gd", None], n, p=[.1, .9]),
            "PoolArea": rng.choice([0, 250], n, p=[.9, .1]),
        })
        raw_features = HouseFeatures().transform(cls.X)
        cls.preprocessing = _make_preprocessor(raw_features)
        y = (cls.X["GrLivArea"] * 60 + cls.X["OverallQual"] * 4000).to_numpy()
        cls.reg = Pipeline([
            ("preprocessing", _make_preprocessor(raw_features)),
            ("model", XGBRegressor(n_estimators=15, max_depth=2, n_jobs=1, random_state=22)),
        ]).fit(cls.X, y)
        cls.clf = Pipeline([
            ("preprocessing", _make_preprocessor(raw_features)),
            ("model", XGBClassifier(n_estimators=15, max_depth=2, n_jobs=1, random_state=22,
                                    objective="binary:logistic")),
        ]).fit(cls.X, (y >= np.median(y)).astype(int))

    def test_regression_shap_reconstructs_price(self):
        _, names, shap, base, pred, margin = _xgb_contributions(self.reg, self.X.iloc[:8], "regression")
        self.assertEqual(shap.shape, (8, len(names)))
        np.testing.assert_allclose(base + shap.sum(axis=1), pred, rtol=1e-4, atol=.002)

    def test_classifier_shap_reconstructs_probability(self):
        from scipy.special import expit
        _, names, shap, base, pred, margin = _xgb_contributions(self.clf, self.X.iloc[:8], "classification")
        self.assertEqual(shap.shape, (8, len(names)))
        np.testing.assert_allclose(expit(base + shap.sum(axis=1)), pred, atol=2e-5)

    def test_importance_has_matching_features(self):
        table = _feature_importance(self.reg, self.X)
        self.assertFalse(table.empty)
        self.assertIn("Feature", table)
        self.assertIn("Importance", table)
        self.assertGreaterEqual(table.Importance.min(), 0.)

    def test_special_case_is_selected_by_id(self):
        frame = self.X.iloc[:4].copy()
        frame.index = [2, 13, 1424, 26]
        target = pd.Series([130_000, 120_000, 274_970, 190_000], index=frame.index)
        self.assertEqual(_select_regression_case(frame, target), 2)


if __name__ == "__main__":
    unittest.main()
