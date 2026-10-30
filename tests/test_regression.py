"""Базовые проверки ЛР №3 на синтетических данных (без Kaggle CSV)."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from house_prices_ai.evaluation import regression_metrics, save_mlp_training_diagnostics
from house_prices_ai.regression import cv_compare, make_regression_pipeline


class TestRegression(unittest.TestCase):
    @staticmethod
    def fake_raw(n=75):
        rng = np.random.default_rng(42)
        area = rng.uniform(600, 3200, n)
        X = pd.DataFrame({
            "GrLivArea": area,
            "TotalBsmtSF": area*.42,
            "1stFlrSF": area*.65,
            "2ndFlrSF": area*.35,
            "GarageArea": rng.choice([0, 250, 400], n),
            "PoolArea": rng.choice([0, 0, 0, 240], n),
            "PoolQC": rng.choice([None, "Gd", "Ex"], n),
            "MSSubClass": rng.choice([20, 60], n),
            "OverallQual": rng.integers(2, 10, n),
            "Neighborhood": rng.choice(["NAmes", "Edwards"], n),
        })
        y = pd.Series(42000 + 70*area + 7000*X["OverallQual"] + rng.normal(0, 15000, n))
        return X, y

    def test_metrics_are_in_usd(self):
        m = regression_metrics([100, 200], [110, 190])
        self.assertAlmostEqual(m["MAE"], 10.)
        self.assertAlmostEqual(m["RMSE"], 10.)

    def test_pipeline_has_independent_fit_and_valid_cv(self):
        X, y = self.fake_raw()
        pipe = make_regression_pipeline(X, Ridge(alpha=10))
        self.assertEqual(pipe.named_steps.keys(), {"preprocessing", "model"})
        summary, folds = cv_compare(X, y, KFold(n_splits=3, shuffle=True, random_state=42),
                                     {"DummyRegressor": DummyRegressor(), "Ridge": Ridge(alpha=10)})
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(folds), 6)
        self.assertTrue(np.isfinite(summary[["R2_mean", "MAE_mean", "RMSE_mean"]].to_numpy()).all())


    def test_mlp_learning_summary_uses_actual_history(self):
        reg = SimpleNamespace(
            loss_curve_=[0.24, 0.14, 0.09, 0.04],
            validation_scores_=[0.61, 0.79, 0.76, 0.70],
            early_stopping=True,
        )
        mlp_pipe = SimpleNamespace(named_steps={"model": SimpleNamespace(regressor_=reg)})
        with tempfile.TemporaryDirectory() as directory:
            info = save_mlp_training_diagnostics(mlp_pipe, directory)
            self.assertEqual(info["best_epoch"], 2)
            self.assertAlmostEqual(info["best_validation_r2"], .79)
            self.assertAlmostEqual(info["last_validation_r2"], .70)
            self.assertEqual(len(pd.read_csv(Path(directory) / "mlp_learning_history.csv")), 4)
            self.assertTrue((Path(directory) / "mlp_training_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
