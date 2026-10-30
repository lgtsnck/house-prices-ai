"""Автоматические проверки классификации на синтетических примерах."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from house_prices_ai.classification import (
    build_classification_pipeline, compare_classifiers,
    create_classification_figures, save_mlp_history,
    starter_models, tune_xgboost,
)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 90
        self.X = pd.DataFrame({
            "OverallQual": rng.integers(2, 10, n),
            "GrLivArea": rng.integers(400, 3000, n),
            "YrSold": rng.integers(2006, 2011, n),
            "YearBuilt": rng.integers(1950, 2006, n),
            "Neighborhood": np.where(np.arange(n) % 2, "A", "B"),
            "PoolQC": [None] * n,
        })
        self.y = pd.Series(np.array([0, 1] * (n // 2)))

    def test_pipeline_predicts(self):
        model = build_classification_pipeline(self.X, LogisticRegression(max_iter=1500))
        model.fit(self.X, self.y)
        self.assertEqual(model.predict(self.X.iloc[:3]).shape, (3,))
        self.assertEqual(model.predict_proba(self.X.iloc[:3]).shape, (3, 2))

    def test_cv_expected_shapes(self):
        summary, folds = compare_classifiers(
            self.X, self.y, folds=3,
            models={"Dummy": DummyClassifier(strategy="most_frequent")},
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(folds), 3)
        self.assertAlmostEqual(summary.loc[0, "ROC_AUC_mean"], .5)

    def test_oof_each_object_one_prediction(self):
        summary, folds, oof = compare_classifiers(
            self.X, self.y, folds=3, return_oof=True,
            models={"DummyClassifier": DummyClassifier(strategy="most_frequent"),
                    "LogisticRegression": LogisticRegression(max_iter=1500)},
        )
        self.assertEqual(oof.shape[0], len(self.X))
        self.assertEqual(len(folds), 6)
        self.assertEqual(len(summary), 2)
        self.assertFalse(oof.isna().any().any())
        for name in ("DummyClassifier", "LogisticRegression"):
            self.assertTrue(oof[f"PredictedClass__{name}"].isin([0,1]).all())
            self.assertTrue(oof[f"ProbabilityClass1__{name}"].between(0,1).all())

    def test_reject_single_class(self):
        with self.assertRaises(ValueError):
            compare_classifiers(self.X, pd.Series(np.zeros(len(self.X))), folds=3)

    def test_six_figures_from_oof_and_mlp(self):
        from sklearn.neural_network import MLPClassifier
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1500),
            "MLPClassifier": MLPClassifier(hidden_layer_sizes=(8,), max_iter=50,
                                           early_stopping=True, validation_fraction=.2,
                                           n_iter_no_change=5, random_state=42),
        }
        summary, _, oof = compare_classifiers(self.X, self.y, folds=3, models=models, return_oof=True)
        mlp = build_classification_pipeline(self.X, models["MLPClassifier"])
        mlp.fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, details = save_mlp_history(mlp, root)
            figures = create_classification_figures(self.y, oof, summary,
                                                    root / "figures", history, details)
            self.assertEqual(len(figures), 6)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0
                                for path in figures.values()))
            self.assertEqual(details["epochs"], len(history))
            self.assertIn("best_epoch", json.loads((root/"mlp_training_summary.json").read_text()))

    def test_xgb_tuning_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = tune_xgboost(self.X, self.y, Path(tmp), iterations=1)
            self.assertEqual(result["n_iter"], 1)
            self.assertIn("max_depth", result["best_params"])
            self.assertTrue((Path(tmp)/"xgboost_tuning_trials.csv").exists())

    def test_model_catalog(self):
        models = starter_models()
        self.assertEqual(len(models), 5)
        self.assertIn("XGBoostClassifier", models)
        self.assertIn("MLPClassifier", models)


if __name__ == "__main__":
    unittest.main()
