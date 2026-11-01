"""Regression is launched through main.py, without one-off entry scripts."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from main import main
from house_prices_ai import regression


class RegressionEntrypointTests(unittest.TestCase):
    def test_main_refresh_routes_to_regression_module(self):
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            dest = Path(directory)
            with patch.object(regression, "refresh_regression_outputs", return_value={"mlp": dest / "mlp.png"}) as run:
                with redirect_stdout(output):
                    code = main(["regression", "--refresh-plots", "--output", str(dest)])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["out_dir"], dest)
            self.assertIn("1", output.getvalue())

    def test_refresh_uses_saved_cv_and_refits_only_mlp(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            pd.DataFrame({"Model": ["MLP"], "MAE_mean": [22.0]}).to_csv(dest / "cv_clean_summary.csv", index=False)
            pd.DataFrame({"Model": ["MLP"], "Fold": [1], "R2": [0.8]}).to_csv(dest / "cv_clean_folds.csv", index=False)
            pd.DataFrame({"Id": [101, 103], "OOF_Pred": [160., 190.]}).to_csv(dest / "oof_predictions.csv", index=False)
            X = pd.DataFrame({"LotArea": [1200., 1600.]}, index=[101, 103])
            y = pd.Series([170., 200.], index=[101, 103])

            class FakeMLP:
                def fit(self, features, target):
                    self.called = True
                    return self

            mlp = FakeMLP()

            def record_training(_mlp, path):
                self.assertIs(_mlp, mlp)
                pd.DataFrame({"Epoch": [1, 2], "Training_Loss": [0.2, 0.1],
                              "Validation_R2": [0.6, 0.7]}).to_csv(path / "mlp_learning_history.csv", index=False)
                return {"best_epoch": 2, "epochs_run": 2}

            def record_figures(summary, folds, values, predictions, folder, history):
                self.assertEqual(folder, dest / "figures")
                self.assertEqual(history.shape, (2, 3))
                np.testing.assert_array_equal(predictions, [160, 190])
                np.testing.assert_array_equal(values, [170, 200])
                return {"mlp": folder / "3_6_mlp_training.png"}

            with patch.object(regression, "load_lr02_split", return_value=(X, None, y, None)), \
                 patch.object(regression, "model_catalog", return_value={"MLP": object()}), \
                 patch.object(regression, "make_regression_pipeline", return_value=mlp), \
                 patch.object(regression, "save_mlp_training_diagnostics", side_effect=record_training), \
                 patch("house_prices_ai.visualization.create_regression_figures", side_effect=record_figures):
                with redirect_stdout(StringIO()):
                    files = regression.refresh_regression_outputs(out_dir=dest)
            self.assertTrue(mlp.called)
            self.assertEqual(set(files), {"mlp"})

    def test_refresh_rejects_wrong_oof_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            pd.DataFrame({"Model": ["MLP"]}).to_csv(dest / "cv_clean_summary.csv", index=False)
            pd.DataFrame({"Model": ["MLP"]}).to_csv(dest / "cv_clean_folds.csv", index=False)
            pd.DataFrame({"Id": [2, 1], "OOF_Pred": [10., 20.]}).to_csv(dest / "oof_predictions.csv", index=False)
            features = pd.DataFrame({"a": [0., 1.]}, index=[1, 2])
            target = pd.Series([12., 18.], index=[1, 2])
            with patch.object(regression, "load_lr02_split", return_value=(features, None, target, None)):
                with self.assertRaisesRegex(ValueError, "Порядок OOF"):
                    regression.refresh_regression_outputs(out_dir=dest)


if __name__ == "__main__":
    unittest.main()
