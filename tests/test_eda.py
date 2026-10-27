"""Тесты загрузки и EDA на полностью синтетических данных (без Kaggle CSV)."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from house_prices_ai.data_loader import load_training_data
from house_prices_ai.eda import run_eda
from house_prices_ai.visualization import _draw_boxplot
from main import main


class EDATests(unittest.TestCase):
    def make_test_csv(self, path: Path) -> None:
        rng = np.random.default_rng(42)
        n = 40
        data = pd.DataFrame({
            "Id": range(1, n + 1),
            "SalePrice": np.r_[np.full(20, 150_000), np.full(20, 250_000)] + rng.normal(0, 5000, n),
            "GrLivArea": rng.normal(1500, 200, n),
            "OverallQual": rng.integers(4, 9, n),
            "GarageArea": rng.normal(500, 60, n),
            "Neighborhood": rng.choice(["A", "B", "C"], n),
        })
        data.loc[0, "GarageArea"] = np.nan
        data.to_csv(path, index=False)

    def test_none_category_is_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.csv"
            path.write_text("Id,SalePrice,MasVnrType,Alley\n"
                            "1,100000,None,NA\n"
                            "2,200000,,Pave\n", encoding="utf-8")
            data = load_training_data(path)
            self.assertEqual(data.loc[0, "MasVnrType"], "None")
            self.assertTrue(pd.isna(data.loc[1, "MasVnrType"]))
            self.assertTrue(pd.isna(data.loc[0, "Alley"]))

    def test_missing_file_has_clear_message(self):
        with self.assertRaises(FileNotFoundError):
            load_training_data("not_an_existing_file.csv")

    def test_report_and_class_balance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.csv"
            self.make_test_csv(path)
            result = run_eda(path, root / "out", make_plots=False)
            self.assertEqual(result.observations, 40)
            self.assertEqual(result.missing_total, 1)
            self.assertEqual(result.class_balance["Количество"].sum(), 40)
            self.assertTrue((root / "out" / "lr01_eda_summary.md").is_file())
            self.assertTrue((root / "out" / "feature_dictionary.csv").is_file())
            self.assertIn("Корреляция", (root / "out" / "lr01_eda_summary.md").read_text(encoding="utf-8"))

    def test_graphs_are_generated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.csv"
            self.make_test_csv(path)
            result = run_eda(path, root / "out", make_plots=True)
            self.assertGreaterEqual(len(result.plots), 6)
            self.assertTrue(all(file.is_file() for file in result.plots.values()))
            from PIL import Image
            for file in result.plots.values():
                with Image.open(file) as image:
                    self.assertLess(image.width, image.height, file.name)
            self.assertTrue(any(key.startswith("histograms_") for key in result.plots))
            self.assertTrue(any(key.startswith("boxplots_") for key in result.plots))
            self.assertIn("correlations", result.plots)
            self.assertFalse(any(key.startswith("correlations_") for key in result.plots))

    def test_boxplot_visible_outlier_markers(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            box = _draw_boxplot(ax, np.array([10] * 18 + [11] * 12 + [100]))
            self.assertEqual(len(box["fliers"]), 1)
            self.assertGreater(len(box["fliers"][0].get_ydata()), 0)
            self.assertGreaterEqual(box["fliers"][0].get_markersize(), 3)
        finally:
            plt.close(fig)

    def test_cli_eda_without_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.csv"
            self.make_test_csv(path)
            code = main(["eda", "--data", str(path), "--output", str(root / "results"), "--no-plots"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
