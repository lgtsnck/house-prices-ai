"""Регрессионная проверка отделения расчётов EDA от построения рисунков."""
import unittest
import numpy as np
import pandas as pd

from house_prices_ai.eda import _prepare_eda_plot_inputs
from house_prices_ai.visualization import save_eda_plots


class EDARefactorTests(unittest.TestCase):
    @staticmethod
    def sample(n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'Id': np.arange(1, n + 1),
            'SalePrice': rng.normal(180000, 35000, n),
            'GrLivArea': rng.normal(1500, 250, n),
            'GarageArea': rng.normal(400, 100, n),
            'OverallQual': rng.integers(3, 9, n),
            'YearBuilt': rng.integers(1930, 2010, n),
            'Neighborhood': rng.choice(['A', 'B', 'C'], n),
        })

    def test_numeric_preparation_preserves_data_and_excludes_id(self):
        data = self.sample()
        before = data.copy(deep=True)
        result = _prepare_eda_plot_inputs(data)
        pd.testing.assert_frame_equal(before, data)
        self.assertNotIn('Id', result.numeric)
        self.assertIn('SalePrice', result.numeric)
        self.assertEqual(result.correlations.shape[0], result.numeric.shape[1])

    def test_pca_and_pairplot_are_deterministic(self):
        data = self.sample(450)
        first = _prepare_eda_plot_inputs(data)
        second = _prepare_eda_plot_inputs(data)
        self.assertEqual(len(first.pairplot_samples), 2)
        self.assertLessEqual(max(len(v) for v in first.pairplot_samples), 400)
        for a, b in zip(first.pairplot_samples, second.pairplot_samples):
            pd.testing.assert_frame_equal(a, b)
        self.assertIsNotNone(first.pca_points)
        np.testing.assert_allclose(first.pca_points, second.pca_points)
        self.assertEqual(first.pca_points.shape, (450, 2))

    def test_visualization_does_not_import_analytics_estimators(self):
        import inspect
        import ast
        from house_prices_ai import visualization
        source = inspect.getsource(visualization)
        tree = ast.parse(source)
        analytics = {'PCA', 'SimpleImputer', 'StandardScaler'}
        found = {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)
                 for alias in node.names}
        self.assertFalse(analytics & found)


if __name__ == '__main__':
    unittest.main()
