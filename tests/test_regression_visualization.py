"""Контроль архитектуры и воспроизводимости диаграмм регрессии."""
import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from house_prices_ai import evaluation, regression
from house_prices_ai.visualization import create_regression_figures


class RegressionVisualizationTests(unittest.TestCase):
    def test_evaluation_and_regression_do_not_draw(self):
        drawing = {'scatter', 'plot', 'bar', 'barh', 'boxplot', 'hist', 'subplots',
                   'savefig', 'probplot'}
        for module in (evaluation, regression):
            tree = ast.parse(inspect.getsource(module))
            forbidden_imports = [
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith('matplotlib')
            ]
            forbidden_calls = [
                n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in drawing
            ]
            self.assertEqual(forbidden_imports, [], module.__name__)
            self.assertEqual(forbidden_calls, [], module.__name__)

    def test_all_six_figures_are_portrait_and_global_style_is_unchanged(self):
        models = ['DummyRegressor', 'Ridge', 'XGBoost']
        summary = pd.DataFrame({
            'Model': models, 'R2_mean': [-0.02, .80, .90],
            'MAE_mean': [60000, 21000, 16000], 'RMSE_mean':[85000,30000,24000],
        })
        folds = pd.DataFrame([{'Model': m, 'R2': float(r2)}
                              for m, r2 in zip(models, [0, .8, .9])
                              for _ in range(5)])
        y = np.linspace(85000, 300000, 30)
        pred = y + np.sin(np.arange(30)) * 8000
        mlp = pd.DataFrame({'Epoch': np.arange(1,5),
                            'Training_Loss': [.25,.14,.06,.04],
                            'Validation_R2': [.61,.79,.75,.71]})
        grid_before = plt.rcParams['axes.grid']
        font_before = plt.rcParams['font.size']
        with tempfile.TemporaryDirectory() as folder:
            paths = create_regression_figures(summary, folds, y, pred, folder, mlp)
            self.assertEqual(set(paths), {'metrics','r2','actual_pred','residual','qq','mlp'})
            for path in paths.values():
                with Image.open(path) as picture:
                    self.assertGreater(picture.height, picture.width)
                    self.assertEqual(picture.convert('RGB').getpixel((0,0)),(255,255,255))
        self.assertEqual(plt.rcParams['axes.grid'], grid_before)
        self.assertEqual(plt.rcParams['font.size'], font_before)

if __name__ == '__main__':
    unittest.main()
