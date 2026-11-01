"""Графики предобработки строятся отдельно от вычислительного модуля."""
from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from house_prices_ai import preprocessing
from house_prices_ai.visualization import save_preprocessing_plots


class PreprocessingVisualizationTests(unittest.TestCase):
    def test_preprocessing_does_not_draw_plots(self):
        tree = ast.parse(inspect.getsource(preprocessing))
        imports = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            name.name for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        )
        self.assertFalse(any(name and name.startswith('matplotlib') for name in imports))
        self.assertNotIn('seaborn', imports)
        direct_drawing = {'plot', 'bar', 'barh', 'scatter', 'hist', 'savefig', 'subplots'}
        offenders = [
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in direct_drawing
        ]
        self.assertEqual(offenders, [])

    def test_three_preprocessing_figures_are_portrait_and_white(self):
        missing = pd.Series([7, 3, 0], index=['GarageYrBlt', 'LotFrontage', 'Alley'])
        scores = pd.Series([35.0, 18.0, 4.0], index=['num__OverallQual', 'num__GrLivArea', 'num__GarageCars'])
        pca = np.array([[-1.0, 0.3], [-0.4, -0.1], [0.8, 0.7], [1.2, -0.4]])
        classes = np.array([0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as tmp:
            result = save_preprocessing_plots(
                missing, scores, pca, classes, (0.35, 0.15), Path(tmp)
            )
            self.assertEqual(set(result), {'missing', 'scores', 'pca'})
            self.assertEqual(
                {path.name for path in result.values()},
                {'01_missing_values.png', '02_feature_selection.png', '03_pca_preprocessed.png'},
            )
            for path in result.values():
                self.assertTrue(path.is_file())
                with Image.open(path) as im:
                    self.assertEqual(im.size, (1400, 1900))
                    self.assertEqual(im.convert('RGB').getpixel((0, 0)), (255, 255, 255))
                    self.assertEqual(im.convert('RGB').getpixel((1399, 1899)), (255, 255, 255))


if __name__ == '__main__':
    unittest.main()
