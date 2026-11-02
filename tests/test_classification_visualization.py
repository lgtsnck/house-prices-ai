"""Архитектурные проверки разделения вычислений и диаграмм классификации."""
from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from house_prices_ai import classification
from house_prices_ai.visualization import create_classification_figures


class ClassificationVisualizationTests(unittest.TestCase):
    def test_classification_contains_no_plotting_code_or_second_cli(self):
        tree = ast.parse(inspect.getsource(classification))
        names = {
            item.name for node in ast.walk(tree)
            if isinstance(node, ast.Import) for item in node.names
        }
        names |= {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any(name and name.startswith("matplotlib") for name in names))
        self.assertNotIn("seaborn", names)
        calls = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(calls & {"scatter", "imshow", "savefig", "subplots", "bar", "barh"})
        # Проверяем, что функция не определена в расчётном модуле.
        defined_functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("create_classification_figures", defined_functions)

        # Подтверждаем, что используется функция из слоя визуализации.
        self.assertIs(
            classification.create_classification_figures,
            create_classification_figures,
        )
        source = inspect.getsource(classification)
        self.assertNotIn('if __name__ == "__main__":', source)

    def test_prepared_arrays_and_six_portrait_figures(self):
        models = ["LogisticRegression", "RandomForestClassifier",
                  "XGBoostClassifier", "MLPClassifier"]
        y = pd.Series([0, 1] * 8, dtype=int)
        oof = pd.DataFrame({"ActualClass": y})
        for idx, model in enumerate(models):
            scores = np.where(y.to_numpy() == 1, .8 - idx*.02, .15 + idx*.02)
            oof[f"PredictedClass__{model}"] = (scores > .5).astype(int)
            oof[f"ProbabilityClass1__{model}"] = scores
        summary = pd.DataFrame({
            "Model": models,
            "Accuracy_mean": [.9, .89, .88, .87],
            "Precision_mean": [.9, .89, .88, .87],
            "Recall_mean": [.9, .89, .88, .87],
            "F1_mean": [.9, .89, .88, .87],
        })
        plot_data = classification.prepare_classification_plot_data(y, oof, summary)
        self.assertEqual(plot_data["n_objects"], 16)
        self.assertEqual(set(plot_data["roc"]), set(models))
        self.assertEqual(set(plot_data["confusions"]), set(models))
        for model in models:
            self.assertEqual(int(plot_data["confusions"][model].sum()), 16)
        history = pd.DataFrame({
            "Epoch": [1, 2, 3], "TrainingLoss": [.4, .3, .2],
            "ValidationAccuracy": [.6, .75, .7],
        })
        with tempfile.TemporaryDirectory() as directory:
            files = create_classification_figures(
                plot_data, summary, Path(directory), history, {"best_epoch": 2}
            )
            self.assertEqual(len(files), 6)
            for file in files.values():
                with Image.open(file) as img:
                    self.assertLess(img.width, img.height)
                    self.assertEqual(img.convert("RGB").getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
