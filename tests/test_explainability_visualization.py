"""Проверка архитектуры и графических форматов интерпретации моделей."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import house_prices_ai.explainability as analysis
import house_prices_ai.visualization as visual


class ExplainabilityVisualizationTests(unittest.TestCase):
    def test_explainability_calculates_but_does_not_draw(self):
        path = Path(analysis.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertFalse(defined.intersection({
            "_save", "_plot_importance_pair", "_shap_beeswarm",
            "_shap_bar", "_shap_waterfall", "_shap_force_html",
        }))
        self.assertFalse(any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
                             for node in tree.body))
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("matplotlib.pyplot", source)
        self.assertNotIn("create_a4_figure(", source)
        self.assertIs(analysis.plot_shap_waterfall, visual.plot_shap_waterfall)
        self.assertIs(analysis.save_shap_force_html, visual.save_shap_force_html)

    def test_shap_static_and_interactive_exports(self):
        import shap
        rng = np.random.default_rng(42)
        n, features = 35, 6
        values = rng.normal(0, .1, size=(n, features))
        data = rng.normal(size=(n, features))
        names = [f"Feature_{x}" for x in range(features)]
        explanation = shap.Explanation(
            values=values,
            base_values=np.full(n, .2),
            data=data,
            feature_names=names,
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = {
                "beeswarm": folder / "beeswarm.png",
                "bar": folder / "bar.png",
                "waterfall": folder / "waterfall.png",
            }
            visual.plot_shap_beeswarm(explanation, files["beeswarm"])
            visual.plot_shap_bar(explanation, files["bar"])
            visual.plot_shap_waterfall(explanation[0], files["waterfall"], "log-odds")
            for path in files.values():
                self.assertTrue(path.is_file() and path.stat().st_size > 0, str(path))
                with Image.open(path) as picture:
                    self.assertLess(picture.width, picture.height)
                    self.assertEqual(picture.convert("RGB").getpixel((0, 0)), (255, 255, 255))
            html = folder / "force.html"
            visual.save_shap_force_html(explanation[0], html)
            page = html.read_text(encoding="utf-8")
            self.assertIn("shap", page.lower())
            self.assertIn("<html", page.lower())


if __name__ == "__main__":
    unittest.main()
