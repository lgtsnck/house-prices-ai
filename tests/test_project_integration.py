"""Структурные проверки единой точки входа и расположения подписей."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestProjectIntegration(unittest.TestCase):
    def test_entrypoint_commands(self):
        import main
        self.assertEqual(main.main(["info"]), 0)
        self.assertEqual(main.main(["plan"]), 0)

    def test_clustering_route(self):
        import main
        with patch("house_prices_ai.clustering.run_clustering") as action:
            action.return_value = SimpleNamespace(report_path=Path("report.md"))
            self.assertEqual(main.main(["clustering", "--tsne", "--dbscan"]), 0)
            self.assertTrue(action.call_args.kwargs["tsne"])
            self.assertTrue(action.call_args.kwargs["dbscan"])

    def test_notebook_image_captions(self):
        text = (ROOT / "notebooks" / "06_clustering.py").read_text(encoding="utf-8")
        for idx in range(1, 9):
            self.assertIn(f"Рисунок 6.{idx} —", text)
        self.assertNotIn('display(Markdown(report.read_text', text)
        self.assertNotIn('# ### Рисунок', text)

    def test_syntax(self):
        ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        ast.parse((ROOT / "notebooks" / "06_clustering.py").read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
