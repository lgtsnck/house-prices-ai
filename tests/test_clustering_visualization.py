"""Архитектура и форма графиков кластеризации после разделения слоёв."""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices_ai import clustering
from house_prices_ai.visualization import create_clustering_figures


class ClusteringVisualizationTests(unittest.TestCase):
    def test_clustering_module_has_no_plotting_or_secondary_cli(self):
        module = Path(clustering.__file__)
        source = module.read_text(encoding='utf-8')
        tree = ast.parse(source)
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn('make_figures', defined)
        self.assertNotIn('main', defined)
        self.assertNotIn('import matplotlib', source)
        self.assertNotIn('plt.', source)
        self.assertIn('prepare_clustering_plot_data', defined)
        self.assertTrue(callable(create_clustering_figures))

    def test_rendering_prepared_arrays_creates_expected_pngs(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal((40, len(clustering.CLUSTER_FEATURES)))
        labels = np.repeat([0, 1], 20)
        prices = pd.Series(rng.uniform(100_000, 350_000, size=40))
        ks = pd.DataFrame({'k': [2, 3], 'Inertia': [80., 60.],
                           'Silhouette': [.30, .25]})
        dbscan_labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
        data, pca = clustering.prepare_clustering_plot_data(
            x, labels, prices, ks, 2, dbscan_labels=dbscan_labels,
        )
        self.assertEqual(data['coords'].shape, (40, 2))
        self.assertEqual(len(pca.explained_variance_ratio_), 2)
        with tempfile.TemporaryDirectory() as tmp:
            images = create_clustering_figures(data, Path(tmp))
            self.assertEqual(len(images), 7)  # 6 основных + DBSCAN, без t-SNE
            self.assertTrue(all(p.is_file() and p.stat().st_size > 0
                                for p in images.values()))


if __name__ == '__main__':
    unittest.main()
