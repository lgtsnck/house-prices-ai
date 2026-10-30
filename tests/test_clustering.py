"""Быстрые проверки кластеризации на синтетических объектах, без Kaggle CSV."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from house_prices_ai.clustering import (
    CLUSTER_FEATURES, analyze_dbscan, describe_clusters, make_figures,
    prepare_features, select_cluster_count,
)


def sample(n=60):
    rng = np.random.default_rng(42)
    type_ = np.repeat([0, 1], n // 2)
    return pd.DataFrame({
        "OverallQual": rng.normal(5 + 3*type_, .4),
        "OverallCond": rng.normal(5, .5, n),
        "TotalBsmtSF": rng.normal(500 + 450*type_, 60),
        "1stFlrSF": rng.normal(700 + 600*type_, 65),
        "2ndFlrSF": rng.normal(0 + 200*type_, 45).clip(0),
        "FullBath": rng.integers(1, 3, n),
        "HalfBath": rng.integers(0, 2, n),
        "BsmtFullBath": rng.integers(0, 2, n),
        "BsmtHalfBath": rng.integers(0, 2, n),
        "GarageCars": rng.integers(1, 3, n),
        "YrSold": np.full(n, 2010),
        "YearBuilt": np.where(type_==0, 1975, 2005),
        "LotArea": rng.normal(6000+3000*type_, 300),
        "BsmtFinSF1": rng.normal(250+300*type_, 30),
    })


class ClusteringTests(unittest.TestCase):
    def test_no_price_or_id_allowed(self):
        X = sample()
        X["SalePrice"] = 100_000
        with self.assertRaisesRegex(ValueError, "Целевые"):
            prepare_features(X)

    def test_preparation_and_selection(self):
        X = sample()
        copy = X.copy(deep=True)
        numeric, clean, _ = prepare_features(X)
        self.assertEqual(numeric.shape, (60, len(CLUSTER_FEATURES)))
        self.assertTrue(np.isfinite(numeric).all())
        self.assertTrue(X.equals(copy))
        table, k = select_cluster_count(numeric, max_k=4, silhouette_sample=50)
        self.assertTrue(2 <= k <= 4)
        self.assertEqual(len(table), 3)
        labels = np.repeat([0,1], 30)
        prices = pd.Series(np.where(labels == 0, 120_000, 280_000))
        profile = describe_clusters(clean, labels, prices)
        self.assertEqual(profile.Count.sum(), 60)
        self.assertEqual(profile.ExpensiveShare.tolist(), [0.,1.])

    def test_figures_and_dbscan(self):
        x, clean, _ = prepare_features(sample())
        table, k = select_cluster_count(x, max_k=3, silhouette_sample=50)
        from sklearn.cluster import KMeans
        labs = KMeans(n_clusters=k, n_init=15, random_state=42).fit_predict(x)
        prices = pd.Series(np.where(np.arange(60) < 30, 120_000, 280_000))
        dlabels, stat = analyze_dbscan(x)
        self.assertEqual(len(dlabels), len(x))
        self.assertGreaterEqual(stat["noise_fraction"],0)
        with tempfile.TemporaryDirectory() as temp:
            figures, pca = make_figures(x, clean, labs, prices, table, k, Path(temp),
                                        dbscan_labels=dlabels)
            self.assertEqual(len(figures), 7)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in figures.values()))
            self.assertEqual(len(pca.explained_variance_ratio_),2)


if __name__ == "__main__":
    unittest.main()
