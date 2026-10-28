"""Регрессионные проверки правила сохранения нулевой площади."""
import unittest
import numpy as np
from house_prices_ai.preprocessing import IQRClipper, RecalculateTotalUsefulArea


class TestStructuralZeroPreservation(unittest.TestCase):
    def test_missing_basement_and_other_structures_remain_zero(self):
        columns = ("TotalBsmtSF", "GarageArea", "2ndFlrSF", "1stFlrSF", "TotalUsefulSF")
        raw = np.array([
            [0, 0, 0, 700, 700],
            [800, 250, 0, 900, 1700],
            [900, 300, 0, 1000, 1900],
            [1000, 350, 500, 1100, 2600],
            [1100, 400, 600, 1200, 2900],
            [1200, 450, 700, 1300, 3200],
            [1300, 500, 800, 1400, 3500],
            [5000, 2500, 2000, 2500, 9500],
        ], dtype=float)
        clip = IQRClipper(columns).fit(raw)
        transformed = clip.transform(raw)
        for i in (0, 1, 2):
            self.assertEqual(transformed[0, i], 0.0)
        self.assertTrue(clip.lower_[0] == -np.inf)
        self.assertLess(transformed[-1, 0], raw[-1, 0])
        reconciled = RecalculateTotalUsefulArea(columns).transform(transformed)
        self.assertEqual(reconciled[0, -1], 700.0)
        np.testing.assert_allclose(reconciled[:, -1], reconciled[:, 0] + reconciled[:, 2] + reconciled[:, 3])


if __name__ == "__main__":
    unittest.main()
