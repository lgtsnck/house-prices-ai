"""Проверка восстановления книжной ориентации EDA при сбойном холсте."""
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from PIL import Image

from house_prices_ai.visualization import _finish_plot, save_a4_figure


class EDAPortraitExportTests(unittest.TestCase):
    def test_eda_corrects_landscape_canvas(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / '01a_numeric_histograms.png'
            fig, ax = plt.subplots(figsize=(9.2, 6.6))
            ax.hist([2, 3, 5, 8, 13])
            _finish_plot(fig, target, dpi=90)
            with Image.open(target) as img:
                self.assertLess(img.width, img.height)
                self.assertEqual(img.getpixel((0, 0))[:3], (255, 255, 255))

    def test_general_export_still_rejects_landscape(self):
        with tempfile.TemporaryDirectory() as folder:
            fig, ax = plt.subplots(figsize=(9.2, 6.6))
            ax.plot([0, 1], [0, 1])
            with self.assertRaisesRegex(ValueError, 'книжная'):
                save_a4_figure(fig, Path(folder) / 'bad.png', dpi=90)

if __name__ == '__main__':
    unittest.main()
