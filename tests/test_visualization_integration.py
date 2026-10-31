"""Регрессионные проверки единого оформления рисунков (без Kaggle CSV)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from house_prices_ai.visualization import (
    A4_PORTRAIT_INCHES,
    A4_RASTER_SIZE_180DPI,
    create_a4_figure,
    create_report_figure,
    save_a4_figure,
    save_shap_a4_figure,
)


class VisualizationIntegrationTests(unittest.TestCase):
    def test_a4_factory_is_portrait_and_white(self):
        fig, ax = create_a4_figure()
        try:
            self.assertEqual(tuple(round(v, 2) for v in fig.get_size_inches()), A4_PORTRAIT_INCHES)
            self.assertEqual(tuple(fig.get_facecolor()), (1.0, 1.0, 1.0, 1.0))
            ax.plot([0, 1], [1, 0])
        finally:
            plt.close(fig)

    def test_shared_factory_accepts_old_custom_size(self):
        fig, axes = create_report_figure(2, 1, figsize=(7, 9.5))
        try:
            self.assertEqual(tuple(fig.get_size_inches()), (7.0, 9.5))
            self.assertEqual(len(axes), 2)
        finally:
            plt.close(fig)

    def test_save_without_tight_crop_keeps_exact_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'nested' / 'fixed.png'
            fig, ax = create_report_figure(figsize=(6.6, 9.2))
            ax.plot([0, 1], [1, 0])
            save_a4_figure(fig, p, dpi=100, bbox_inches=None)
            with Image.open(p) as img:
                self.assertEqual(img.width, 660)
                self.assertLessEqual(abs(img.height - 920), 1)
                self.assertEqual(img.getpixel((0, 0))[:3], (255, 255, 255))
            self.assertFalse(plt.fignum_exists(fig.number))

    def test_tight_save_preserves_legends_outside_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'legend.png'
            fig, ax = create_a4_figure()
            ax.plot([0, 1], [0, 1], label='A')
            ax.legend(loc='center left', bbox_to_anchor=(1.02, .5))
            save_a4_figure(fig, p, dpi=90)
            with Image.open(p) as img:
                self.assertLess(img.width, img.height)
                self.assertGreater(img.width, 50)

    def test_shap_canvas_exact_a4_and_no_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'shap.png'
            fig, ax = create_a4_figure()
            ax.barh(['Очень длинное название признака домовой стоимости'] * 1, [2.3])
            save_shap_a4_figure(fig, p)
            with Image.open(p) as img:
                self.assertEqual(img.size, A4_RASTER_SIZE_180DPI)
                self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))

    def test_landscape_figures_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig, _ = create_report_figure(figsize=(10, 4))
            with self.assertRaisesRegex(ValueError, 'книжная'):
                save_a4_figure(fig, Path(tmp) / 'bad.png')
            self.assertFalse(plt.fignum_exists(fig.number))


if __name__ == '__main__':
    unittest.main()
