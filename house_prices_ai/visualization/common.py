"""Общие функции создания листов A4 и сохранения изображений."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Режим без графического окна для командного запуска и тестов.
import matplotlib.pyplot as plt


# Общие функции оформления для всех этапов исследования.
# Размеры и существующие параметры сохранения сохранены ради совместимости.
A4_PORTRAIT_INCHES = (8.27, 11.69)
A4_RASTER_SIZE_180DPI = (1488, 2104)


def create_report_figure(*args, figsize=A4_PORTRAIT_INCHES, **kwargs):
    """Создаёт белую фигуру с возможностью указать размер и параметры осей.

    figsize может быть задан явно для старых графиков, чтобы не изменять
    пропорции и поля во время рефакторинга.
    """
    kwargs.setdefault("facecolor", "white")
    return plt.subplots(*args, figsize=figsize, **kwargs)


def create_a4_figure(*args, **kwargs):
    """Один или несколько графиков на книжной странице A4."""
    return create_report_figure(*args, **kwargs)


def save_a4_figure(fig, path, *, dpi=200, pad_inches=0.22,
                   bbox_inches="tight", transparent=False):
    """Общий способ сохранения книжных изображений.

    По умолчанию сохраняется содержимое с полями (историческое поведение
    регрессии/классификации/кластеризации). Для изображений с заранее
    рассчитанными полями передавайте bbox_inches=None (EDA/предобработка).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if fig.get_figwidth() >= fig.get_figheight():
            raise ValueError(f"Ожидалась книжная ориентация изображения: {path}")
        fig.set_facecolor("white")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches,
                    pad_inches=pad_inches if bbox_inches else None,
                    facecolor="white", transparent=transparent)
    finally:
        plt.close(fig)


def save_shap_a4_figure(fig, path, *, dpi=180, margin_px=(110, 120)):
    """Вписать график SHAP с длинными подписями в белый лист A4.

    Сначала делается tight-crop, затем он масштабируется без искажения
    пропорций, как в исходном модуле интерпретации. Возвращается PNG
    стандартного размера 1488x2104 при dpi=180.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    try:
        fig.set_facecolor("white")
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=.25, facecolor="white")
    finally:
        plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as raw:
        cropped = raw.convert("RGB")
    width, height = A4_RASTER_SIZE_180DPI
    free_width = width - margin_px[0]
    free_height = height - margin_px[1]
    ratio = min(free_width / cropped.width, free_height / cropped.height, 1.0)
    fitted = cropped.resize((round(cropped.width * ratio), round(cropped.height * ratio)),
                            Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    xy = ((width - fitted.width) // 2, (height - fitted.height) // 2)
    canvas.paste(fitted, xy)
    canvas.save(path, dpi=(dpi, dpi))
