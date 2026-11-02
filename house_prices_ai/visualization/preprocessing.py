"""Графики предобработки данных и инженерии признаков."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import create_report_figure, save_a4_figure


# Графики строятся по результатам вычислений из preprocessing.py.
def _center_preprocessing_bar(fig: plt.Figure, ax: plt.Axes) -> None:
    """Центрировать горизонтальные столбцы вместе с длинными подписями."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer)
    if bbox is None:
        return
    width = fig.bbox.width
    desired_dx = (width - bbox.x0 - bbox.x1) / 2.0
    padding = width * 0.04
    dx = max(padding - bbox.x0, min(desired_dx, width - padding - bbox.x1))
    x, y, w, h = ax.get_position().bounds
    ax.set_position([x + dx / width, y, w, h])


def _draw_preprocessing_bar(values: pd.Series, xlabel: str, left: float) -> plt.Figure:
    """Единая разметка двух горизонтальных диаграмм предобработки."""
    fig, ax = create_report_figure(figsize=(7, 9.5), facecolor="white")
    values.iloc[::-1].plot.barh(ax=ax, color="#487da8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dce1e6", linewidth=.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=left, right=.95, bottom=.09, top=.96)
    _center_preprocessing_bar(fig, ax)
    return fig


def save_preprocessing_plots(
    missing_counts: pd.Series,
    feature_scores: pd.Series,
    pca_points: np.ndarray,
    price_classes: np.ndarray,
    pca_explained_variance: np.ndarray | tuple[float, float],
    dest: Path,
) -> dict[str, Path]:
    """Сохранить три рисунка A4 из заранее подготовленных данных."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    plot_style = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#222222",
        "ytick.color": "#222222", "axes.edgecolor": "#555555",
        "font.size": 10, "axes.grid": False,
    }
    plots: dict[str, Path] = {}
    with plt.rc_context(plot_style):
        fig = _draw_preprocessing_bar(
            missing_counts, "Число пропусков в обучающей выборке", .30
        )
        plots["missing"] = dest / "01_missing_values.png"
        save_a4_figure(fig, plots["missing"], dpi=200, bbox_inches=None)

        fig = _draw_preprocessing_bar(
            feature_scores, "Статистика F для связи с ценой продажи", .34
        )
        plots["scores"] = dest / "02_feature_selection.png"
        save_a4_figure(fig, plots["scores"], dpi=200, bbox_inches=None)

        fig, ax = create_report_figure(figsize=(7, 9.5), facecolor="white")
        for cls, label, color in (
            (0, "Цена ниже $200 000", "#3269ab"),
            (1, "Цена от $200 000", "#cc6a35"),
        ):
            filt = (price_classes == cls)
            ax.scatter(pca_points[filt, 0], pca_points[filt, 1],
                       label=label, color=color, s=18, alpha=.55)
        ax.set_xlabel(f"PC1 ({pca_explained_variance[0]:.1%} дисперсии)")
        ax.set_ylabel(f"PC2 ({pca_explained_variance[1]:.1%} дисперсии)")
        ax.legend(loc="best", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="#bbbbbb")
        ax.grid(color="#e3e6e9", linewidth=.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=.16, right=.95, bottom=.10, top=.96)
        plots["pca"] = dest / "03_pca_preprocessed.png"
        save_a4_figure(fig, plots["pca"], dpi=200, bbox_inches=None)

    return plots
