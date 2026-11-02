"""Публичный интерфейс визуализации с разделением реализаций по этапам исследования.

This façade retains existing imports such as:
    from house_prices_ai.visualization import create_regression_figures
"""
from __future__ import annotations

# Сначала импортируем common, чтобы Matplotlib настроил режим без графического окна.
from .common import (
    A4_PORTRAIT_INCHES,
    A4_RASTER_SIZE_180DPI,
    create_report_figure,
    create_a4_figure,
    save_a4_figure,
    save_shap_a4_figure,
)
from .eda import save_eda_plots, _draw_boxplot, _finish_plot
from .preprocessing import save_preprocessing_plots
from .regression import create_regression_figures
from .classification import create_classification_figures
from .explainability import (
    plot_importance_pair,
    plot_shap_beeswarm,
    plot_shap_bar,
    plot_shap_waterfall,
    save_shap_force_html,
    plot_rf_xgb_shap_comparison,
)
from .clustering import create_clustering_figures

__all__ = [
    "A4_PORTRAIT_INCHES",
    "A4_RASTER_SIZE_180DPI",
    "create_report_figure",
    "create_a4_figure",
    "save_a4_figure",
    "save_shap_a4_figure",
    "save_eda_plots",
    # Вспомогательные функции сохранены для совместимости с тестами и импортами.
    "_draw_boxplot",
    "_finish_plot",
    "save_preprocessing_plots",
    "create_regression_figures",
    "create_classification_figures",
    "plot_importance_pair",
    "plot_shap_beeswarm",
    "plot_shap_bar",
    "plot_shap_waterfall",
    "save_shap_force_html",
    "plot_rf_xgb_shap_comparison",
    "create_clustering_figures",
]
