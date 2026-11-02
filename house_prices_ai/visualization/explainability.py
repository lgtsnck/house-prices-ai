"""SHAP and model feature-importance visualizations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import create_a4_figure, save_shap_a4_figure


# Построение объяснений; их вычисление выполняет explainability.py.
def _save_explainability_figure(fig, path: Path) -> None:
    """A4 с вписыванием SHAP и длинных подписей без обрезки."""
    save_shap_a4_figure(fig, path)


def plot_importance_pair(rf_table: pd.DataFrame, xgb_table: pd.DataFrame, path: Path) -> None:
    fig, axes = create_a4_figure(2, 1, figsize=(8.27, 11.69), facecolor="white")
    for ax, table, label, color in zip(
        axes, (rf_table, xgb_table),
        ("Random Forest — уменьшение неоднородности", "XGBoost — важность разделений"),
        ("#487cac", "#378f70"),
    ):
        subset = table.head(12).iloc[::-1]
        ax.barh(subset["Feature"], subset["Importance"], color=color)
        ax.set_xlabel("Относительная важность в данной модели")
        ax.set_title(label, fontsize=11, pad=10)
        ax.grid(axis="x", alpha=.16)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.40, right=.95, bottom=.07, top=.94, hspace=.34)
    _save_explainability_figure(fig, path)

def plot_shap_beeswarm(explanation, path: Path) -> None:
    import shap
    plt.close("all")
    shap.plots.beeswarm(explanation, max_display=16, show=False,
                       plot_size=(8.27, 11.69), s=11)
    _save_explainability_figure(plt.gcf(), path)


def plot_shap_bar(explanation, path: Path) -> None:
    import shap
    plt.close("all")
    plt.figure(figsize=(8.27, 11.69))
    shap.plots.bar(explanation.abs.mean(0), max_display=16, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.27, 11.69)
    _save_explainability_figure(fig, path)


def plot_shap_waterfall(one_row, path: Path, units: str) -> None:
    """Печатная каскадная диаграмма аддитивных вкладов SHAP."""
    from matplotlib.ticker import StrMethodFormatter
    base = float(np.asarray(one_row.base_values).item())
    raw_values = np.asarray(one_row.values, dtype=float)
    names = list(one_row.feature_names)
    leading = np.argsort(np.abs(raw_values))[::-1][:11]
    rows = [(names[int(i)].replace("num__", "").replace("cat__", ""),
             float(raw_values[i])) for i in leading]
    others = float(raw_values.sum() - raw_values[leading].sum())
    rows.append(("Остальные признаки", others))
    starts = [base]
    for _, value in rows:
        starts.append(starts[-1] + value)
    predicted = starts[-1]
    low, high = min(starts), max(starts)
    delta = max(high - low, abs(predicted) * .03, 1.)
    pad = .24 * delta

    fig, ax = create_a4_figure(figsize=(8.27, 11.69), facecolor="white")
    for i, (name, value) in enumerate(rows):
        left = min(starts[i], starts[i + 1])
        ax.barh(i, abs(value), left=left, height=.62,
                color="#cf4263" if value >= 0 else "#337ebc")
        ax.text(starts[i + 1] + (.018 * delta if value >= 0 else -.018 * delta), i,
                f"{value:+,.2f}", ha="left" if value >= 0 else "right",
                va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(low - pad, high + pad)
    ax.axvline(base, color="#73808b", ls="--", lw=1.2, label="Базовое значение")
    ax.axvline(predicted, color="#242d36", ls=":", lw=1.4, label="Прогноз модели")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}" if abs(predicted) > 1000 else "{x:.2f}"))
    ax.set_xlabel("Прогноз стоимости, USD" if units == "USD" else "Логарифм отношения шансов (log-odds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.14)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.34, right=.96, top=.91, bottom=.10)
    fig.text(.12, .967, f"Базовый уровень: {base:,.2f}    Итоговый прогноз: {predicted:,.2f}",
             ha="left", va="top", fontsize=10)
    _save_explainability_figure(fig, path)


def save_shap_force_html(one_row, path: Path) -> None:
    import shap
    force = shap.plots.force(one_row, matplotlib=False, show=False)
    shap.save_html(str(path), force, full_html=True)

def plot_rf_xgb_shap_comparison(compare: pd.DataFrame, path: Path) -> None:
    """Сопоставить средние абсолютные SHAP-вклады на общей шкале USD."""
    fig, axes = create_a4_figure(2, 1, figsize=(8.27, 11.69), facecolor="white")
    for ax, name, col, color in (
        (axes[0], "Random Forest", "RF_MeanAbsSHAP_USD", "#487cac"),
        (axes[1], "XGBoost", "XGBoost_MeanAbsSHAP_USD", "#378f70"),
    ):
        df = compare.nlargest(12, col).iloc[::-1]
        ax.barh(df.index, df[col], color=color)
        ax.set_xlabel("Средний абсолютный вклад SHAP, USD")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="x", alpha=.16)
        ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=.38, right=.94, top=.94, bottom=.08, hspace=.32)
    _save_explainability_figure(fig, path)
