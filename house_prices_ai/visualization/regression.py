"""Regression model comparison and residual diagnostics charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .common import create_a4_figure, save_a4_figure


# Диаграммы регрессии: аналитические входные данные подготовлены в evaluation.py
# и regression.py. Из этого модуля не выполняется обучение моделей.
def _save_regression_figure(fig: plt.Figure, path: Path) -> None:
    """Сохранить исторический формат рисунков регрессии (A4 с полями)."""
    save_a4_figure(fig, path, dpi=200, pad_inches=.25)


def create_regression_figures(cv_summary: pd.DataFrame, cv_folds: pd.DataFrame,
                              y_true, oof_pred, out_dir: str | Path,
                              mlp_history: pd.DataFrame | None = None) -> dict[str, Path]:
    """Печатные рисунки 3.1–3.6 в книжном формате A4.

    Диаграммы 3.3–3.5 основаны на OOF-прогнозах обучающей выборки,
    а не на повторно просмотренном внешнем test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    colors = "#477baf"
    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": .18}):

        # 3.1 — три сопоставимые по моделям, но РАЗДЕЛЬНЫЕ шкалы метрик.
        # Нельзя строить MAE, RMSE (USD) и R² на одной числовой оси.
        order = cv_summary.sort_values("MAE_mean")["Model"].tolist()
        ranked = cv_summary.set_index("Model").loc[order]
        bar_colors = ["#287c6b" if name == order[0] else "#477baf" for name in order]
        fig, axes = create_a4_figure(3, 1, figsize=(8.27, 11.69))
        panels = [
            ("R2_mean", "R² — коэффициент детерминации (больше — лучше)", ".3f"),
            ("MAE_mean", "MAE, USD (меньше — лучше)", ",.0f"),
            ("RMSE_mean", "RMSE, USD (меньше — лучше)", ",.0f"),
        ]
        for ax, (metric, xlabel, label_format) in zip(axes, panels):
            values = ranked[metric].to_numpy(dtype=float)
            y_pos = np.arange(len(order))
            ax.barh(y_pos, values, color=bar_colors, height=.68)
            ax.set_yticks(y_pos, labels=order)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            if metric == "R2_mean":
                ax.set_xlim(min(-.10, float(values.min()) - .07), 1.04)
                ax.axvline(0, color="#b65d4f", ls="--", lw=1)
                dx = .02
            else:
                ax.set_xlim(0, float(values.max()) * 1.25)
                dx = float(values.max()) * .018
            for pos, value in enumerate(values):
                ax.text(value + dx, pos, format(value, label_format),
                        va="center", ha="left", fontsize=8.4)
            ax.grid(axis="y", visible=False)
        fig.subplots_adjust(left=.32, right=.96, top=.96, bottom=.07, hspace=.59)
        result["metrics"] = out_dir / "3_1_cv_mae.png"
        _save_regression_figure(fig, result["metrics"])

        # 3.2 — вариация валидационного R² между пятью фолдами.
        model_order = cv_summary.sort_values("R2_mean", ascending=False)["Model"].tolist()
        values = [cv_folds.loc[cv_folds.Model == name, "R2"].to_numpy() for name in model_order]
        fig, ax = create_a4_figure(figsize=(8.27, 11.69))
        ax.boxplot(values, tick_labels=model_order, orientation="horizontal", patch_artist=True,
                   boxprops={"facecolor":"#c5d9eb", "edgecolor":colors},
                   medianprops={"color":"#1d456a", "linewidth":2})
        ax.axvline(0, color="#c06b5a", ls="--", lw=1)
        ax.set_xlabel("R² на валидационных фолдах (больше — лучше)")
        ax.invert_yaxis()
        fig.subplots_adjust(left=.28, right=.95, top=.91, bottom=.12)
        result["r2"] = out_dir / "3_2_cv_r2_folds.png"
        _save_regression_figure(fig, result["r2"])

        y = np.asarray(y_true, dtype=float)
        pred = np.asarray(oof_pred, dtype=float)
        residual = y-pred

        # 3.3 — прогноз против факта: out-of-fold prediction.
        fig, ax = create_a4_figure(figsize=(8.27, 11.69))
        ax.scatter(y, pred, s=13, alpha=.48, color=colors)
        lo, hi = float(min(y.min(),pred.min())), float(max(y.max(), pred.max()))
        ax.plot([lo,hi], [lo,hi], '--', lw=1.5, color="#b65d4f")
        ax.set_xlabel("Фактическая цена, USD")
        ax.set_ylabel("Прогноз (OOF), USD")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        # Не принуждаем оси к квадратной области: рисунок остаётся книжным A4.
        fig.subplots_adjust(left=.15, right=.94, top=.94, bottom=.15)
        result["actual_pred"] = out_dir / "3_3_oof_actual_vs_pred.png"
        _save_regression_figure(fig, result["actual_pred"])

        # 3.4 — остатки относительно прогноза.
        fig, ax = create_a4_figure(figsize=(8.27, 11.69))
        ax.scatter(pred, residual, s=13, alpha=.48, color=colors)
        ax.axhline(0, lw=1.5, ls='--', color="#b65d4f")
        ax.set_xlabel("Прогноз (OOF), USD")
        ax.set_ylabel("Остаток: фактическая цена − прогноз, USD")
        fig.subplots_adjust(left=.18, right=.94, top=.94, bottom=.15)
        result["residual"] = out_dir / "3_4_oof_residuals.png"
        _save_regression_figure(fig, result["residual"])

        # 3.5 — Q-Q. Нормальность остатков не обязательна для XGBoost,
        # график является диагностикой хвостов распределения ошибок.
        fig, ax = create_a4_figure(figsize=(8.27, 11.69))
        stats.probplot(residual, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(colors)
        ax.get_lines()[0].set_markeredgecolor(colors)
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color("#b65d4f")
        ax.set_xlabel("Теоретические квантили нормального распределения")
        ax.set_ylabel("Наблюдаемые квантили остатков, USD")
        ax.set_title("")
        fig.subplots_adjust(left=.18, right=.94, top=.94, bottom=.15)
        result["qq"] = out_dir / "3_5_oof_qq.png"
        _save_regression_figure(fig, result["qq"])

        # 3.6 — динамика MLP, обученной на ВСЕЙ внешней train-части.
        if mlp_history is not None and not mlp_history.empty:
            loss = mlp_history["Training_Loss"].to_numpy(dtype=float)
            validation = mlp_history["Validation_R2"].to_numpy(dtype=float)
            if len(loss) > 0:
                fig, ax = create_a4_figure(figsize=(8.27, 11.69))
                epochs = np.arange(1, len(loss)+1)
                ax.plot(epochs, loss, color=colors, lw=2)
                ax.set_xlabel("Эпоха")
                ax.set_ylabel("Training Loss (масштабированная цель)")
                ax.set_yscale("log")
                if len(validation) > 0:
                    ax2 = ax.twinx()
                    ax2.plot(epochs, validation,
                             color="#c47c41", alpha=.85, lw=1.8)
                    best_ep = int(np.argmax(validation))+1
                    ax.axvline(best_ep, color="#7f8890", ls="--", lw=1)
                    ax2.scatter([best_ep], [validation[best_ep - 1]],
                                color="#c47c41", s=48, zorder=5)
                    ax2.annotate(
                        f"Лучшая эпоха: {best_ep}\nR² = {validation[best_ep - 1]:.3f}",
                        (best_ep, validation[best_ep - 1]),
                        xytext=(10, -25), textcoords="offset points", fontsize=9,
                        bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#dddddd"},
                    )
                    ax2.set_ylabel("Validation R²")
                fig.subplots_adjust(left=.16, right=.84, top=.94, bottom=.14)
                result["mlp"] = out_dir / "3_6_mlp_training.png"
                _save_regression_figure(fig, result["mlp"])
        return result
