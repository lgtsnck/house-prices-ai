"""Метрики и диаграммы для лабораторной работы №3 (регрессия)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Сопоставимые MAE, RMSE и R² в исходной шкале SalePrice (USD)."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _save_figure(fig, path: Path) -> None:
    fig.set_facecolor("white")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)



def save_mlp_training_diagnostics(mlp_pipeline, output_dir: str | Path) -> dict:
    """Сохраняет фактическую историю обучения MLP для анализа Early Stopping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reg = mlp_pipeline.named_steps["model"].regressor_
    loss = np.asarray(reg.loss_curve_, dtype=float)
    validation = np.asarray(reg.validation_scores_, dtype=float)
    if len(loss) != len(validation) or len(loss) == 0:
        raise ValueError("Не найдена полная история обучения MLP с Early Stopping")
    best_index = int(np.nanargmax(validation))
    frame = pd.DataFrame({
        "Epoch": np.arange(1, len(loss) + 1),
        "Training_Loss": loss,
        "Validation_R2": validation,
    })
    frame.to_csv(output_dir / "mlp_learning_history.csv", index=False)
    data = {
        "epochs_run": int(len(loss)),
        "best_epoch": best_index + 1,
        "first_loss": float(loss[0]),
        "best_epoch_loss": float(loss[best_index]),
        "last_loss": float(loss[-1]),
        "first_validation_r2": float(validation[0]),
        "best_validation_r2": float(validation[best_index]),
        "last_validation_r2": float(validation[-1]),
        "early_stopping": bool(reg.early_stopping),
        "best_weights_restored": bool(reg.early_stopping),
        "note": "После Early Stopping восстановлены лучшие веса; loss_ относится к последней эпохе.",
    }
    (output_dir / "mlp_training_summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data

def create_regression_figures(cv_summary: pd.DataFrame, cv_folds: pd.DataFrame,
                              y_true, oof_pred, out_dir: str | Path,
                              mlp_model=None) -> dict[str, Path]:
    """Печатные рисунки 3.1–3.6 в книжном формате A4.

    Диаграммы 3.3–3.5 основаны на OOF-прогнозах обучающей выборки,
    а не на повторно просмотренном внешнем test.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    colors = "#477baf"
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .18})

    # 3.1 — три сопоставимые по моделям, но РАЗДЕЛЬНЫЕ шкалы метрик.
    # Нельзя строить MAE, RMSE (USD) и R² на одной числовой оси.
    order = cv_summary.sort_values("MAE_mean")["Model"].tolist()
    ranked = cv_summary.set_index("Model").loc[order]
    bar_colors = ["#287c6b" if name == order[0] else "#477baf" for name in order]
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
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
    _save_figure(fig, result["metrics"])

    # 3.2 — вариация валидационного R² между пятью фолдами.
    model_order = cv_summary.sort_values("R2_mean", ascending=False)["Model"].tolist()
    values = [cv_folds.loc[cv_folds.Model == name, "R2"].to_numpy() for name in model_order]
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.boxplot(values, tick_labels=model_order, vert=False, patch_artist=True,
               boxprops={"facecolor":"#c5d9eb", "edgecolor":colors},
               medianprops={"color":"#1d456a", "linewidth":2})
    ax.axvline(0, color="#c06b5a", ls="--", lw=1)
    ax.set_xlabel("R² на валидационных фолдах (больше — лучше)")
    ax.invert_yaxis()
    fig.subplots_adjust(left=.28, right=.95, top=.91, bottom=.12)
    result["r2"] = out_dir / "3_2_cv_r2_folds.png"
    _save_figure(fig, result["r2"])

    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(oof_pred, dtype=float)
    residual = y-pred

    # 3.3 — прогноз против факта: out-of-fold prediction.
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.scatter(y, pred, s=13, alpha=.48, color=colors)
    lo, hi = float(min(y.min(),pred.min())), float(max(y.max(), pred.max()))
    ax.plot([lo,hi], [lo,hi], '--', lw=1.5, color="#b65d4f")
    ax.set_xlabel("Фактическая цена, USD")
    ax.set_ylabel("Прогноз (OOF), USD")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    # Не принуждаем оси к квадратной области: рисунок остаётся книжным A4.
    fig.subplots_adjust(left=.15, right=.94, top=.94, bottom=.15)
    result["actual_pred"] = out_dir / "3_3_oof_actual_vs_pred.png"
    _save_figure(fig, result["actual_pred"])

    # 3.4 — остатки относительно прогноза.
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.scatter(pred, residual, s=13, alpha=.48, color=colors)
    ax.axhline(0, lw=1.5, ls='--', color="#b65d4f")
    ax.set_xlabel("Прогноз (OOF), USD")
    ax.set_ylabel("Остаток: фактическая цена − прогноз, USD")
    fig.subplots_adjust(left=.18, right=.94, top=.94, bottom=.15)
    result["residual"] = out_dir / "3_4_oof_residuals.png"
    _save_figure(fig, result["residual"])

    # 3.5 — Q-Q. Нормальность остатков не обязательна для XGBoost,
    # график является диагностикой хвостов распределения ошибок.
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
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
    _save_figure(fig, result["qq"])

    # 3.6 — динамика MLP, обученной на ВСЕЙ внешней train-части.
    if mlp_model is not None:
        regressor = mlp_model.named_steps["model"].regressor_
        if getattr(regressor, "loss_curve_", None) is not None and len(regressor.loss_curve_) > 0:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            epochs = np.arange(1, len(regressor.loss_curve_)+1)
            ax.plot(epochs, regressor.loss_curve_, color=colors, lw=2)
            ax.set_xlabel("Эпоха")
            ax.set_ylabel("Training Loss (масштабированная цель)")
            ax.set_yscale("log")
            if getattr(regressor, "validation_scores_", None) is not None and len(regressor.validation_scores_) > 0:
                ax2 = ax.twinx()
                ax2.plot(epochs, regressor.validation_scores_,
                         color="#c47c41", alpha=.85, lw=1.8)
                best_ep = int(np.argmax(regressor.validation_scores_))+1
                ax.axvline(best_ep, color="#7f8890", ls="--", lw=1)
                ax2.scatter([best_ep], [regressor.validation_scores_[best_ep - 1]],
                            color="#c47c41", s=48, zorder=5)
                ax2.annotate(
                    f"Лучшая эпоха: {best_ep}\nR² = {regressor.validation_scores_[best_ep - 1]:.3f}",
                    (best_ep, regressor.validation_scores_[best_ep - 1]),
                    xytext=(10, -25), textcoords="offset points", fontsize=9,
                    bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#dddddd"},
                )
                ax2.set_ylabel("Validation R²")
            fig.subplots_adjust(left=.16, right=.84, top=.94, bottom=.14)
            result["mlp"] = out_dir / "3_6_mlp_training.png"
            _save_figure(fig, result["mlp"])
    return result
