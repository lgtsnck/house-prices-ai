"""Графики сравнения моделей классификации."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import create_a4_figure, save_a4_figure


# Числовые данные для графиков подготавливает classification.py.
_CLASSIFICATION_COLORS = {
    "DummyClassifier": "#8d969f",
    "LogisticRegression": "#427bb2",
    "RandomForestClassifier": "#399079",
    "XGBoostClassifier": "#cd8d37",
    "MLPClassifier": "#8c75b4",
}


def _save_classification_figure(fig: plt.Figure, path: Path) -> None:
    save_a4_figure(fig, path, dpi=185, pad_inches=.24)


def _classification_figure() -> tuple[plt.Figure, plt.Axes]:
    return create_a4_figure()


def create_classification_figures(plot_data: dict,
                                  summary: pd.DataFrame, output_dir: Path,
                                  mlp_history: pd.DataFrame | None = None,
                                  mlp_details: dict | None = None) -> dict[str, Path]:
    """Шесть рисунков A4 с составом выборки, сравнением метрик, матрицами ошибок, ROC, PR и MLP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    models = plot_data["models"]

    fig, ax = _classification_figure()
    counts = plot_data["counts"]
    bars = ax.bar(["Ниже $200 000", "От $200 000"], counts,
                  color=["#427bb2", "#cd8d37"], width=.58)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{count} ({count/plot_data["n_objects"]:.1%})", ha="center", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set_ylabel("Число объектов")
    ax.grid(axis="y", alpha=.18)
    fig.subplots_adjust(left=.17, right=.94, bottom=.13, top=.91)
    figures["balance"] = output_dir / "4_1_class_distribution.png"
    _save_classification_figure(fig, figures["balance"])

    # Три отдельных шкалы, чтобы метрики не смешивались в одной системе координат.
    fig, axes = create_a4_figure(4, 1, figsize=(8.27, 11.69), facecolor="white")
    measure = (("Accuracy_mean", "Accuracy"), ("Precision_mean", "Precision"),
               ("Recall_mean", "Recall"), ("F1_mean", "F1-score"))
    names = summary["Model"].tolist()[::-1]
    for ax, (col, title) in zip(axes, measure):
        vals = summary.set_index("Model").loc[names, col]
        bars = ax.barh(names, vals, color=[_CLASSIFICATION_COLORS.get(n, "#64748b") for n in names], height=.63)
        for bar, value in zip(bars, vals):
            ax.text(min(value + .012, 1.018), bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.10)
        ax.set_title(title, fontsize=11, pad=9)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=.35, right=.94, top=.94, bottom=.06, hspace=.42)
    figures["comparison"] = output_dir / "4_2_comparison_metrics.png"
    _save_classification_figure(fig, figures["comparison"])

    fig, axes = create_a4_figure(2, 2, figsize=(8.27, 11.69), facecolor="white")
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, models):
        cm = plot_data["confusions"][name]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        ax.set_title(name, fontsize=10, pad=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=13,
                        color="white" if cm[i,j] > cm.max()*.55 else "black")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Факт")
    for ax in axes_flat[len(models):]:
        ax.axis("off")
    fig.subplots_adjust(left=.15, right=.95, top=.94, bottom=.06, wspace=.35, hspace=.28)
    figures["confusion"] = output_dir / "4_3_confusion_matrices.png"
    _save_classification_figure(fig, figures["confusion"])

    fig, ax = _classification_figure()
    for name in models:
        fpr, tpr, auc = plot_data["roc"][name]
        ax.plot(fpr, tpr, lw=2, color=_CLASSIFICATION_COLORS.get(name), label=f"{name}, AUC={auc:.3f}")
    ax.plot([0,1], [0,1], linestyle="--", color="#9299a3", lw=1.2, label="Случайный ориентир")
    ax.set_xlabel("FPR — доля ложноположительных")
    ax.set_ylabel("TPR — полнота")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=.16)
    ax.legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(left=.17, right=.95, top=.93, bottom=.14)
    figures["roc"] = output_dir / "4_4_roc_comparison.png"
    _save_classification_figure(fig, figures["roc"])

    fig, ax = _classification_figure()
    for name in models:
        recall, precision, ap = plot_data["pr"][name]
        ax.plot(recall, precision, lw=2, color=_CLASSIFICATION_COLORS.get(name), label=f"{name}, AP={ap:.3f}")
    ax.axhline(plot_data["prevalence"], ls="--", lw=1.2, color="#9299a3",
               label=f"Частота класса 1: {plot_data["prevalence"]:.1%}")
    ax.set_xlabel("Recall — полнота")
    ax.set_ylabel("Precision — точность положительных решений")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=.16)
    ax.legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(left=.17, right=.95, top=.93, bottom=.14)
    figures["pr"] = output_dir / "4_5_precision_recall.png"
    _save_classification_figure(fig, figures["pr"])

    if mlp_history is not None and len(mlp_history):
        fig, (ax1, ax2) = create_a4_figure(2, 1, figsize=(8.27, 11.69), facecolor="white")
        e = mlp_history["Epoch"]
        ax1.plot(e, mlp_history["TrainingLoss"], color=_CLASSIFICATION_COLORS["MLPClassifier"], linewidth=2)
        ax1.set_ylabel("Training loss")
        ax1.set_xlabel("Эпоха")
        ax1.grid(alpha=.16)
        ax2.plot(e, mlp_history["ValidationAccuracy"], color=_CLASSIFICATION_COLORS["LogisticRegression"], linewidth=2)
        best = (mlp_details or {}).get("best_epoch")
        if best is not None:
            best_value = float(mlp_history.loc[mlp_history["Epoch"] == best, "ValidationAccuracy"].iloc[0])
            ax2.axvline(best, ls="--", color="#cd8d37", lw=1)
            ax2.scatter([best], [best_value], color="#cd8d37", zorder=5)
            ax2.annotate(f"Эпоха {best}, Accuracy={best_value:.3f}",
                         (best, best_value), xytext=(8, -20), textcoords="offset points", fontsize=9)
        ax2.set_xlabel("Эпоха")
        ax2.set_ylabel("Validation Accuracy")
        ax2.grid(alpha=.16)
        fig.subplots_adjust(left=.17, right=.94, top=.94, bottom=.10, hspace=.29)
        figures["mlp"] = output_dir / "4_6_mlp_training.png"
        _save_classification_figure(fig, figures["mlp"])
    return figures
