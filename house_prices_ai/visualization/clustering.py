"""Графики диагностики и сравнения результатов кластеризации."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import create_a4_figure, save_a4_figure


# Изображения кластеризации. Числовые проекции и агрегаты рассчитываются в clustering.py.
_CLUSTER_COLORS = ("#477BAF", "#C78A41", "#398D76", "#8B71AE", "#B65E68", "#76944C", "#4D96A8", "#BD916D")


def _save_clustering_figure(fig: plt.Figure, path: Path) -> None:
    save_a4_figure(fig, path, dpi=200, pad_inches=.22)


def create_clustering_figures(data: dict, out: Path) -> dict[str, Path]:
    """Построить восемь диаграмм из подготовленных числовых данных."""
    out.mkdir(parents=True, exist_ok=True)
    figures = {}
    plot_labels = data["plot_labels"]
    labels = data["labels"]
    k_table = data["k_table"]
    selected_k = data["selected_k"]
    coords = data["coords"]
    explained = data["explained"]
    centers = data["centers"]
    groups = data["groups"]
    table = data["price_classes"]
    feature_names = data["features"]
    friendly = data["friendly"]
    two_d = data["tsne_coords"]
    dbscan_labels = data["dbscan_labels"]
    cmap = {c: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, c in enumerate(plot_labels)}

    # Рисунок 6.1. Выбор k. Вертикальная компоновка подходит для A4.
    fig, ax = create_a4_figure()
    ax.plot(k_table.k, k_table.Inertia, marker="o", color=_CLUSTER_COLORS[0])
    ax.axvline(selected_k, color=_CLUSTER_COLORS[1], ls="--", alpha=.7)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Инерция (внутрикластерная сумма квадратов)")
    ax.grid(alpha=.2)
    fig.subplots_adjust(left=.16, right=.95, bottom=.12, top=.95)
    p = out / "6_1_elbow.png"; _save_clustering_figure(fig, p); figures["elbow"] = p

    fig, ax = create_a4_figure()
    ax.plot(k_table.k, k_table.Silhouette, marker="o", color=_CLUSTER_COLORS[2])
    ax.axvline(selected_k, color=_CLUSTER_COLORS[1], ls="--", alpha=.7)
    ax.set_xlabel("Число кластеров k")
    ax.set_ylabel("Silhouette Score (коэффициент силуэта)")
    ax.grid(alpha=.2)
    fig.subplots_adjust(left=.16, right=.95, bottom=.12, top=.95)
    p = out / "6_2_silhouette.png"; _save_clustering_figure(fig, p); figures["silhouette"] = p

    fig, ax = create_a4_figure()
    for cl in plot_labels:
        pts = labels == cl
        ax.scatter(coords[pts, 0], coords[pts, 1], s=13, alpha=.66,
                   color=cmap[cl], label=f"Кластер {cl}, n={pts.sum()}")
    ax.set_xlabel(f"PC1 (объяснено {explained[0]:.1%})")
    ax.set_ylabel(f"PC2 (объяснено {explained[1]:.1%})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=.15)
    fig.subplots_adjust(left=.17, right=.95, bottom=.12, top=.96)
    p = out / "6_3_clusters_pca.png"; _save_clustering_figure(fig, p); figures["pca"] = p

    # Профиль по выбранным восьми характеристикам на стандартизированной шкале.
    fig, ax = create_a4_figure()
    positions = np.arange(len(feature_names))
    width = .8 / len(plot_labels)
    for i, cl in enumerate(plot_labels):
        ax.barh(positions + (i - (len(plot_labels)-1)/2)*width, centers.loc[cl],
                height=width, label=f"Кластер {cl}", color=cmap[cl])
    ax.set_yticks(positions, [friendly.get(v, v) for v in feature_names], fontsize=9)
    ax.axvline(0, color="gray", lw=.8)
    ax.set_xlabel("Отклонение от среднего, стандартных отклонений")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", alpha=.18)
    fig.subplots_adjust(left=.32, right=.96, bottom=.12, top=.96)
    p = out / "6_4_cluster_profiles.png"; _save_clustering_figure(fig, p); figures["profiles"] = p

    fig, ax = create_a4_figure()
    bp = ax.boxplot(groups, tick_labels=[f"{cl} (n={len(g)})" for cl, g in zip(plot_labels, groups)],
                    showfliers=True, patch_artist=True, widths=.5)
    for patch, cl in zip(bp["boxes"], plot_labels):
        patch.set_facecolor(cmap[cl]); patch.set_alpha(.42)
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Цена продажи, USD")
    ax.grid(axis="y", alpha=.2)
    fig.subplots_adjust(left=.17, right=.95, bottom=.12, top=.96)
    p = out / "6_5_price_by_cluster.png"; _save_clustering_figure(fig, p); figures["price"] = p

    fig, ax = create_a4_figure()
    ax.bar([str(v) for v in plot_labels], table[0], color=_CLUSTER_COLORS[0], label="Менее $200 000")
    ax.bar([str(v) for v in plot_labels], table[1], bottom=table[0], color=_CLUSTER_COLORS[1],
           label="От $200 000")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Доля домов")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=.15)
    fig.subplots_adjust(left=.16, right=.96, bottom=.12, top=.96)
    p = out / "6_6_price_classes.png"; _save_clustering_figure(fig, p); figures["classes"] = p

    if two_d is not None:
        fig, ax = create_a4_figure()
        for cl in plot_labels:
            pts = labels == cl
            ax.scatter(two_d[pts, 0], two_d[pts, 1], s=13, alpha=.65,
                       color=cmap[cl], label=f"Кластер {cl}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best")
        ax.grid(alpha=.15)
        fig.subplots_adjust(left=.16, right=.95, bottom=.12, top=.96)
        p = out / "6_7_clusters_tsne.png"; _save_clustering_figure(fig, p); figures["tsne"] = p

    if dbscan_labels is not None:
        fig, ax = create_a4_figure()
        for cl in sorted(np.unique(dbscan_labels)):
            filt = dbscan_labels == cl
            label = "Шум (-1)" if cl == -1 else f"Группа {cl}"
            color = "#9fa4ad" if cl == -1 else _CLUSTER_COLORS[int(cl) % len(_CLUSTER_COLORS)]
            ax.scatter(coords[filt, 0], coords[filt, 1], s=13, alpha=.65,
                       color=color, label=label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=.15)
        fig.subplots_adjust(left=.16, right=.95, bottom=.12, top=.96)
        p = out / "6_8_dbscan_pca.png"; _save_clustering_figure(fig, p); figures["dbscan"] = p
    return figures
