"""Визуализация результатов EDA для ЛР №1 (файлы остаются локально)."""

from __future__ import annotations

from pathlib import Path
from math import ceil

from matplotlib.transforms import Bbox

import matplotlib
matplotlib.use("Agg")  # Не требовать открытого окна для командного запуска.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from house_prices_ai.config import RANDOM_STATE


# Краткие подписи к исходным названиям полей. Смысл признаков
# соответствует официальному описанию Ames Housing на Kaggle.
FEATURE_LABELS_RU = {
    "MSSubClass": "Тип здания", "LotFrontage": "Ширина участка, футы",
    "LotArea": "Площадь участка, кв. футы", "OverallQual": "Качество дома (1–10)",
    "OverallCond": "Состояние дома (1–10)", "YearBuilt": "Год постройки",
    "YearRemodAdd": "Год реконструкции", "MasVnrArea": "Площадь облицовки, кв. футы",
    "BsmtFinSF1": "Отделанная площадь подвала №1", "BsmtFinSF2": "Отделанная площадь подвала №2",
    "BsmtUnfSF": "Неотделанная площадь подвала", "TotalBsmtSF": "Общая площадь подвала",
    "1stFlrSF": "Площадь 1-го этажа", "2ndFlrSF": "Площадь 2-го этажа",
    "LowQualFinSF": "Площадь низкого качества", "GrLivArea": "Жилая площадь, кв. футы",
    "BsmtFullBath": "Полные ванные в подвале", "BsmtHalfBath": "Полуванные в подвале",
    "FullBath": "Полные ванные комнаты", "HalfBath": "Полуванные комнаты",
    "BedroomAbvGr": "Число спален", "KitchenAbvGr": "Число кухонь",
    "TotRmsAbvGrd": "Число комнат без ванных", "Fireplaces": "Число каминов",
    "GarageYrBlt": "Год постройки гаража", "GarageCars": "Вместимость гаража (авто)",
    "GarageArea": "Площадь гаража", "WoodDeckSF": "Площадь деревянной террасы",
    "OpenPorchSF": "Площадь открытой веранды", "EnclosedPorch": "Площадь закрытой веранды",
    "3SsnPorch": "Трёхсезонная веранда", "ScreenPorch": "Веранда с сетками",
    "PoolArea": "Площадь бассейна", "MiscVal": "Стоимость доп. объектов, $",
    "MoSold": "Месяц продажи", "YrSold": "Год продажи",
    "SalePrice": "Цена продажи, $",
}


def _plot_title(feature: str) -> str:
    description = FEATURE_LABELS_RU.get(feature, "")
    return f"{feature}\n{description}" if description else feature


def _finish_plot(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    """Сохранить рисунок с одинаковыми внешними полями и книжной ориентацией.

    Реальную область надписей и графиков измеряем после отрисовки, чтобы
    не оставлять пустые строки сетки и не обрезать длинные подписи/легенды.
    Формат A4 задаётся в документе при вставке, а PNG сохраняет пропорции.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())  # Координаты в дюймах.

    padding = 0.18  # Одинаковый отступ от крайних видимых элементов.
    left, bottom = bbox.x0 - padding, bbox.y0 - padding
    right, top = bbox.x1 + padding, bbox.y1 + padding
    width = right - left
    height = top - bottom
    # Некоторые компактные рисунки почти квадратные. Добавляем белое поле
    # поровну сверху и снизу, чтобы все файлы оставались книжными.
    min_height = width * 1.06
    if height < min_height:
        extra = (min_height - height) / 2
        bottom -= extra
        top += extra

    # Используем явный bbox: в отличие от сохранения полной фигуры,
    # на страницах с 5 признаками не возникает огромного белого подвала.
    fig.savefig(path, dpi=dpi, facecolor="white", pad_inches=0,
                bbox_inches=Bbox.from_extents(left, bottom, right, top))
    plt.close(fig)


def _sheet_axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    """Компактная страница: до восьми панелей, без пустого четвёртого ряда."""
    if not 1 <= count <= 8:
        raise ValueError("Количество панелей должно быть от 1 до 8")
    rows = ceil(count / 2)
    height = 0.70 + 2.13 * rows
    fig, axes = plt.subplots(rows, 2, figsize=(6.6, height), squeeze=False)
    fig.subplots_adjust(left=.15, right=.955, top=.955, bottom=.065,
                        hspace=.82, wspace=.44)
    flat = axes.reshape(-1)
    if count % 2:
        # Нечётный последний график располагаем по центру страницы,
        # а не в левой половине листа.
        last = flat[count - 1]
        position = last.get_position()
        last.set_position([.5 - position.width / 2, position.y0,
                           position.width, position.height])
    return fig, flat


def _draw_boxplot(ax: plt.Axes, values: np.ndarray) -> dict:
    """Отображать отдельные точки за пределами 1,5×IQR заметно при печати A4.

    Для дискретных признаков это лишь формальный критерий, не признак ошибки.
    Возвращаем объект Matplotlib для возможности проверить наличие точек в тесте.
    """
    return ax.boxplot(
        values, patch_artist=True, orientation="vertical",
        whis=1.5, showfliers=True, widths=.45,
        boxprops={"facecolor": "#b8cde1", "edgecolor": "#42698e", "linewidth": 1.0},
        medianprops={"color": "#b2604f", "linewidth": 1.5},
        whiskerprops={"color": "#42698e", "linewidth": 1},
        capprops={"color": "#42698e", "linewidth": 1},
        flierprops={"marker": "o", "markersize": 3.2, "markerfacecolor": "#333333",
                    "markeredgecolor": "#333333", "markeredgewidth": .25, "alpha": .75},
    )


def save_eda_plots(data: pd.DataFrame, dest: Path, *, price_threshold: int) -> dict[str, Path]:
    """Построить рисунки ЛР №1 с учётом книжного формата A4.

    Для 01/02 создаём по 8 диаграмм на страницу; корреляции — на одном
    портретном листе; pairplot — на двух листах. Все иллюстрации книжные.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.85)
    numeric = data.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    numeric = numeric.loc[:, numeric.notna().any()]
    if "SalePrice" not in numeric.columns:
        raise ValueError("Нет числовой целевой переменной SalePrice")
    result: dict[str, Path] = {}
    columns = list(numeric.columns)

    # 01: гистограммы (5 страниц для 37 признаков).
    for p, offset in enumerate(range(0, len(columns), 8)):
        page_columns = columns[offset:offset + 8]
        fig, axes = _sheet_axes(len(page_columns))
        for ax, col in zip(axes, page_columns):
            values = numeric[col].dropna()
            # Дискретные признаки рисуем с целочисленными интервалами,
            # иначе для признаков вроде GarageCars гистограмма misleading.
            bins = 25
            if col in {"GarageCars", "Fireplaces", "FullBath", "HalfBath", "MoSold", "OverallQual", "OverallCond"}:
                unique_vals = np.sort(values.unique())
                if len(unique_vals) < 20:
                    bins = np.arange(values.min() - .5, values.max() + 1.5, 1)
            ax.hist(values, bins=bins, color="#537fab", edgecolor="white", linewidth=.35)
            ax.set_title(_plot_title(col), fontsize=9.0, pad=11, linespacing=1.18)
            ax.tick_params(labelsize=7.6)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for ax in axes[len(page_columns):]:
            ax.set_visible(False)
        file = dest / f"01{chr(97+p)}_numeric_histograms.png"
        _finish_plot(fig, file)
        result[f"histograms_{p+1:02d}"] = file

    # 02: диаграммы размаха. Отдельные точки за усами показываются явно.
    for p, offset in enumerate(range(0, len(columns), 8)):
        page_columns = columns[offset:offset + 8]
        fig, axes = _sheet_axes(len(page_columns))
        for ax, col in zip(axes, page_columns):
            values = numeric[col].dropna().to_numpy()
            _draw_boxplot(ax, values)
            ax.set_title(_plot_title(col), fontsize=9.0, pad=11, linespacing=1.18)
            ax.tick_params(axis="y", labelsize=7.6)
            ax.set_xticks([])
        for ax in axes[len(page_columns):]:
            ax.set_visible(False)
        file = dest / f"02{chr(97+p)}_numeric_boxplots.png"
        _finish_plot(fig, file)
        result[f"boxplots_{p+1:02d}"] = file

    # 03: ценовая цель. На весь рисунок оставляем только аналитические подписи.
    fig, ax = plt.subplots(figsize=(6.6, 8.2))
    sns.histplot(data=data, x="SalePrice", bins=35, kde=True, color="#537fab", ax=ax)
    ax.axvline(price_threshold, linestyle="--", color="#b76555",
               label=f"Порог класса: ${price_threshold:,}")
    ax.set_xlabel("Цена продажи, доллары")
    ax.set_ylabel("Количество объектов")
    ax.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=.14, right=.965, bottom=.12, top=.95)
    file = dest / "03_saleprice_distribution.png"
    _finish_plot(fig, file)
    result["target"] = file

    # 04: полная корреляционная матрица на одном книжном листе.
    # Подписи остаются техническими: их расшифровка — в Markdown Notebook.
    corr = numeric.corr()
    # Больше полезной площади листа: увеличиваем матрицу, а не вставляем
    # растянутую готовую картинку. Все коэффициенты считаются на лету.
    fig, ax = plt.subplots(figsize=(7.45, 8.25))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, square=False, linewidths=.10, linecolor="white",
                cbar_kws={"shrink": .82, "pad": .012, "fraction": .035,
                          "ticks": [-1, -.5, 0, .5, 1], "format": "%.1f"}, ax=ax)
    ax.set_box_aspect(1.14)  # Чуть выше, чтобы заполнить книжный лист без белых полос.
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7.0, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.0, rotation=0)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=.13, right=.945, top=.99, bottom=.12)
    file = dest / "04_correlation_heatmap.png"
    _finish_plot(fig, file)
    result["correlations"] = file

    # 05: две читаемые группы pairplot; на обеих есть SalePrice.
    target_corr = corr["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
    important = target_corr.index[: min(5, len(target_corr))].tolist()
    groups = [important[:3], important[3:]]
    for p, group in enumerate(groups):
        if len(group) < 1:
            continue
        subset = data[group + ["SalePrice"]].dropna()
        if len(subset) > 400:
            subset = subset.sample(n=400, random_state=RANDOM_STATE)
        pair = sns.pairplot(subset, diag_kind="hist",
                            plot_kws={"s": 13, "alpha": .5, "color": "#537fab"},
                            diag_kws={"color": "#537fab"})
        pair.figure.set_size_inches(6.6, 8.1)
        pair.figure.subplots_adjust(left=.15, right=.96, top=.96, bottom=.10,
                                    hspace=.13, wspace=.13)
        from matplotlib.ticker import MaxNLocator
        for ax in pair.axes.flat:
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
        file = dest / f"05{chr(97+p)}_pairplot.png"
        _finish_plot(pair.figure, file)
        result[f"pairplot_{p+1:02d}"] = file

    # 06: двумерный PCA только для визуальной разведки данных.
    pca_features = numeric.drop(columns=["SalePrice"], errors="ignore")
    pca_features = pca_features.loc[:, pca_features.nunique(dropna=True) > 1]
    if pca_features.shape[1] >= 2:
        imputed = SimpleImputer(strategy="median").fit_transform(pca_features)
        transformed = StandardScaler().fit_transform(imputed)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        points = pca.fit_transform(transformed)
        fig, ax = plt.subplots(figsize=(6.6, 8.1))
        ax.scatter(points[:, 0], points[:, 1], c="#537fab", s=16, alpha=.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)")
        fig.subplots_adjust(left=.14, right=.96, top=.96, bottom=.12)
        file = dest / "06_pca_numeric_features.png"
        _finish_plot(fig, file)
        result["pca"] = file

    return result


# === Общие функции оформления изображений проекта ===
# Не изменяют математическую логику визуализаций EDA.
def create_a4_figure(**kwargs):
    """Книжная A4-страница с белым фоном для графиков следующих этапов."""
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8.27, 11.69), facecolor="white", **kwargs)


def save_a4_figure(fig, path, *, dpi=200, pad_inches=0.22):
    """Сохраняет график на белом фоне с полями для полных подписей."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_figwidth() >= fig.get_figheight():
        raise ValueError(f"Ожидалась книжная ориентация рисунка: {path}")
    fig.set_facecolor("white")
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                pad_inches=pad_inches, facecolor="white")
    plt.close(fig)
