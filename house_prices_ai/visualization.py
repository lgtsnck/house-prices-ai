"""Общие средства оформления графиков House Prices AI и иллюстрации EDA."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from math import ceil

import matplotlib
matplotlib.use("Agg")  # Не требовать открытого окна для командного запуска.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats



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
    """Сохранить страницу EDA, сохранив поля и книжную ориентацию.

    Ориентация контролируется здесь, а не в общем save_a4_figure:
    строгая проверка последнего остаётся полезной для остальных модулей.
    """
    width, height = (float(v) for v in fig.get_size_inches())
    if width >= height:
        # В EDA все страницы должны быть книжными. Исправляем только
        # неожиданно альбомный или квадратный холст, а не макет осей.
        if width > height:
            new_width, new_height = height, width
        else:
            new_width, new_height = width, width * (11.69 / 8.27)
        fig.set_size_inches(new_width, new_height, forward=False)
        print(
            f"EDA: исправлена ориентация {path.name}: "
            f"{width:.2f}×{height:.2f} → {new_width:.2f}×{new_height:.2f} дюймов"
        )
    save_a4_figure(fig, path, dpi=dpi, bbox_inches=None)


def _sheet_axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    """Компактная страница: до восьми панелей, без пустого четвёртого ряда."""
    if not 1 <= count <= 8:
        raise ValueError("Количество панелей должно быть от 1 до 8")
    rows = ceil(count / 2)
    height = 0.70 + 2.13 * rows
    fig, axes = create_report_figure(rows, 2, figsize=(6.6, height), squeeze=False)
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


def save_eda_plots(
    numeric: pd.DataFrame, dest: Path, *, price_threshold: int,
    correlations: pd.DataFrame,
    pairplot_samples: tuple[pd.DataFrame, ...],
    pca_points: np.ndarray | None,
    pca_explained_variance: tuple[float, float] | None,
) -> dict[str, Path]:
    """Нарисовать EDA, используя исключительно подготовленные аналитические данные."""
    # Контекст вместо глобального sns.set_theme: настройки других этапов не меняем.
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook", font_scale=0.85):
        return _render_eda_plots(
            numeric, dest, price_threshold=price_threshold,
            correlations=correlations, pairplot_samples=pairplot_samples,
            pca_points=pca_points, pca_explained_variance=pca_explained_variance,
        )


def _render_eda_plots(
    numeric: pd.DataFrame, dest: Path, *, price_threshold: int,
    correlations: pd.DataFrame,
    pairplot_samples: tuple[pd.DataFrame, ...],
    pca_points: np.ndarray | None,
    pca_explained_variance: tuple[float, float] | None,
) -> dict[str, Path]:
    """Построить EDA-рисунки в книжной ориентации A4.

    Для 01/02 создаём по 8 диаграмм на страницу; корреляции — на одном
    портретном листе; pairplot — на двух листах. Все иллюстрации книжные.
    """
    dest.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    columns = list(numeric.columns)

    # 01: гистограммы числовых характеристик, до восьми панелей на страницу.
    for p, offset in enumerate(range(0, len(columns), 8)):
        page_columns = columns[offset:offset + 8]
        fig, axes = _sheet_axes(len(page_columns))
        for ax, col in zip(axes, page_columns):
            values = numeric[col].dropna()
            # Дискретные признаки рисуем с целочисленными интервалами,
            # иначе интервалы для дискретных признаков (например, GarageCars) будут некорректными.
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
    fig, ax = create_report_figure(figsize=(6.6, 8.2))
    sns.histplot(data=numeric, x="SalePrice", bins=35, kde=True, color="#537fab", ax=ax)
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
    # Увеличиваем матрицу в макете, не растягивая готовое изображение.
    fig, ax = create_report_figure(figsize=(7.45, 8.25))
    sns.heatmap(correlations, cmap="coolwarm", center=0, vmin=-1, vmax=1,
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
    for p, subset in enumerate(pairplot_samples):
        if subset.empty:
            continue
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

    # 06: отображение подготовленных координат PCA, без повторных вычислений.
    if pca_points is not None and pca_explained_variance is not None:
        fig, ax = create_report_figure(figsize=(6.6, 8.1))
        ax.scatter(pca_points[:, 0], pca_points[:, 1], c="#537fab", s=16, alpha=.5)
        ax.set_xlabel(f"PC1 ({pca_explained_variance[0]:.1%} дисперсии)")
        ax.set_ylabel(f"PC2 ({pca_explained_variance[1]:.1%} дисперсии)")
        fig.subplots_adjust(left=.14, right=.96, top=.96, bottom=.12)
        file = dest / "06_pca_numeric_features.png"
        _finish_plot(fig, file)
        result["pca"] = file

    return result


# Общие функции оформления для всех этапов исследования.
# Размеры и существующие параметры сохранения сохранены ради совместимости.
A4_PORTRAIT_INCHES = (8.27, 11.69)
A4_RASTER_SIZE_180DPI = (1488, 2104)


def create_report_figure(*args, figsize=A4_PORTRAIT_INCHES, **kwargs):
    """Создать белую фигуру с той же сигнатурой, что create_report_figure().

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

# Графики предобработки: расчёты выполнены в preprocessing.py.
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
        ax.boxplot(values, tick_labels=model_order, vert=False, patch_artist=True,
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
