"""Поиск сегментов недвижимости без использования цены при кластеризации.

K-means и DBSCAN обучаются только по характеристикам тренировочных домов.
SalePrice применяется после построения кластеров для описательного анализа.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from house_prices_ai.config import OUTPUT_DIR, RANDOM_STATE
from house_prices_ai.preprocessing import HouseFeatures, IQRClipper
from house_prices_ai.regression import load_lr02_split
from house_prices_ai.visualization import create_clustering_figures

# Осознанно не включаем SalePrice, PriceClass, Id, OHE и несколько почти
# дублирующих друг друга площадей. Это исследовательский выбор признаков.
CLUSTER_FEATURES = (
    "OverallQual", "OverallCond", "TotalUsefulSF", "TotalBathrooms",
    "GarageCars", "PropertyAgeAtSale", "LotArea", "BsmtFinSF1",
)
FRIENDLY = {
    "OverallQual": "Общее качество",
    "OverallCond": "Общее состояние",
    "TotalUsefulSF": "Суммарная площадь",
    "TotalBathrooms": "Санузлы (экв.)",
    "GarageCars": "Вместимость гаража",
    "PropertyAgeAtSale": "Возраст дома",
    "LotArea": "Площадь участка",
    "BsmtFinSF1": "Отделанный подвал",
}


@dataclass
class ClusteringResult:
    selected_k: int
    selection_path: Path
    assignments_path: Path
    profile_path: Path
    report_path: Path
    figures: dict[str, Path]


def prepare_features(raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, Pipeline]:
    """Обучает обработку только на исследуемой обучающей части.

    Для интерпретируемости исключаем OHE и используем восемь предметных числовых
    характеристик. Структурные нули не превращаем в пропуски.
    """
    if any(c in raw.columns for c in ("SalePrice", "PriceClass", "Id")):
        raise ValueError("Целевые переменные и идентификатор нельзя подавать в кластеризацию")
    engineered = HouseFeatures().transform(raw)
    absent = [c for c in CLUSTER_FEATURES if c not in engineered.columns]
    if absent:
        raise ValueError(f"Отсутствуют входные характеристики: {absent}")
    data = engineered.loc[:, CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    if len(data) < 15:
        raise ValueError("Для анализа кластеров требуется не меньше 15 домов")
    if data.isna().all().any():
        raise ValueError("Одна из выбранных характеристик полностью пуста")
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", IQRClipper(tuple(CLUSTER_FEATURES),
                            target_columns=("LotArea", "TotalUsefulSF", "BsmtFinSF1"))),
        ("scale", StandardScaler()),
    ])
    features = pipeline.fit_transform(data)
    if not np.isfinite(features).all():
        raise ValueError("Найдены нечисловые результаты преобразования")
    clean = pipeline.named_steps["clip"].transform(
        pipeline.named_steps["impute"].transform(data)
    )
    cleaned = pd.DataFrame(clean, columns=CLUSTER_FEATURES, index=raw.index)
    return features, cleaned, pipeline


def select_cluster_count(features: np.ndarray, *, max_k: int = 8,
                         seed: int = RANDOM_STATE,
                         min_fraction: float = .05,
                         silhouette_sample: int = 800) -> tuple[pd.DataFrame, int]:
    """Подбирает k только по характеристикам, без цен и меток классов."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or len(x) < 15 or not np.isfinite(x).all():
        raise ValueError("Ожидается числовая матрица из 15+ объектов")
    rows = []
    for k in range(2, min(max_k, len(x) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=15, random_state=seed)
        labels = km.fit_predict(x)
        if len(set(labels)) < 2:
            continue
        # Вычисляем silhouette по подвыборке для предсказуемого времени запуска.
        sil = silhouette_score(x, labels, sample_size=min(silhouette_sample, len(x)),
                               random_state=seed)
        sizes = np.bincount(labels, minlength=k)
        rows.append({
            "k": int(k), "Inertia": float(km.inertia_),
            "Silhouette": float(sil),
            "CalinskiHarabasz": float(calinski_harabasz_score(x, labels)),
            "DaviesBouldin": float(davies_bouldin_score(x, labels)),
            "MinClusterSize": int(sizes.min()),
            "MinClusterFraction": float(sizes.min() / len(x)),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Не удалось получить хотя бы два разных кластера")
    candidates = table.loc[table.MinClusterFraction >= min_fraction]
    if candidates.empty:
        candidates = table  # явно укажем это ограничение в отчёте
    best = candidates.sort_values(["Silhouette", "k"], ascending=[False, True]).iloc[0]
    return table, int(best.k)


def describe_clusters(cleaned: pd.DataFrame, labels: np.ndarray,
                      prices: pd.Series, threshold: float = 200_000) -> pd.DataFrame:
    """Цена применяется ПОСЛЕ построения кластеров, только для описания групп."""
    if len(cleaned) != len(labels) or len(labels) != len(prices):
        raise ValueError("Несовпадение длины характеристик, меток и цен")
    table = cleaned.copy()
    table["Cluster"] = np.asarray(labels, dtype=int)
    table["SalePrice"] = np.asarray(prices, dtype=float)
    table["Expensive"] = (table.SalePrice >= threshold).astype(int)
    summary = table.groupby("Cluster", sort=True).agg(
        Count=("SalePrice", "size"),
        MeanPriceUSD=("SalePrice", "mean"),
        MedianPriceUSD=("SalePrice", "median"),
        ExpensiveShare=("Expensive", "mean"),
        MeanTotalUsefulSF=("TotalUsefulSF", "mean"),
        MeanOverallQual=("OverallQual", "mean"),
        MeanAge=("PropertyAgeAtSale", "mean"),
    ).reset_index()
    return summary


def analyze_dbscan(features: np.ndarray, eps: float = 1.8,
                   min_samples: int = 10) -> tuple[np.ndarray, dict]:
    """Сравнение по плотности: -1 обозначает шум, один кластер не считается успехом."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_labels = sorted(v for v in counts.index if v >= 0)
    non_noise = labels != -1
    score = None
    if len(cluster_labels) >= 2 and len(set(labels[non_noise])) < non_noise.sum():
        score = float(silhouette_score(features[non_noise], labels[non_noise],
                                       sample_size=min(800, int(non_noise.sum())),
                                       random_state=RANDOM_STATE))
    return labels, {
        "n_clusters": len(cluster_labels), "noise_count": int((labels == -1).sum()),
        "noise_fraction": float((labels == -1).mean()),
        "silhouette_without_noise": score,
        "eps": float(eps), "min_samples": int(min_samples),
    }


def prepare_clustering_plot_data(
    x: np.ndarray, labels: np.ndarray, prices: pd.Series,
    k_table: pd.DataFrame, selected_k: int, *,
    tsne: bool = False, dbscan_labels: np.ndarray | None = None,
) -> tuple[dict, PCA]:
    """Подготовить агрегаты и проекции для визуализации без построения графиков.

    Цены используются исключительно для описания уже полученных кластеров.
    """
    plot_labels = sorted(np.unique(labels))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x)
    profiles = pd.DataFrame(x, columns=CLUSTER_FEATURES).assign(Cluster=labels)
    centers = profiles.groupby("Cluster")[list(CLUSTER_FEATURES)].mean()
    groups = [np.asarray(prices)[labels == cl] for cl in plot_labels]
    class_labels = (np.asarray(prices) >= 200_000).astype(int)
    table = pd.crosstab(pd.Series(labels, name="Cluster"),
                        pd.Series(class_labels, name="PriceClass"), normalize="index")
    table = table.reindex(index=plot_labels, columns=[0, 1], fill_value=0)
    two_d = None
    if tsne:
        projector = TSNE(n_components=2, random_state=RANDOM_STATE,
                         perplexity=min(30, max(5, (len(x)-1)//4)),
                         learning_rate="auto", init="pca")
        two_d = projector.fit_transform(x)
    return {
        "labels": labels, "plot_labels": plot_labels,
        "k_table": k_table, "selected_k": selected_k,
        "coords": coords, "explained": pca.explained_variance_ratio_,
        "centers": centers, "groups": groups, "price_classes": table,
        "features": CLUSTER_FEATURES, "friendly": FRIENDLY,
        "tsne_coords": two_d, "dbscan_labels": dbscan_labels,
    }, pca


def run_clustering(*, out_dir: str | Path = OUTPUT_DIR / "clustering",
                   raw_path: str | Path | None = None,
                   processed_dir: str | Path | None = None,
                   max_k: int = 8, make_plots: bool = True,
                   tsne: bool = False, dbscan: bool = False,
                   dbscan_eps: float = 1.8) -> ClusteringResult:
    """Полный анализ только на прежних тренировочных Id, без использования test."""
    load_kw = {}
    if raw_path is not None: load_kw["raw_path"] = raw_path
    if processed_dir is not None: load_kw["processed_dir"] = processed_dir
    raw, _, prices, _ = load_lr02_split(**load_kw)
    x, cleaned, transformer = prepare_features(raw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "figures"
    table, k = select_cluster_count(x, max_k=max_k)
    model = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
    labels = model.fit_predict(x)
    profile = describe_clusters(cleaned, labels, prices)
    selection_path = out_dir / "cluster_k_selection.csv"
    assignment_path = out_dir / "cluster_assignments.csv"
    profile_path = out_dir / "cluster_profiles.csv"
    table.to_csv(selection_path, index=False)
    pd.DataFrame({"Id": raw.index.to_numpy(), "Cluster": labels}).to_csv(assignment_path, index=False)
    profile.to_csv(profile_path, index=False)
    group_summary = cleaned.copy()
    group_summary["Cluster"] = labels
    group_summary.groupby("Cluster").mean().to_csv(out_dir / "cluster_mean_characteristics.csv")

    density_labels, density_summary = (None, None)
    if dbscan:
        density_labels, density_summary = analyze_dbscan(x, eps=dbscan_eps)
        pd.DataFrame({"Id": raw.index.to_numpy(), "ClusterDBSCAN": density_labels}).to_csv(
            out_dir / "dbscan_assignments.csv", index=False)
    if make_plots:
        plot_data, pca = prepare_clustering_plot_data(
            x, labels, prices, table, k, tsne=tsne, dbscan_labels=density_labels
        )
        figures = create_clustering_figures(plot_data, figure_dir)
    else:
        figures = {}
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(x)
    best = table.loc[table.k == k].iloc[0]
    dominant = profile.sort_values("MeanPriceUSD", ascending=False).iloc[0]
    lines = [
        "# Поиск групп недвижимости методом кластеризации", "",
        "## Постановка задачи", "",
        f"Исследована структура **{len(raw):,}** объектов из ранее выделенной обучающей выборки. "
        "Внешняя тестовая часть не использовалась.",
        "Кластеры строились без SalePrice и без бинарной ценовой метки. "
        "Цены сопоставлены с найденными группами только после обучения.",
        "Для кластеризации выбраны восемь предметных числовых характеристик: "
        + ", ".join(FRIENDLY[v].lower() for v in CLUSTER_FEATURES) + ".",
        "Категориальные OHE-признаки исключены из первой постановки, чтобы редкие "
        "категории не получили чрезмерный вес при вычислении евклидовых расстояний.", "",
        "## Подбор числа групп", "",
        f"K-means: выбран k = **{k}**, коэффициент силуэта = **{best.Silhouette:.4f}**, "
        f"инерция = **{best.Inertia:.2f}**. Проверены варианты k от 2 до {int(table.k.max())}.",
        "Критерий выбора — максимальный коэффициент силуэта среди вариантов, "
        "в которых каждая группа содержит хотя бы 5% объектов. "
        "Если условие не выполняется ни для одного k, выбирается лучший из всех вариантов.",
        "Силуэт оценивался на детерминированной случайной подвыборке до 800 домов. "
        "PCA использована для двумерной визуализации и не участвовала в подборе кластеров.",
        f"Две компоненты PCA объясняют **{sum(pca.explained_variance_ratio_):.1%}** "
        "вариации подготовленных характеристик.", "",
        "## Описание групп", "",
        "Группы получены без знания цен. По завершении кластеризации рассчитаны "
        "численность, средняя и медианная стоимость, а также доля домов стоимостью от $200 000.",
    ]
    for row in profile.itertuples(index=False):
        lines.append(f"Кластер {row.Cluster}: **{row.Count}** домов, средняя цена "
                     f"**${row.MeanPriceUSD:,.0f}**, медианная **${row.MedianPriceUSD:,.0f}**, "
                     f"доля дорогих объектов **{row.ExpensiveShare:.1%}**.")
    lines += [
        f"Наибольшая средняя цена обнаружена в кластере **{int(dominant.Cluster)}**. "
        "Это описательная ассоциация, а не причинная зависимость.", "",
        "## Ограничения интерпретации", "",
        "Силуэт характеризует геометрическое разделение групп в выбранном пространстве признаков. "
        "Его значение не гарантирует, что найденные группы пригодны для улучшения прогноза цены.",
        "Число кластеров и их содержимое зависят от выбора характеристик, "
        "масштабирования, ограничений экстремальных значений и метрики расстояния.",
        "Для доказательства пользы кластеризации в прогнозных моделях потребовалась бы "
        "дополнительная вложенная валидация, исключающая утечки при формировании групп.",
    ]
    if dbscan and density_summary:
        lines.extend(["", "## Дополнительное сопоставление DBSCAN", "",
                      f"При eps={density_summary['eps']:.2f} и "
                      f"min_samples={density_summary['min_samples']} обнаружено "
                      f"{density_summary['n_clusters']} групп, доля шумовых объектов "
                      f"{density_summary['noise_fraction']:.1%}.",
                      "DBSCAN исследован как дополнительный метод, без заявления об оптимальности eps."])
    report_path = out_dir / "clustering_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"K-means: k={k}, Silhouette={best.Silhouette:.4f}, PCA={sum(pca.explained_variance_ratio_):.1%}")
    print(f"Сохранено изображений: {len(figures)}")
    print(f"Отчёт: {report_path}")
    return ClusteringResult(k, selection_path, assignment_path, profile_path, report_path, figures)
