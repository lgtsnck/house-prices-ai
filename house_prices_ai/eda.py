"""ЛР №1: разведочный анализ House Prices и постановка трёх ML-задач."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices_ai.config import OUTPUT_DIR, RAW_TRAIN_FILE
from house_prices_ai.data_dictionary import FIELD_DESCRIPTIONS
from house_prices_ai.data_loader import load_training_data
from house_prices_ai.visualization import save_eda_plots

HIGH_PRICE_THRESHOLD = 200_000  # Предварительная фиксированная граница класса.


@dataclass
class EDAResult:
    """Итоги изучения данных; не содержит изменённого обучающего набора."""

    observations: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    duplicate_rows: int
    duplicate_feature_rows: int
    duplicate_ids: int
    missing_total: int
    missing: pd.DataFrame
    dictionary: pd.DataFrame
    numeric_summary: pd.DataFrame
    skewness: pd.Series
    kurtosis: pd.Series
    correlations: pd.Series
    correlated_pairs: pd.DataFrame
    class_balance: pd.DataFrame
    price_summary: pd.Series
    plots: dict[str, Path]
    report_path: Path


def _describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    roles = []
    for name in df.columns:
        if name == "Id":
            role = "Идентификатор"
        elif name == "SalePrice":
            role = "Целевая переменная регрессии"
        elif name == "MSSubClass":
            role = "Категориальный код (хранится числом)"
        elif pd.api.types.is_numeric_dtype(df[name]):
            role = "Числовой признак"
        else:
            role = "Категориальный признак"
        roles.append(role)
    return pd.DataFrame({
        "Признак": df.columns,
        "Описание": [FIELD_DESCRIPTIONS.get(name, "Уточнить по data_description.txt") for name in df.columns],
        "Тип данных": df.dtypes.astype(str).values,
        "Роль": roles,
        "Пропуски": df.isna().sum().values,
        "Пропуски, %": (df.isna().mean() * 100).round(2).values,
        "Уникальных значений": df.nunique(dropna=True).values,
    })


def _correlated_pairs(data: pd.DataFrame, limit: float = 0.8) -> pd.DataFrame:
    """Пары сильной корреляции без дублирования (кроме Id и SalePrice)."""
    numeric = data.select_dtypes(include=np.number).drop(columns=["Id", "SalePrice"], errors="ignore")
    matrix = numeric.corr()
    pairs = []
    for i, first in enumerate(matrix.columns):
        for second in matrix.columns[i + 1:]:
            coef = matrix.loc[first, second]
            if pd.notna(coef) and abs(coef) >= limit:
                pairs.append((first, second, float(coef)))
    return pd.DataFrame(pairs, columns=["Признак A", "Признак B", "Корреляция r"]).sort_values(
        "Корреляция r", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def _format_report(result: EDAResult) -> str:
    missing = result.missing.head(15)
    corr = result.correlations.head(10)
    class_counts = result.class_balance
    price = result.price_summary

    rows = [
        "# ЛР №1. Разведочный анализ House Prices и постановка задач",
        "",
        "## 1. Источник и цель исследования",
        "Набор данных: Kaggle House Prices — Advanced Regression Techniques (Ames Housing).",
        "Цель: изучить качество данных, распределение стоимости недвижимости и сформулировать",
        "задачи регрессии, классификации и кластеризации для последующих лабораторных работ.",
        "",
        "## 2. Структура и качество данных",
        f"- Объектов: **{result.observations}**; признаков/столбцов: **{result.columns}**.",
        f"- Числовых столбцов (включая Id и SalePrice): **{result.numeric_columns}**; текстовых/категориальных: **{result.categorical_columns}**.",
        f"- Пропущенных ячеек: **{result.missing_total}**.",
        f"- Полных дубликатов: **{result.duplicate_rows}**; совпадений признаков без Id: **{result.duplicate_feature_rows}**; повторов Id: **{result.duplicate_ids}**.",
        "",
        "### Признаки с наибольшим числом пропусков",
        "| Признак | Пропуски | Доля, % |",
        "|---|---:|---:|",
    ]
    for row in missing.itertuples(index=False):
        rows.append(f"| {row[0]} | {row[1]} | {row[2]:.2f} |")
    rows.extend([
        "",
        "Часть пропусков (например, Alley, PoolQC, GarageType) может обозначать",
        "**отсутствие соответствующего объекта**, а не потерю информации.",
        "Стратегии обработки будут выбраны в ЛР №2, исходный CSV не изменяется.",
        "",
        "## 3. Распределение цены продажи",
        f"- Минимум: **${price['min']:,.0f}**; медиана: **${price['50%']:,.0f}**; среднее: **${price['mean']:,.0f}**; максимум: **${price['max']:,.0f}**.",
        f"- Стандартное отклонение: **${price['std']:,.0f}**.",
        "- Оценить по графику 03 асимметрию и возможные выбросы; медиана лучше среднего описывает типичную стоимость при правой асимметрии.",
        "",
        "## 4. Корреляционный анализ",
        "Наибольшие по модулю коэффициенты Пирсона с SalePrice:",
        "| Признак | Коэффициент r |",
        "|---|---:|",
    ])
    for name, value in corr.items():
        rows.append(f"| {name} | {value:.3f} |")
    rows.extend([
        "",
        f"Число пар взаимосвязанных признаков с |r| ≥ 0,80: **{len(result.correlated_pairs)}**.",
        "Корреляция не означает причинно-следственную связь.",
        "",
        "## 5. Постановка задач машинного обучения (уровень «5»)",
        "1. **Регрессия** — предсказывать SalePrice (цена в долларах) по характеристикам объекта; метрики MAE, RMSE, R².",
        f"2. **Классификация** — предсказывать категорию цены: 0, если SalePrice < ${HIGH_PRICE_THRESHOLD:,}, 1 — иначе. Метрики Precision, Recall, F1 и ROC-AUC.",
        "   Порог предварительный: оценим баланс классов и при необходимости обоснуем изменение до ЛР №2.",
        "3. **Кластеризация** — искать группы домов на основе характеристик без Id, SalePrice и производных от них целевых признаков; K-means, PCA/t-SNE.",
        "",
        "### Баланс предварительной ценовой классификации",
        "| Класс | Количество | Доля, % |",
        "|---|---:|---:|",
    ])
    for row in class_counts.itertuples(index=False):
        rows.append(f"| {row[0]} | {row[1]} | {row[2]:.1f} |")
    rows.extend([
        "",
        "## 6. Выводы и ограничения",
        "- Обнаруженные пропуски, выбросы и распределения требуют осмысленной обработки на следующем этапе.",
        "- Id не используется как обучающий признак; SalePrice исключается из входов классификации и кластеризации.",
        "- PCA в ЛР №1 выполняется только для **описательной визуализации**. Его преобразования не используются для оценки моделей:",
        "  в ЛР №2 предобработка будет обучаться на тренировочной выборке для исключения утечки данных.",
        "- Выводы по конкретным графикам рекомендуется уточнить после просмотра фактических результатов.",
        "",
        "## 7. Перечень графиков",
    ])
    for key, path in result.plots.items():
        rows.append(f"- `{path.name}` — {key}")
    rows.extend([
        "",
        "Полный перечень всех исходных признаков с русскими описаниями — `feature_dictionary.csv`.",
        "Результаты сформированы автоматически на локальном датасете; исходные записи не публикуются.",
    ])
    return "\n".join(rows) + "\n"


def run_eda(
    data_path: Path | str = RAW_TRAIN_FILE,
    output_dir: Path | str = OUTPUT_DIR / "eda",
    *,
    make_plots: bool = True,
) -> EDAResult:
    """Запустить ЛР №1, сохранив только сводные результаты без исходных записей."""
    df = load_training_data(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary = _describe_dataset(df)
    missing = pd.DataFrame({
        "Признак": df.columns,
        "Пропуски": df.isna().sum().values,
        "Доля, %": (df.isna().mean().values * 100).round(2),
    }).sort_values(["Пропуски", "Признак"], ascending=[False, True]).reset_index(drop=True)
    missing = missing[missing["Пропуски"] > 0]
    numeric = df.select_dtypes(include=np.number)
    correlation = numeric.drop(columns=["Id"], errors="ignore").corr()["SalePrice"]
    correlation = correlation.drop("SalePrice").sort_values(key=lambda v: v.abs(), ascending=False)
    cls = (df["SalePrice"] >= HIGH_PRICE_THRESHOLD).astype(int)
    counts = cls.value_counts().reindex([0, 1], fill_value=0)
    class_balance = pd.DataFrame({
        "Класс": [f"0 — менее ${HIGH_PRICE_THRESHOLD:,}", f"1 — от ${HIGH_PRICE_THRESHOLD:,}"],
        "Количество": counts.to_numpy(),
        "Доля, %": (counts.to_numpy() / len(df) * 100).round(1),
    })
    plots = save_eda_plots(df, output_dir, price_threshold=HIGH_PRICE_THRESHOLD) if make_plots else {}
    result = EDAResult(
        observations=len(df),
        columns=len(df.columns),
        numeric_columns=len(numeric.columns),
        categorical_columns=len(df.select_dtypes(exclude=np.number).columns),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_feature_rows=int(df.drop(columns="Id", errors="ignore").duplicated().sum()),
        duplicate_ids=int(df["Id"].duplicated().sum()),
        missing_total=int(df.isna().sum().sum()),
        missing=missing,
        dictionary=dictionary,
        numeric_summary=numeric.drop(columns="Id", errors="ignore").describe().T,
        skewness=numeric.drop(columns="Id", errors="ignore").skew(),
        kurtosis=numeric.drop(columns="Id", errors="ignore").kurt(),
        correlations=correlation,
        correlated_pairs=_correlated_pairs(df),
        class_balance=class_balance,
        price_summary=df["SalePrice"].describe(),
        plots=plots,
        report_path=output_dir / "lr01_eda_summary.md",
    )
    dictionary.to_csv(output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output_dir / "missing_values.csv", index=False, encoding="utf-8-sig")
    result.numeric_summary.to_csv(output_dir / "numeric_summary.csv", encoding="utf-8-sig")
    result.correlated_pairs.to_csv(output_dir / "correlated_feature_pairs.csv", index=False, encoding="utf-8-sig")
    result.report_path.write_text(_format_report(result), encoding="utf-8")
    return result
