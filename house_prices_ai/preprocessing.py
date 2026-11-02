"""Подготовка House Prices к регрессии, классификации и кластеризации.

Все обучаемые преобразования подбираются исключительно по тренировочным
строкам. На тестовых строках используется только transform().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices_ai.config import PROCESSED_DATA_DIR, RANDOM_STATE, RAW_TRAIN_FILE, OUTPUT_DIR
from house_prices_ai.data_loader import load_training_data
from house_prices_ai.visualization import save_preprocessing_plots

CLASS_THRESHOLD = 200_000
CONTINUOUS_TO_CAP = ("LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea", "1stFlrSF", "2ndFlrSF")
# Для этих площадей ноль означает отсутствие элемента дома, а не ошибочный выброс.
ZERO_IS_ABSENCE_COLUMNS = ("TotalBsmtSF", "GarageArea", "2ndFlrSF")
# Здесь NA обозначает отсутствие инфраструктуры, но не неизвестное значение.
ABSENCE_COLUMNS = (
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
)


class HouseFeatures(BaseEstimator, TransformerMixin):
    """Новые признаки, рассчитываемые независимо для каждой записи."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "Id" in df.columns or "SalePrice" in df.columns:
            raise ValueError("В матрицу признаков попали Id или SalePrice")
        if "MSSubClass" in df:
            df["MSSubClass"] = df["MSSubClass"].astype("string").fillna("Неизвестно")
        # Пропуск в указанных категориях означает отсутствие соответствующего объекта.
        for col in ABSENCE_COLUMNS:
            if col in df:
                df[col] = df[col].fillna("Отсутствует")
        def num(name: str) -> pd.Series:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        if all(c in df for c in ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF")):
            df["TotalUsefulSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
        if all(c in df for c in ("YrSold", "YearBuilt")):
            df["PropertyAgeAtSale"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
        if all(c in df for c in ("YrSold", "YearRemodAdd")):
            df["YearsSinceRemodel"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)
        if all(c in df for c in ("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath")):
            df["TotalBathrooms"] = (num("FullBath") + .5 * num("HalfBath")
                                     + num("BsmtFullBath") + .5 * num("BsmtHalfBath"))
        if "GarageArea" in df:
            df["HasGarage"] = (num("GarageArea") > 0).astype("int8")
        if "TotalBsmtSF" in df:
            df["HasBasement"] = (num("TotalBsmtSF") > 0).astype("int8")
        return df


class IQRClipper(BaseEstimator, TransformerMixin):
    """Ограничивает только непрерывные признаки границами обучающего IQR.

    Верхние границы не пересчитываются на тестовой выборке. Нулевой IQR
    не считается основанием для ограничения дискретных признаков.
    """

    def __init__(self, numeric_columns: tuple[str, ...],
                 target_columns: tuple[str, ...] = CONTINUOUS_TO_CAP, factor: float = 1.5):
        self.numeric_columns = numeric_columns
        self.target_columns = target_columns
        self.factor = factor

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        self.lower_ = np.full(x.shape[1], -np.inf)
        self.upper_ = np.full(x.shape[1], np.inf)
        for name in self.target_columns:
            if name not in self.numeric_columns:
                continue
            i = self.numeric_columns.index(name)
            q1, q3 = np.quantile(x[:, i], [0.25, 0.75])
            if np.isfinite(q1) and np.isfinite(q3) and q3 > q1:
                # Нулевую площадь отсутствующих элементов сохраняем без изменений.
                # Для них ограничиваем только верхний хвост распределения.
                if name not in ZERO_IS_ABSENCE_COLUMNS:
                    self.lower_[i] = q1 - self.factor * (q3 - q1)
                self.upper_[i] = q3 + self.factor * (q3 - q1)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.numeric_columns if input_features is None else input_features, dtype=object)

    def transform(self, X):
        if not hasattr(self, "lower_"):
            raise ValueError("IQRClipper ещё не обучен")
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class RecalculateTotalUsefulArea(BaseEstimator, TransformerMixin):
    """Согласует суммарную площадь с компонентами после заполнения и IQR.

    Никакие границы или статистики при transform не вычисляются.
    """

    def __init__(self, numeric_columns: tuple[str, ...]):
        self.numeric_columns = numeric_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.array(X, dtype=float, copy=True)
        components = ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF")
        if "TotalUsefulSF" in self.numeric_columns and all(
            name in self.numeric_columns for name in components
        ):
            total_index = self.numeric_columns.index("TotalUsefulSF")
            component_indices = [self.numeric_columns.index(name) for name in components]
            values[:, total_index] = values[:, component_indices].sum(axis=1)
        return values

    def get_feature_names_out(self, input_features=None):
        names = self.numeric_columns if input_features is None else input_features
        return np.asarray(names, dtype=object)


def _outlier_change_summary(transformer: Pipeline, split_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Сколько значений действительно ограничено IQR на train и test.

    В обоих случаях используются только границы, определённые по train.
    Диагностика выполняется после числового заполнения, но до масштабирования.
    """
    encoder = transformer.named_steps["encode"]
    numeric_pipeline = encoder.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["impute"]
    clipper = numeric_pipeline.named_steps["clip"]
    columns = clipper.numeric_columns
    report_rows = []
    for split_name, raw_features in split_features.items():
        engineered = transformer.named_steps["features"].transform(raw_features)
        before = np.asarray(imputer.transform(engineered[list(columns)]), dtype=float)
        after = clipper.transform(before)
        for name in clipper.target_columns:
            if name not in columns:
                continue
            position = columns.index(name)
            low = float(clipper.lower_[position])
            high = float(clipper.upper_[position])
            low_count = int(np.count_nonzero(before[:, position] < low))
            high_count = int(np.count_nonzero(before[:, position] > high))
            changed = int(np.count_nonzero(before[:, position] != after[:, position]))
            if changed != low_count + high_count:
                raise ValueError("Несогласованные результаты расчёта ограничений IQR")
            report_rows.append({
                "Выборка": split_name,
                "Признак": name,
                "Нижняя граница": low if np.isfinite(low) else np.nan,
                "Верхняя граница": high if np.isfinite(high) else np.nan,
                "Ниже границы": low_count,
                "Выше границы": high_count,
                "Всего изменено": changed,
                "Доля изменённых, %": round(100 * changed / len(before), 2),
                "Минимум до": float(before[:, position].min()),
                "Максимум до": float(before[:, position].max()),
                "Минимум после": float(after[:, position].min()),
                "Максимум после": float(after[:, position].max()),
            })
    return pd.DataFrame(report_rows)


@dataclass
class PreprocessingResult:
    report_path: Path
    train_rows: int
    test_rows: int
    features_count: int
    generated_plots: dict[str, Path]
    selected_kbest: list[str]
    selected_rfe: list[str]
    pca_explained_variance: tuple[float, float]
    missing_before: int
    class_balance: dict[int, int]


def _make_preprocessor(train_features: pd.DataFrame) -> Pipeline:
    numeric_columns = tuple(train_features.select_dtypes(include=np.number).columns)
    categorical_columns = tuple(c for c in train_features if c not in numeric_columns)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", IQRClipper(numeric_columns)),
        # Восстанавливаем суммарную площадь из уже обработанных компонент.
        ("reconcile_area", RecalculateTotalUsefulArea(numeric_columns)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Неизвестно")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    encoder = ColumnTransformer([
        ("num", numeric, list(numeric_columns)),
        ("cat", categorical, list(categorical_columns)),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("features", HouseFeatures()), ("encode", encoder)])


# Анализируем одинаковую группу содержательных числовых характеристик,
# исключая сотни бинарных столбцов One-Hot Encoding.
VIF_FEATURES = (
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd", "LotArea",
    "TotalUsefulSF", "TotalBathrooms", "PropertyAgeAtSale", "YearsSinceRemodel",
)
# Эти производные характеристики делают набор точно или почти линейно
# зависимым. Исключение действует лишь для *диагностики*, не для моделей.
VIF_REDUCED_EXCLUDE = ("TotalUsefulSF", "PropertyAgeAtSale", "YearsSinceRemodel")


def _evaluate_vif(train_processed: pd.DataFrame, *,
                  exclude: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Рассчитать VIF по центрированным признакам через вспомогательную OLS.

    При точной линейной зависимости стандартный variance_inflation_factor
    вызывает предупреждения вырожденной матрицы. Здесь используем ту же
    формулу VIF = 1 / (1 - R²), но R² рассчитываем через np.linalg.lstsq.
    Бесконечный VIF не маскируется: это диагностический результат.
    """
    candidates = [
        "num__" + name for name in VIF_FEATURES
        if name not in exclude and "num__" + name in train_processed
        and float(train_processed["num__" + name].std()) > 1e-8
    ]
    if not candidates:
        return pd.DataFrame(columns=["Признак", "VIF", "Интерпретация"]), {
            "Признаков": 0, "Ранг": 0, "Число обусловленности": np.nan,
            "Линейная зависимость": False,
        }
    data = train_processed[candidates].to_numpy(dtype=float)
    # Стандартизация в расчёте VIF (не меняет сам показатель) стабилизирует
    # вычисление SVD и минимизирует влияние единиц измерения.
    centered = data - data.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centered ** 2, axis=0))
    z = centered / scale
    rank = int(np.linalg.matrix_rank(z))
    cond = float(np.linalg.cond(z))
    values: list[tuple[str, float, str]] = []
    for i, name in enumerate(candidates):
        other = np.delete(z, i, axis=1)
        if other.shape[1] == 0:
            vif = 1.0
        else:
            coeff, *_ = np.linalg.lstsq(other, z[:, i], rcond=1e-12)
            residual = z[:, i] - other @ coeff
            fraction = float(np.dot(residual, residual) / np.dot(z[:, i], z[:, i]))
            # Практически нулевой относительный остаток указывает на точную зависимость в
            # пределах численной точности вычислений.
            vif = np.inf if fraction <= 1e-12 else 1.0 / fraction
        if np.isinf(vif):
            interpretation = "Точная линейная зависимость"
        elif vif > 10:
            interpretation = "Сильная мультиколлинеарность"
        elif vif > 5:
            interpretation = "Требует внимания"
        else:
            interpretation = "Умеренная или низкая"
        values.append((name.removeprefix("num__"), vif, interpretation))
    result = pd.DataFrame(values, columns=["Признак", "VIF", "Интерпретация"])
    result.sort_values("VIF", ascending=False, inplace=True, kind="stable")
    return result, {
        "Признаков": len(candidates), "Ранг": rank,
        "Число обусловленности": cond,
        "Линейная зависимость": bool(rank < len(candidates)),
    }


def run_preprocessing(data_path: str | Path = RAW_TRAIN_FILE,
                      output_dir: str | Path = OUTPUT_DIR / "preprocessing",
                      processed_dir: str | Path = PROCESSED_DATA_DIR,
                      make_plots: bool = True,
                      run_diagnostics: bool = True) -> PreprocessingResult:
    """Создать согласованные train/test выборки и сохранить все артефакты."""
    raw = load_training_data(data_path)
    if raw["SalePrice"].isna().any() or raw["Id"].duplicated().any():
        raise ValueError("В исходном наборе обнаружены пустые цены или повторяющиеся Id")
    raw = raw.reset_index(drop=True)
    prices = pd.to_numeric(raw["SalePrice"], errors="raise")
    class_labels = (prices >= CLASS_THRESHOLD).astype(int)
    if class_labels.nunique() != 2 or class_labels.value_counts().min() < 2:
        raise ValueError("Для стратифицированного разбиения нужны хотя бы два объекта каждого класса")
    train_idx, test_idx = train_test_split(
        np.arange(len(raw)), test_size=.2, random_state=RANDOM_STATE, stratify=class_labels
    )
    X = raw.drop(columns=["Id", "SalePrice"])
    X_train_raw, X_test_raw = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train_reg, y_test_reg = prices.iloc[train_idx], prices.iloc[test_idx]
    y_train_clf, y_test_clf = class_labels.iloc[train_idx], class_labels.iloc[test_idx]

    transformer = _make_preprocessor(HouseFeatures().transform(X_train_raw))
    X_train_np = transformer.fit_transform(X_train_raw, y_train_reg)
    X_test_np = transformer.transform(X_test_raw)
    cols = transformer.named_steps["encode"].get_feature_names_out()
    X_train = pd.DataFrame(X_train_np, columns=cols, index=train_idx)
    X_test = pd.DataFrame(X_test_np, columns=cols, index=test_idx)
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("После предобработки остались бесконечные или пропущенные значения")
    if any("SalePrice" in c or c.endswith("Id") for c in X_train.columns):
        raise ValueError("Утечка целевой переменной или идентификатора в признаках")
    output_dir, processed_dir = Path(output_dir), Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Диагностика отражает изменения отдельно в каждой выборке без
    # повторного обучения IQR на тестовых объектах.
    outlier_changes = _outlier_change_summary(transformer, {
        "train": X_train_raw, "test": X_test_raw,
    })
    outlier_changes.to_csv(output_dir / "outlier_changes.csv", index=False)

    for suffix in ("reg", "clf"):
        X_train.to_csv(processed_dir / f"X_train_{suffix}.csv", index=False)
        X_test.to_csv(processed_dir / f"X_test_{suffix}.csv", index=False)
    for suffix, y_train, y_test in (("reg", y_train_reg, y_test_reg),
                                    ("clf", y_train_clf, y_test_clf)):
        y_train.to_frame(name="SalePrice" if suffix == "reg" else "PriceClass").to_csv(
            processed_dir / f"y_train_{suffix}.csv", index=False)
        y_test.to_frame(name="SalePrice" if suffix == "reg" else "PriceClass").to_csv(
            processed_dir / f"y_test_{suffix}.csv", index=False)
    pd.DataFrame({"Id":raw.iloc[train_idx]["Id"].to_numpy()}).to_csv(processed_dir / "train_ids.csv", index=False)
    pd.DataFrame({"Id":raw.iloc[test_idx]["Id"].to_numpy()}).to_csv(processed_dir / "test_ids.csv", index=False)
    joblib.dump(transformer, output_dir / "preprocessing_pipeline.joblib")
    pd.DataFrame({"Признак": cols}).to_csv(output_dir / "feature_names.csv", index=False)

    # Признаки отбираются только по тренировочным данным.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kb = SelectKBest(score_func=f_regression, k=min(20, X_train.shape[1]))
        kb.fit(X_train, y_train_reg)
    kbest = list(X_train.columns[kb.get_support()])
    feature_scores = pd.Series(kb.scores_, index=X_train.columns).replace(
        [np.inf, -np.inf], np.nan
    )
    rfe_list: list[str] = []
    vif_df = pd.DataFrame(columns=["Признак", "VIF", "Интерпретация"])
    vif_reduced = vif_df.copy()
    empty_vif_info = {"Признаков": 0, "Ранг": 0, "Число обусловленности": np.nan,
                      "Линейная зависимость": False}
    vif_info = empty_vif_info.copy()
    vif_reduced_info = empty_vif_info.copy()
    if run_diagnostics:
        # RFE применяется к 50 наиболее значимым по F-критерию признакам. Это снижает вычислительную
        # стоимость и меньше риска нестабильности из-за редких dummy-полей.
        candidates = feature_scores.fillna(-np.inf).nlargest(
            min(50, len(feature_scores))
        ).index.tolist()
        rfe = RFE(LinearRegression(), n_features_to_select=min(15, len(candidates)), step=.3)
        rfe.fit(X_train[candidates], y_train_reg)
        rfe_list = list(np.asarray(candidates)[rfe.get_support()])
        vif_df, vif_info = _evaluate_vif(X_train)
        vif_reduced, vif_reduced_info = _evaluate_vif(
            X_train, exclude=VIF_REDUCED_EXCLUDE,
        )
    pd.DataFrame({"Признак": kbest}).to_csv(output_dir / "select_k_best.csv", index=False)
    pd.DataFrame({"Признак": rfe_list}).to_csv(output_dir / "rfe_selected.csv", index=False)
    vif_df.to_csv(output_dir / "vif.csv", index=False)
    vif_reduced.to_csv(output_dir / "vif_reduced.csv", index=False)
    pd.DataFrame([
        {"Набор": "Полный", **vif_info},
        {"Набор": "Сокращённый", **vif_reduced_info},
    ]).to_csv(output_dir / "vif_diagnostics.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_train = pca.fit_transform(X_train)
    # Используем обученную PCA для тестовой выборки, а не обучаем заново.
    pca_test = pca.transform(X_test)
    joblib.dump(pca, output_dir / "pca_visualization.joblib")
    pd.DataFrame(pca_train, columns=["PC1", "PC2"]).to_csv(output_dir / "train_pca.csv", index=False)
    pd.DataFrame(pca_test, columns=["PC1", "PC2"]).to_csv(output_dir / "test_pca.csv", index=False)

    # KNN как альтернативный метод восстановления числовых признаков.
    # Стратегию следует сравнивать по качеству моделей, а не по самой
    # возможности заполнить пустые ячейки. Основной конвейер использует медиану.
    engineered_train = HouseFeatures().transform(X_train_raw)
    numerical_raw = engineered_train.select_dtypes(include=np.number)
    knn_columns = [c for c in ("LotFrontage", "MasVnrArea", "GarageYrBlt") if c in numerical_raw]
    if knn_columns:
        knn_filled = KNNImputer(n_neighbors=5).fit_transform(numerical_raw[knn_columns])
        pd.DataFrame({"Признак": knn_columns,
                      "Пропуски до": [int(numerical_raw[c].isna().sum()) for c in knn_columns],
                      "Пропуски после KNN": [int(pd.isna(knn_filled[:, i]).sum()) for i in range(len(knn_columns))]
                      }).to_csv(output_dir / "knn_imputation_review.csv", index=False)

    plots: dict[str, Path] = {}
    if make_plots:
        # Здесь подготавливаются данные для графиков. Рисунки строятся в пакете visualization.
        missing_counts = X_train_raw.isna().sum().sort_values(ascending=False).head(14)
        top_scores = feature_scores.fillna(0).sort_values(ascending=False).head(14)
        plots = save_preprocessing_plots(
            missing_counts=missing_counts,
            feature_scores=top_scores,
            pca_points=pca_train,
            price_classes=y_train_clf.to_numpy(),
            pca_explained_variance=pca.explained_variance_ratio_,
            dest=output_dir,
        )

    counts = class_labels.value_counts().sort_index()
    train_change_rows = outlier_changes[outlier_changes["Выборка"] == "train"]
    test_change_rows = outlier_changes[outlier_changes["Выборка"] == "test"]
    train_changed = int(train_change_rows["Всего изменено"].sum())
    test_changed = int(test_change_rows["Всего изменено"].sum())
    highest_changes = train_change_rows.sort_values("Всего изменено", ascending=False).head(3)
    changed_description = ", ".join(
        f"{r['Признак']} ({int(r['Всего изменено'])})" for _, r in highest_changes.iterrows()
    ) if not highest_changes.empty else "не обнаружены"
    vif_high = int((vif_df["VIF"] > 10).sum()) if not vif_df.empty else 0
    vif_infinite = int(np.isinf(vif_df["VIF"].to_numpy(dtype=float)).sum()) if not vif_df.empty else 0
    vif_reduced_high = int((vif_reduced["VIF"] > 10).sum()) if not vif_reduced.empty else 0
    vif_reduced_infinite = int(np.isinf(vif_reduced["VIF"].to_numpy(dtype=float)).sum()) if not vif_reduced.empty else 0
    report_lines = [
        "# 2. Предобработка данных и инженерия признаков", "",
        "## Исходные данные и разделение", "",
        f"- Всего объектов: **{len(raw)}**.",
        f"- Обучающая выборка **{len(train_idx)}** объектов. Тестовая выборка **{len(test_idx)}** объектов.",
        f"- Исходных пропущенных ячеек: **{int(X.isna().sum().sum())}** (без Id и SalePrice).",
        f"- Итоговых признаков после кодирования: **{X_train.shape[1]}**.",
        f"- Ценовые классы. Ниже 200 000 USD - **{int(counts[0])}** объектов, от 200 000 USD - **{int(counts[1])}**.",
        "- Разделение фиксировано `random_state=42` и стратифицировано по ценовой категории.", "",
        "## Принятая стратегия", "",
        "Категориальные пропуски, обозначающие отсутствие объекта (например, гаража или бассейна),",
        "выделяются в отдельную категорию. Остальные пропуски заполняются по статистикам обучающей выборки.",
        "медианой для числовых признаков и константой для категориальных. Для некоторых непрерывных",
        "признаков значения за пределами 1,5 IQR ограничиваются границами, рассчитанными по train.",
        "Категориальные признаки кодируются One-Hot Encoding, числовые стандартизируются.",
        "", "## Обработка выбросов", "",
        f"После заполнения пропусков методом IQR скорректировано **{train_changed} значений в обучающей**",
        f"и **{test_changed} значений в тестовой выборке**. Это количество ячеек, а не объектов.",
        "Чаще всего изменения затрагивали " + changed_description + ".",
        "Подробные границы, минимумы, максимумы и доли приведены в `outlier_changes.csv`.",
        "Границы определены только по обучающей части и применены к тестовой без переобучения.",
        "Отсутствие подвала, гаража или второго этажа по-прежнему обозначается нулевой площадью.",
        "для таких признаков IQR ограничивает только чрезмерно большие значения.",
        "Ограничение экстремальных площадей является исследовательским решением. Его влияние на прогнозы",
        "нужно будет сравнить с вариантом без ограничений.", "",
        "В качестве альтернативы рассмотрено KNN-заполнение числовых признаков, но его пользу",
        "для качества прогнозов предстоит проверить при обучении моделей.", "",
        "## Новые характеристики недвижимости", "",
        "В зависимости от доступных столбцов формируются суммарная площадь, возраст дома при продаже,",
        "число лет после реконструкции, число санузлов с учётом половинных и наличие гаража/подвала.",
        "`TotalUsefulSF` (суммарная площадь) пересчитывается после заполнения пропусков",
        "и ограничения исходных площадей. Поэтому она равна сумме обработанных",
        "`TotalBsmtSF` (площадь подвала), `1stFlrSF` (площадь первого этажа)",
        "и `2ndFlrSF` (площадь второго этажа) до масштабирования признаков.", "",
        "## Отбор признаков и снижение размерности", "",
        "**SelectKBest** (до 20 признаков) - " + ", ".join(kbest[:10]) + ".", "",
        "**RFE**: " + (", ".join(rfe_list[:10]) if rfe_list else "не выполнялся в ускоренном режиме") + ".", "",
        f"Две компоненты PCA объясняют **{sum(pca.explained_variance_ratio_):.1%}** дисперсии признаков.",
        "PCA обучен только на тренировочной выборке. Тестовые объекты лишь преобразуются.", "",
        "**VIF. Полный набор** включает 14 основных и производных числовых характеристик.",
        f"Матрица имеет ранг {vif_info['Ранг']} при {vif_info['Признаков']} признаках.",
        f"VIF выше 10 у {vif_high} признаков, в том числе бесконечный у {vif_infinite}.",
        "Точная зависимость, в частности, обусловлена тем, что суммарная площадь",
        "равна сумме площадей подвала, первого и второго этажей.",
        "**VIF. Сокращённый набор** построен после исключения признаков только для диагностики. Исключены",
        "`TotalUsefulSF` (суммарная площадь), `PropertyAgeAtSale` (возраст при продаже)",
        "и `YearsSinceRemodel` (лет после реконструкции).",
        f"Матрица имеет ранг {vif_reduced_info['Ранг']} при {vif_reduced_info['Признаков']} признаках.",
        f"VIF выше 10 у {vif_reduced_high} признаков, бесконечный у {vif_reduced_infinite}.",
        "Точные значения приведены в `vif.csv` и `vif_reduced.csv`,",
        "Сведения о ранге находятся в `vif_diagnostics.csv`.",
        "Это сравнение показывает эффект взаимозависимых признаков, но не доказывает",
        "улучшение качества модели. В итоговых матрицах никакие признаки не удалены.", "",
        "## Выходные данные", "",
        "Числовые матрицы и целевые переменные сохранены в `data/processed/`,",
        "Обученный pipeline находится в `outputs/preprocessing/preprocessing_pipeline.joblib`.",
        "Регрессия и классификация используют одинаковые строки и входные признаки.",
        "Все обучаемые преобразования были настроены только по тренировочным строкам.", "",
    ]
    report = output_dir / "preprocessing_summary.md"
    report.write_text("\n".join(report_lines), encoding="utf-8")
    return PreprocessingResult(
        report_path=report, train_rows=len(train_idx), test_rows=len(test_idx),
        features_count=X_train.shape[1], generated_plots=plots,
        selected_kbest=kbest, selected_rfe=rfe_list,
        pca_explained_variance=tuple(map(float, pca.explained_variance_ratio_)),
        missing_before=int(X.isna().sum().sum()),
        class_balance={int(k): int(v) for k, v in counts.items()}
    )
