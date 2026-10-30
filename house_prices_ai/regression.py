"""ЛР №3: честная CV с предобработкой внутри каждого фолда.

Использует исходные строки, но только ID строк, отведённых ЛР №2 в train.
Внешняя тестовая выборка не участвует в отборе моделей и настройке параметров.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from house_prices_ai.config import OUTPUT_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_TRAIN_FILE
from house_prices_ai.evaluation import (
    create_regression_figures, regression_metrics, save_mlp_training_diagnostics,
)
from house_prices_ai.preprocessing import HouseFeatures, _make_preprocessor

SCORING = {"R2": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}


@dataclass
class RegressionResult:
    summary_path: Path
    test_path: Path
    folds_path: Path
    report_path: Path
    figures: dict[str, Path]
    winner: str


def load_lr02_split(raw_path: str | Path = RAW_TRAIN_FILE,
                    processed_dir: str | Path = PROCESSED_DATA_DIR):
    """Воссоздаёт именно разбиение ЛР №2, используя сохранённые Id.

    Возвращает исходные признаки, а не X_train_reg.csv; последнее важно для CV.
    """
    raw_path, processed_dir = Path(raw_path), Path(processed_dir)
    raw = pd.read_csv(raw_path)
    if "Id" not in raw or "SalePrice" not in raw:
        raise ValueError("В исходном train.csv нужны колонки Id и SalePrice")
    if raw["Id"].duplicated().any():
        raise ValueError("В исходном наборе повторяются Id")
    train_ids = pd.read_csv(processed_dir / "train_ids.csv")["Id"].to_numpy()
    test_ids = pd.read_csv(processed_dir / "test_ids.csv")["Id"].to_numpy()
    if len(set(train_ids) & set(test_ids)):
        raise ValueError("Пересекаются обучающие и тестовые Id")
    by_id = raw.set_index("Id", verify_integrity=True)
    if len(train_ids)+len(test_ids) != len(raw) or len(set(np.r_[train_ids,test_ids])) != len(raw):
        raise ValueError("Id из ЛР №2 не совпадают с числом исходных строк")
    train = by_id.loc[train_ids].copy()
    test = by_id.loc[test_ids].copy()
    if train["SalePrice"].isna().any() or test["SalePrice"].isna().any():
        raise ValueError("Отсутствует целевая переменная SalePrice")
    X_train = train.drop(columns="SalePrice")
    X_test = test.drop(columns="SalePrice")
    y_train = train["SalePrice"].astype(float)
    y_test = test["SalePrice"].astype(float)
    return X_train, X_test, y_train, y_test


def make_regression_pipeline(X_train_raw: pd.DataFrame, estimator) -> Pipeline:
    """Каждый fit() независимо подбирает imputer, IQR, scaler и OneHot.

    Набор исходных колонок определяется по метаданным, не по значениям и target.
    """
    schema = HouseFeatures().transform(X_train_raw)
    preprocessor = _make_preprocessor(schema)
    return Pipeline([("preprocessing", preprocessor), ("model", estimator)])


def model_catalog() -> dict[str, object]:
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=100.0, max_iter=50000, tol=1e-4),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, objective="reg:squarederror",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "MLP": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                alpha=.01, learning_rate_init=.001, max_iter=700,
                early_stopping=True, validation_fraction=.15,
                n_iter_no_change=25, random_state=RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        ),
    }


def cv_compare(X_train: pd.DataFrame, y_train: pd.Series, cv: KFold,
               models: dict[str, object] | None = None):
    """Одинаковые фолды; независимые fit() трансформера и модели в каждом фолде."""
    models = model_catalog() if models is None else models
    summaries, folds = [], []
    for name, estimator in models.items():
        print(f"CV: {name}", flush=True)
        pipe = make_regression_pipeline(X_train, clone(estimator))
        values = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING,
                                return_train_score=True, n_jobs=1, error_score="raise")
        test_r2 = values["test_R2"]
        train_r2 = values["train_R2"]
        mae = -values["test_MAE"]
        rmse = -values["test_RMSE"]
        summaries.append({
            "Model": name, "R2_mean": test_r2.mean(), "R2_std": test_r2.std(),
            "Train_R2_mean": train_r2.mean(), "MAE_mean": mae.mean(),
            "MAE_std": mae.std(), "RMSE_mean": rmse.mean(), "RMSE_std": rmse.std(),
        })
        for k, (a, b, c, d) in enumerate(zip(test_r2,train_r2,mae,rmse), start=1):
            folds.append({"Model":name,"Fold":k,"R2":a,"Train_R2":b,"MAE":c,"RMSE":d})
    return (pd.DataFrame(summaries).sort_values("MAE_mean").reset_index(drop=True),
            pd.DataFrame(folds))


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, out_dir: Path,
                 iterations: int = 8) -> dict:
    """Небольшой RandomizedSearchCV исключительно по train, без внешнего теста.

    Оптимизация на трёх фолдах — подбор кандидата, НЕ несмещённая финальная CV.
    """
    pipe = make_regression_pipeline(X_train, model_catalog()["XGBoost"])
    params = {
        "model__n_estimators": [200, 300, 450],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [.025, .05, .08],
        "model__subsample": [.75, .9, 1.0],
        "model__colsample_bytree": [.75, .9, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1.0, 3.0, 6.0],
    }
    search = RandomizedSearchCV(
        pipe, param_distributions=params, n_iter=iterations,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE+1),
        n_jobs=1, refit=True, random_state=RANDOM_STATE,
        error_score="raise", verbose=1,
    )
    search.fit(X_train, y_train)
    pd.DataFrame(search.cv_results_).to_csv(out_dir / "xgboost_tuning_trials.csv", index=False)
    best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
            for k, v in search.best_params_.items()}
    search_space_size = int(np.prod([len(values) for values in params.values()]))
    payload = {
        "best_params": best,
        "cv3_mae_usd": float(-search.best_score_),
        "scoring": "negative MAE (минимизация MAE в USD)",
        "cv_folds": 3,
        "iterations": int(iterations),
        "search_space_combinations": search_space_size,
        "random_state": RANDOM_STATE,
        "candidate_values": params,
        "note": "Подбор по train, оценка CV3 оптимистична; не независимая проверка настроенной модели.",
    }
    (out_dir / "xgboost_tuning.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(search.best_estimator_, out_dir / "xgboost_tuned_pipeline.joblib")
    return payload


def _report(cv_summary: pd.DataFrame, test_metrics: pd.DataFrame,
            winner: str, tuning: dict | None) -> str:
    best = cv_summary.iloc[0]
    mlp = cv_summary.loc[cv_summary.Model == "MLP"].iloc[0]
    linear = cv_summary.loc[cv_summary.Model == "LinearRegression"].iloc[0]
    lines = [
        "# Лабораторная работа №3 — регрессия: черновые выводы", "",
        "## Постановка задачи", "",
        "Прогноз SalePrice (USD) по историческим характеристикам недвижимости в Эймсе, штат Айова.",
        "80% исходного датасета — обучающая выборка, 20% — отдельная тестовая выборка из ЛР №2.",
        "Выполнена пятикратная CV только на обучающих объектах; обработка пропусков,",
        "ограничение выбросов, One-Hot и StandardScaler обучаются **внутри каждого фолда**.", "",
        "## Выбор модели по CV", "",
        f"Лидер по средней MAE: **{winner}**; MAE = **${best.MAE_mean:,.2f}**; "
        f"RMSE = **${best.RMSE_mean:,.2f}**; R² = **{best.R2_mean:.4f}** "
        f"(стандартное отклонение R² = {best.R2_std:.4f}).", "",
        f"Обычная линейная регрессия: CV R² = **{linear.R2_mean:.4f}**, "
        f"Train R² = **{linear.Train_R2_mean:.4f}**; высокая разница — признак слабого обобщения.",
        f"MLP: CV R² = **{mlp.R2_mean:.4f}**, Train R² = **{mlp.Train_R2_mean:.4f}**. ", "",
        "## Графики", "",
        "- Рисунок 3.1 — сравнение R², MAE и RMSE на отдельных шкалах.",
        "- Рисунок 3.2 — разброс R² между фолдами.",
        "- Рисунок 3.3 — OOF-прогнозы против фактических цен у лидирующей модели.",
        "- Рисунок 3.4 — остатки OOF-прогнозов.",
        "- Рисунок 3.5 — Q-Q график остатков.",
        "- Рисунок 3.6 — процесс обучения MLP на всей тренировочной части.", "",
        "## Методические оговорки", "",
        "- Средние CV-метрики получены по одинаковым фолдам. Стандартное отклонение — не доверительный интервал.",
        "- Фиксированный внешний test ранее использовался для диагностики отдельных объектов;",
        "  поэтому его результаты описательные, а не полностью независимое подтверждение выбора.",
        "- SHAP и подробная интерпретация отнесены к ЛР №5.",
        "- Признаки бассейна редки; изучение конкретного объекта не доказывает причинного эффекта.",
        "- Предобработка применяется отдельно на каждом CV-фолде; внешний test не участвует в подборе.",
    ]
    if tuning:
        lines.extend(["", "## Подбор XGBoost", "",
                      f"RandomizedSearchCV (3-fold, MAE): {tuning['cv3_mae_usd']:,.2f} USD;",
                      "это оценка при подборе параметров, не независимая оценка качества.",
                      f"Подобранные параметры: `{tuning['best_params']}`."])
    return "\n".join(lines)+"\n"


def run_regression(raw_path: str | Path = RAW_TRAIN_FILE,
                   processed_dir: str | Path = PROCESSED_DATA_DIR,
                   out_dir: str | Path = OUTPUT_DIR / "regression",
                   folds: int = 5, tune: bool = False, tuning_iterations: int = 8,
                   make_plots: bool = True) -> RegressionResult:
    """Полный черновик ЛР №3: CV, диагностический test, OOF графики и артефакты."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_lr02_split(raw_path, processed_dir)
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    catalog = model_catalog()
    print(f"Обучающая выборка: {X_train.shape}; внешняя тестовая: {X_test.shape}", flush=True)
    print("Валидация: обучаемая предобработка выполняется внутри каждого фолда.", flush=True)

    summary, individual = cv_compare(X_train, y_train, cv, catalog)
    summary_path = out_dir / "cv_clean_summary.csv"
    folds_path = out_dir / "cv_clean_folds.csv"
    summary.to_csv(summary_path, index=False)
    individual.to_csv(folds_path, index=False)
    winner = str(summary.iloc[0]["Model"])
    print("\nCV (сортировка по MAE):\n", summary.round(4).to_string(index=False), flush=True)

    # Holdout — для сопоставления с уже изученным тестом, НЕ подбора параметров.
    trained = {}
    test_rows = []
    test_predictions = {}
    for name, estimator in catalog.items():
        pipe = make_regression_pipeline(X_train, clone(estimator))
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained[name] = pipe
        test_predictions[name] = pred
        test_rows.append({"Model": name, **regression_metrics(y_test, pred)})
    test_metrics = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
    test_path = out_dir / "test_descriptive.csv"
    test_metrics.to_csv(test_path, index=False)
    joblib.dump(trained[winner], out_dir / "best_regression_pipeline.joblib")
    joblib.dump(trained["XGBoost"], out_dir / "xgboost_baseline_pipeline.joblib")
    print("\nВнешний тест (только описание):\n", test_metrics.round(3).to_string(index=False), flush=True)

    # Локальный ранее исследованный объект: находим по сырой категории и цене,
    # а не по нестабильному порядковому индексу 259.
    special_mask = (X_test.get("PoolQC", pd.Series(index=X_test.index,dtype=object)).eq("Gd")
                    & y_test.eq(274970))
    if special_mask.any():
        row_pos = int(np.flatnonzero(special_mask.to_numpy())[0])
        case_rows = [{"Model": name, "Id": X_test.index[row_pos],
                      "Actual_USD": float(y_test.iloc[row_pos]),
                      "Pred_USD": float(pred[row_pos]),
                      "Abs_error_USD": float(abs(y_test.iloc[row_pos]-pred[row_pos]))}
                     for name, pred in test_predictions.items()]
        pd.DataFrame(case_rows).to_csv(out_dir / "rare_pool_case.csv", index=False)

    mlp_training = save_mlp_training_diagnostics(trained["MLP"], out_dir)
    print(
        f"MLP: лучшая эпоха {mlp_training['best_epoch']} из {mlp_training['epochs_run']}; "
        f"R² validation {mlp_training['best_validation_r2']:.4f} -> "
        f"{mlp_training['last_validation_r2']:.4f} к концу обучения.", flush=True,
    )

    print(f"\nOOF-диагностика: {winner}", flush=True)
    oof = cross_val_predict(make_regression_pipeline(X_train, clone(catalog[winner])),
                            X_train, y_train, cv=cv, n_jobs=1)
    pd.DataFrame({"Id": X_train.index.to_numpy(), "SalePrice": y_train.to_numpy(),
                  "OOF_Pred": oof, "Residual": y_train.to_numpy()-oof}).to_csv(
        out_dir / "oof_predictions.csv", index=False)
    figures = (create_regression_figures(summary, individual, y_train, oof,
                                       out_dir / "figures", trained["MLP"])
               if make_plots else {})

    tuning_info = tune_xgboost(X_train, y_train, out_dir, tuning_iterations) if tune else None
    report_path = out_dir / "lr03_summary.md"
    report_path.write_text(_report(summary, test_metrics, winner, tuning_info), encoding="utf-8")
    return RegressionResult(summary_path, test_path, folds_path, report_path, figures, winner)
