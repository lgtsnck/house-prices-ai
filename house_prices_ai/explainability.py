"""Интерпретация XGBoost и Random Forest на исторических данных недвижимости.

Рассматриваются обученные модели из этапов регрессии и классификации.
Для SHAP используется внутренняя реализация Tree SHAP в XGBoost.
Предсказания исследуются описательно, без повторного выбора модели по test.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from xgboost import DMatrix, XGBClassifier, XGBRegressor

from house_prices_ai.classification import PRICE_THRESHOLD, build_classification_pipeline, starter_models
from house_prices_ai.config import OUTPUT_DIR, RANDOM_STATE
from house_prices_ai.regression import load_lr02_split, make_regression_pipeline, model_catalog
from house_prices_ai.visualization import (
    plot_importance_pair, plot_shap_beeswarm, plot_shap_bar,
    plot_shap_waterfall, save_shap_force_html, plot_rf_xgb_shap_comparison,
)


@dataclass
class ExplainabilityResult:
    report_path: Path
    figures: dict[str, Path]
    tables: dict[str, Path]
    local_ids: dict[str, int]


def _features_and_values(pipeline, raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if not {"preprocessing", "model"}.issubset(pipeline.named_steps):
        raise ValueError("Ожидается обученный Pipeline с шагами preprocessing и model")
    encoder = pipeline.named_steps["preprocessing"]
    numeric = np.asarray(encoder.transform(raw), dtype=np.float32)
    names = list(encoder.named_steps["encode"].get_feature_names_out())
    if numeric.ndim != 2 or numeric.shape[1] != len(names):
        raise ValueError("Названия признаков не совпали с трансформированной матрицей")
    if not np.isfinite(numeric).all():
        raise ValueError("После предобработки обнаружены NaN/Inf")
    return numeric, names


def _xgb_contributions(pipeline, raw: pd.DataFrame, task: str):
    """Возвращает значения Tree SHAP на шкале raw margin модели XGBoost.

    Регрессия — в долларах, бинарная классификация — в логарифме шансов.
    Проверяет аддитивность и соответствие сохранённому Pipeline.
    """
    features, names = _features_and_values(pipeline, raw)
    model = pipeline.named_steps["model"]
    if task == "regression" and not isinstance(model, XGBRegressor):
        raise TypeError("Для регрессии ожидается XGBRegressor")
    if task == "classification" and not isinstance(model, XGBClassifier):
        raise TypeError("Для классификации ожидается XGBClassifier")
    booster = model.get_booster()
    dm = DMatrix(features)
    contribution_matrix = np.asarray(booster.predict(dm, pred_contribs=True))
    if contribution_matrix.ndim == 3 and contribution_matrix.shape[1] == 1:
        contribution_matrix = contribution_matrix[:, 0, :]
    if contribution_matrix.shape != (len(raw), len(names) + 1):
        raise ValueError(f"Некорректная форма SHAP: {contribution_matrix.shape}")
    values = contribution_matrix[:, :-1]
    base = contribution_matrix[:, -1]
    margin = base + values.sum(axis=1)
    if task == "regression":
        predictions = pipeline.predict(raw)
        reconstructed = margin
    else:
        predictions = pipeline.predict_proba(raw)[:, 1]
        reconstructed = expit(margin)
    if not np.allclose(reconstructed, predictions, rtol=1e-4, atol=2e-3 if task == "regression" else 2e-5):
        raise ValueError("SHAP-вклады не воспроизводят прогноз сохранённой модели")
    return features, names, values, base, np.asarray(predictions), margin


def _feature_importance(pipeline, raw: pd.DataFrame) -> pd.DataFrame:
    _, names = _features_and_values(pipeline, raw.iloc[:1])
    importance = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    if importance.shape != (len(names),):
        raise ValueError("Длина Feature Importance не совпала с количеством признаков")
    return pd.DataFrame({"Feature": names, "Importance": importance}).sort_values(
        "Importance", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _make_explanation(values, base, data, names):
    import shap
    return shap.Explanation(
        values=np.asarray(values), base_values=np.asarray(base),
        data=np.asarray(data), feature_names=list(names),
    )


def _select_regression_case(X_test: pd.DataFrame, y_test: pd.Series) -> int:
    # Ранее исследованный дом: реальная цена $274 970, качество бассейна Gd.
    if 1424 in X_test.index:
        return int(np.flatnonzero(X_test.index.to_numpy() == 1424)[0])
    pool = X_test.get("PoolQC", pd.Series(index=X_test.index, dtype=object))
    mask = pool.eq("Gd") & y_test.eq(274970)
    return int(np.flatnonzero(mask.to_numpy())[0]) if mask.any() else 0


def run_explainability(output_dir: str | Path = OUTPUT_DIR / "explainability",
                       sample_size: int = 180, compare_rf_shap: bool = False) -> ExplainabilityResult:
    """Использует сохранённые XGBoost, RF переобучает на тех же train-строках.

    Внешний тест используется только для описательной интерпретации прогнозов.
    Настройка и выбор модели здесь НЕ производятся.
    """
    if sample_size < 10:
        raise ValueError("sample_size должен быть не меньше 10")
    out = Path(output_dir)
    figure_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reg_model_path = OUTPUT_DIR / "regression" / "xgboost_baseline_pipeline.joblib"
    clf_model_path = OUTPUT_DIR / "classification" / "best_classification_pipeline.joblib"
    missing = [str(p) for p in (reg_model_path, clf_model_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены обученные Pipeline: " + ", ".join(missing) +
            ". Сначала выполните регрессию и классификацию."
        )
    xgb_reg = joblib.load(reg_model_path)
    xgb_clf = joblib.load(clf_model_path)
    if not isinstance(xgb_reg.named_steps["model"], XGBRegressor):
        raise ValueError("Файл регрессии не содержит XGBRegressor")
    if not isinstance(xgb_clf.named_steps["model"], XGBClassifier):
        raise ValueError("Победитель классификации в сохранённом файле не XGBClassifier")

    X_train, X_test, y_train, y_test = load_lr02_split()
    # Исходный train используется для обучения RF и сохранённых Pipeline, не весь train.csv.
    y_train_clf = (y_train >= PRICE_THRESHOLD).astype(int)
    rf_reg = make_regression_pipeline(X_train, clone(model_catalog()["Random Forest"]))
    rf_clf = build_classification_pipeline(X_train, starter_models()["RandomForestClassifier"])
    print("Обучение Random Forest для сравнения важности признаков…", flush=True)
    rf_reg.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train_clf)

    tables: dict[str, Path] = {}
    figs: dict[str, Path] = {}
    for task, rf, xgb in (("regression", rf_reg, xgb_reg),
                          ("classification", rf_clf, xgb_clf)):
        rf_table = _feature_importance(rf, X_train)
        xgb_table = _feature_importance(xgb, X_train)
        for algorithm, df in (("random_forest", rf_table), ("xgboost", xgb_table)):
            p = out / f"importance_{task}_{algorithm}.csv"
            df.to_csv(p, index=False)
            tables[f"importance_{task}_{algorithm}"] = p
        p = figure_dir / ("5_1_importance_regression.png" if task == "regression"
                           else "5_2_importance_classification.png")
        plot_importance_pair(rf_table, xgb_table, p)
        figs[f"importance_{task}"] = p

    rng = np.random.default_rng(RANDOM_STATE)
    positions = np.sort(rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False))
    reg_case = _select_regression_case(X_test, y_test)
    if reg_case not in positions:
        positions[0] = reg_case
        positions = np.sort(positions)
    # Классификация: наиболее неоднозначный прогноз по всему внешнему test.
    uncertain = np.abs(xgb_clf.predict_proba(X_test)[:, 1] - .5)
    clf_case = int(np.argmin(uncertain))
    if clf_case not in positions:
        positions[-1] = clf_case
        positions = np.sort(positions)

    import shap
    for task, model, case_idx in (
        ("regression", xgb_reg, reg_case), ("classification", xgb_clf, clf_case)
    ):
        sample_raw = X_test.iloc[positions]
        features, names, values, bases, preds, margin = _xgb_contributions(model, sample_raw, task)
        explanation = _make_explanation(values, bases, features, names)
        name_prefix = "regression" if task == "regression" else "classification"
        p_values = out / f"shap_values_{name_prefix}.csv"
        pd.DataFrame(values, columns=names).assign(Id=sample_raw.index.to_numpy()).to_csv(p_values, index=False)
        tables[f"shap_values_{task}"] = p_values
        p_importance = out / f"shap_global_{name_prefix}.csv"
        pd.DataFrame({"Feature": names, "MeanAbsSHAP": np.abs(values).mean(axis=0)})\
            .sort_values("MeanAbsSHAP", ascending=False).to_csv(p_importance, index=False)
        tables[f"shap_global_{task}"] = p_importance
        if task == "regression":
            num_prefix, idx = 3, 5
        else:
            num_prefix, idx = 6, 8
        p = figure_dir / f"5_{num_prefix}_shap_beeswarm_{name_prefix}.png"
        plot_shap_beeswarm(explanation, p)
        figs[f"beeswarm_{task}"] = p
        p = figure_dir / f"5_{num_prefix + 1}_shap_bar_{name_prefix}.png"
        plot_shap_bar(explanation, p)
        figs[f"bar_{task}"] = p
        # Локальный объект: возможно находится вне случайной выборки SHAP.
        local_raw = X_test.iloc[[case_idx]]
        local_data, local_names, local_values, local_base, local_pred, local_margin = \
            _xgb_contributions(model, local_raw, task)
        local_exp = _make_explanation(local_values, local_base, local_data, local_names)[0]
        p = figure_dir / f"5_{idx}_shap_waterfall_{name_prefix}.png"
        plot_shap_waterfall(local_exp, p, "USD" if task == "regression" else "log-odds")
        figs[f"waterfall_{task}"] = p
        p = out / f"force_{name_prefix}.html"
        save_shap_force_html(local_exp, p)
        tables[f"force_{task}"] = p
        pred_label = "Pred_USD" if task == "regression" else "ProbabilityClass1"
        target = float(y_test.iloc[case_idx]) if task == "regression" else int(y_test.iloc[case_idx] >= PRICE_THRESHOLD)
        p = out / f"local_{name_prefix}.csv"
        local_df = pd.DataFrame({
            "Feature": local_names, "FeatureValueScaled": local_data[0],
            "SHAP": local_values[0],
        }).sort_values("SHAP", key=abs, ascending=False)
        local_df.to_csv(p, index=False)
        tables[f"local_{task}"] = p
        meta = {
            "Id": int(X_test.index[case_idx]), "Actual": target, pred_label: float(local_pred[0]),
            "BaseMargin": float(local_base[0]), "PredictionMargin": float(local_margin[0]),
            "InterpretationUnits": "USD" if task == "regression" else "log-odds",
            "Features": len(local_names),
        }
        (out / f"local_{name_prefix}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{task}: {len(names)} признаков; локальный объект Id={meta['Id']}; "
              f"SHAP успешно восстановил прогноз.", flush=True)

    if compare_rf_shap:
        # Дополнительное сравнение на ОДНИХ И ТЕХ ЖЕ объектах, не на разных подвыборках.
        compare_raw = X_test.iloc[positions[:min(80, len(positions))]]
        rf_values, rf_names = _features_and_values(rf_reg, compare_raw)
        _, xg_names, xg_shap, _, _, _ = _xgb_contributions(xgb_reg, compare_raw, "regression")
        rf_explainer = shap.TreeExplainer(rf_reg.named_steps["model"])
        rf_shap = np.asarray(rf_explainer.shap_values(rf_values), dtype=float)
        if rf_shap.shape != rf_values.shape:
            raise ValueError("Форма SHAP для Random Forest не совпала с матрицей")
        rf_expected = np.asarray(rf_explainer.expected_value).item()
        predicted = rf_reg.predict(compare_raw)
        if not np.allclose(rf_expected + rf_shap.sum(axis=1), predicted, atol=.05):
            raise ValueError("SHAP Random Forest не восстановил прогноз")
        rf_imp = pd.Series(np.abs(rf_shap).mean(axis=0), index=rf_names)
        xgb_imp = pd.Series(np.abs(xg_shap).mean(axis=0), index=xg_names)
        compare = pd.DataFrame({"RF_MeanAbsSHAP_USD": rf_imp,
                                "XGBoost_MeanAbsSHAP_USD": xgb_imp}).fillna(0.0)
        compare.index.name = "Feature"
        compare = compare.sort_values("XGBoost_MeanAbsSHAP_USD", ascending=False)
        p = out / "shap_compare_rf_xgb_regression.csv"
        compare.to_csv(p)
        tables["compare_rf_shap"] = p
        p = figure_dir / "5_9_shap_rf_xgb_regression.png"
        plot_rf_xgb_shap_comparison(compare, p)
        figs["compare_rf_shap"] = p
        print("Дополнительное сравнение SHAP Random Forest / XGBoost создано.", flush=True)

    report = out / "explainability_summary.md"
    reg_top = pd.read_csv(out / "shap_global_regression.csv").iloc[0]
    clf_top = pd.read_csv(out / "shap_global_classification.csv").iloc[0]
    reg_meta = json.loads((out / "local_regression.json").read_text(encoding="utf-8"))
    clf_meta = json.loads((out / "local_classification.json").read_text(encoding="utf-8"))
    report.write_text("\n".join([
        "# Интерпретация моделей стоимости недвижимости", "",
        "Проанализированы XGBoost и Random Forest для регрессии и бинарной классификации.",
        "Все обучаемые преобразования получены из исходных train-строк. Внешний тест",
        "использован для описания уже построенных прогнозов, а не для выбора гиперпараметров.", "",
        "## Глобальная важность", "",
        "Встроенная Feature Importance характеризует использование признака моделью,",
        "но не измеряет причинно-следственную связь с рыночной стоимостью.",
        "Для Random Forest применяется уменьшение неоднородности, для XGBoost — относительная важность разделений.",
        "Числовые величины разных алгоритмов не следует напрямую приравнивать.", "",
        f"Наиболее высокий средний абсолютный SHAP-вклад в регрессии: **{reg_top.Feature}**.",
        f"В классификации: **{clf_top.Feature}**.", "",
        "## Локальные объяснения", "",
        f"Регрессия: Id={reg_meta['Id']}, фактическая цена ${reg_meta['Actual']:,.0f}, "
        f"прогноз ${reg_meta['Pred_USD']:,.0f}.",
        f"Классификация: Id={clf_meta['Id']}, фактический класс {clf_meta['Actual']}, "
        f"вероятность дорогого сегмента {clf_meta['ProbabilityClass1']:.1%}.", "",
        "SHAP-регрессия интерпретируется в долларах США. SHAP-классификация — на шкале",
        "логарифма шансов (log-odds), где сумма базового значения и вкладов преобразуется",
        "в вероятность через логистическую функцию.", "",
        "## Ограничения", "",
        "Высокие корреляции могут приводить к распределению вклада между сходными признаками.",
        "Отдельные SHAP-вклады не являются доказательством причинного эффекта.",
        "Выбранные объекты и тест ранее участвовали в исследовательской диагностике,",
        "поэтому этот анализ предназначен для интерпретации, а не независимой проверки качества.", "",
    ]), encoding="utf-8")
    return ExplainabilityResult(report, figs, tables,
                                {"regression": int(X_test.index[reg_case]),
                                 "classification": int(X_test.index[clf_case])})
