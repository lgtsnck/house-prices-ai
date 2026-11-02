"""Оценка ценового сегмента недвижимости: CV, сравнение моделей и диагностика.

Все предсказания OOF формируются обучением полного Pipeline на иных строках.
Внешняя тестовая выборка не используется для сравнения или подбора моделей.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from house_prices_ai.config import OUTPUT_DIR, RANDOM_STATE
from house_prices_ai.preprocessing import HouseFeatures, _make_preprocessor
from house_prices_ai.regression import load_lr02_split
from house_prices_ai.visualization import create_classification_figures

PRICE_THRESHOLD = 200_000
METRICS = ("Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "ROC_AUC")



@dataclass
class ClassificationResult:
    summary_path: Path
    folds_path: Path
    oof_path: Path
    report_path: Path
    figures: dict[str, Path]


def build_classification_pipeline(raw_features: pd.DataFrame, estimator) -> Pipeline:
    """Вся обучаемая предобработка входит в CV и в подбор параметров."""
    schema = HouseFeatures().transform(raw_features)
    encoder = _make_preprocessor(schema)
    return Pipeline([("preprocessing", encoder), ("model", clone(estimator))])


def starter_models() -> dict[str, object]:
    """Единый каталог исходных конфигураций, без подбора по внешнему тесту."""
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoostClassifier": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=.01, learning_rate_init=.001, max_iter=700,
            early_stopping=True, validation_fraction=.15,
            n_iter_no_change=25, random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true, labels, scores) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, labels),
        "BalancedAccuracy": balanced_accuracy_score(y_true, labels),
        "Precision": precision_score(y_true, labels, zero_division=0),
        "Recall": recall_score(y_true, labels, zero_division=0),
        "F1": f1_score(y_true, labels, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
    }


def compare_classifiers(X: pd.DataFrame, y: pd.Series, folds: int = 5,
                        models: dict[str, object] | None = None,
                        return_oof: bool = False):
    """CV и OOF за один проход, одинаковые разбиения для всех моделей.

    return_oof=False сохраняет интерфейс существующих тестов.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    if len(X) != len(y):
        raise ValueError("Количество объектов и меток должно совпадать")
    if folds < 2 or y.nunique() != 2 or y.value_counts().min() < folds:
        raise ValueError("Для StratifiedKFold необходимы оба класса и достаточное число объектов")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    models = starter_models() if models is None else models
    summary_rows, fold_rows = [], []
    oof = pd.DataFrame({"ActualClass": y.to_numpy()})

    for name, estimator in models.items():
        print(f"CV: {name}", flush=True)
        labels = np.full(len(y), -1, dtype=int)
        probabilities = np.full(len(y), np.nan, dtype=float)
        scores_by_fold = []
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            pipe = build_classification_pipeline(X, estimator)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            labels[val_idx] = pipe.predict(X.iloc[val_idx])
            probabilities[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            values = _metrics(y.iloc[val_idx], labels[val_idx], probabilities[val_idx])
            fold_rows.append({"Model": name, "Fold": fold, **values})
            scores_by_fold.append(values)
        if (labels < 0).any() or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Незаполненные OOF-прогнозы: {name}")
        row = {"Model": name}
        for metric in METRICS:
            arr = np.array([item[metric] for item in scores_by_fold], dtype=float)
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std())
        summary_rows.append(row)
        oof[f"PredictedClass__{name}"] = labels
        oof[f"ProbabilityClass1__{name}"] = probabilities

    summary = pd.DataFrame(summary_rows).sort_values("F1_mean", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    if return_oof:
        return summary, fold_df, oof
    return summary, fold_df


def prepare_classification_plot_data(
    y: pd.Series, oof: pd.DataFrame, summary: pd.DataFrame,
) -> dict:
    """Подготовить числовые массивы для шести диаграмм классификации.

    ROC, PR, матрицы ошибок и базовые частоты рассчитываются в этом модуле,
    графический слой получает исключительно готовые числовые данные.
    """
    y_true = np.asarray(y, dtype=int)
    models = [name for name in summary["Model"] if name != "DummyClassifier"]
    confusions, roc_curves, pr_curves = {}, {}, {}
    for name in models:
        predicted = oof[f"PredictedClass__{name}"].to_numpy(dtype=int)
        probs = oof[f"ProbabilityClass1__{name}"].to_numpy(dtype=float)
        confusions[name] = confusion_matrix(y_true, predicted, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_curves[name] = (fpr, tpr, roc_auc_score(y_true, probs))
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_curves[name] = (
            recall, precision, average_precision_score(y_true, probs),
        )
    return {
        "models": models,
        "counts": pd.Series(y_true).value_counts().reindex([0, 1], fill_value=0),
        "n_objects": len(y_true),
        "prevalence": float(y_true.mean()),
        "confusions": confusions,
        "roc": roc_curves,
        "pr": pr_curves,
    }


def save_mlp_history(model: Pipeline, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """История отдельного fit на полном внешнем train, не история CV."""
    mlp: MLPClassifier = model.named_steps["model"]
    losses = np.asarray(mlp.loss_curve_, dtype=float)
    scores = np.asarray(mlp.validation_scores_, dtype=float)
    history = pd.DataFrame({
        "Epoch": np.arange(1, len(losses) + 1),
        "TrainingLoss": losses,
        "ValidationAccuracy": scores,
    })
    history.to_csv(output_dir / "mlp_learning_history.csv", index=False)
    if np.isfinite(scores).any():
        best_epoch = int(np.nanargmax(scores)) + 1
        best_score = float(scores[best_epoch - 1])
    else:
        best_epoch, best_score = None, None
    details = {
        "epochs": int(mlp.n_iter_), "best_epoch": best_epoch,
        "best_validation_accuracy": best_score,
        "first_training_loss": float(losses[0]),
        "last_training_loss": float(losses[-1]),
        "early_stopping": bool(mlp.early_stopping),
        "validation_fraction": float(mlp.validation_fraction),
        "note": "MLPClassifier early_stopping monitors accuracy, not ROC-AUC or F1."
    }
    (output_dir / "mlp_training_summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return history, details


def tune_xgboost(X: pd.DataFrame, y: pd.Series, output_dir: Path,
                 iterations: int = 8) -> dict:
    """Ограниченный поиск по F1: 3-fold CV, внешний test не используется.

    Результат поиска не является несмещённой оценкой оптимизированной модели.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_classification_pipeline(X, starter_models()["XGBoostClassifier"])
    space = {
        "model__n_estimators": [200, 300, 450],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [.025, .05, .08],
        "model__subsample": [.75, .9, 1.0],
        "model__colsample_bytree": [.75, .9, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1.0, 3.0, 6.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1)
    search = RandomizedSearchCV(
        pipeline, param_distributions=space, n_iter=iterations,
        scoring="f1", cv=cv, n_jobs=1, refit=True,
        random_state=RANDOM_STATE, error_score="raise", verbose=1,
    )
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(output_dir / "xgboost_tuning_trials.csv", index=False)
    params = {
        k.replace("model__", ""): (v.item() if isinstance(v, np.generic) else v)
        for k, v in search.best_params_.items()
    }
    result = {
        "best_params": params, "search_cv_f1": float(search.best_score_),
        "n_iter": iterations, "cv_folds": 3, "selection_metric": "F1",
        "note": "Search score is for selection, not an independent final evaluation."
    }
    (output_dir / "xgboost_tuning.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(search.best_estimator_, output_dir / "xgboost_tuned_pipeline.joblib")
    return result


def _report(X, y, summary, oof, mlp_info, tuning=None) -> str:
    best = summary.iloc[0]
    cols = ["Model", "Accuracy_mean", "Precision_mean", "Recall_mean", "F1_mean", "ROC_AUC_mean"]
    lines = [
        "# Классификация ценового сегмента недвижимости", "",
        f"Для оценки ценового сегмента использованы {len(X):,} исходных обучающих наблюдений. "
        f"Положительный класс соответствует цене сделки от ${PRICE_THRESHOLD:,}.",
        f"Доля положительного класса составляет {y.mean():.1%}. Признаки целевой цены и идентификаторы не используются.", "",
        "## Методика оценки", "",
        "Пять стратифицированных разбиений выполнены на одной и той же выборке для всех моделей. "
        "Заполнение пропусков, обработка экстремальных значений, масштабирование и категориальное кодирование "
        "обучаются независимо внутри каждого разбиения. Внешняя тестовая выборка не использовалась.", "",
        "## Сравнение алгоритмов", "", "```text", summary[cols].round(4).to_string(index=False), "```", "",
        f"Наибольший средний F1 при исходных настройках показал {best['Model']} "
        f"({best['F1_mean']:.4f}). Различия в средних метриках не подтверждают сами по себе статистическую значимость.", "",
        "## Анализ ошибок и устойчивости", "",
        "Объединённые OOF-прогнозы используются для построения матриц ошибок, ROC-кривых "
        "и Precision–Recall-кривых. Показатели AUC и Average Precision по объединённым OOF-прогнозам "
        "могут отличаться от средних значений по пяти разбиениям.",
    ]
    for name in summary["Model"]:
        if name == "DummyClassifier":
            continue
        pred = oof[f"PredictedClass__{name}"].to_numpy(dtype=int)
        true = oof["ActualClass"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0,1]).ravel()
        lines.append(f"{name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, "
                     f"OOF F1={f1_score(true,pred):.4f}.")
    lines += ["", "## Обучение многослойного персептрона", "",
              f"MLP обучался {mlp_info['epochs']} эпох. Лучший показатель внутренней "
              f"Validation Accuracy={mlp_info['best_validation_accuracy']:.4f} "
              f"получен на эпохе {mlp_info['best_epoch']}. "
              "Ранняя остановка использует Accuracy, поэтому снижение Training Loss "
              "не обязательно сопровождается улучшением этой метрики. "
              "Отдельная история обучения не является метрикой CV. "
              "Предобработка обучается до внутреннего разбиения MLP, поэтому этот "
              "Validation Accuracy используется только как критерий остановки, "
              "а не как независимая оценка обобщения.", ""]
    if tuning:
        lines += ["## Подбор параметров XGBoost", "",
                  f"Выполнен RandomizedSearchCV: {tuning['n_iter']} комбинаций, "
                  f"{tuning['cv_folds']} стратифицированных разбиения, критерий F1. "
                  f"Лучшее среднее F1 поиска: {tuning['search_cv_f1']:.4f}.",
                  "Параметры подбирались по числу деревьев, глубине, скорости обучения, "
                  "доле строк и признаков, минимальному весу узла и L2-регуляризации.",
                  "Лучший результат поиска не следует напрямую сравнивать с независимой "
                  "пятифолдовой оценкой исходной модели, поскольку это разные процедуры.", "",
                  "```json", json.dumps(tuning["best_params"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Итоговые выводы", "",
              "Сравнение показывает, что модели, использующие нелинейные зависимости, "
              "в текущем эксперименте обеспечивают высокое качество разделения ценовых сегментов. "
              "Выбор основного алгоритма производится по среднему F1 при контроле Precision, Recall, "
              "ROC-AUC и устойчивости между разбиениями. Дополнительная интерпретация признаков "
              "и проверка редких типов объектов предусмотрены в последующем анализе.", ""]
    return "\n".join(lines)


def run_classification(output_dir: str | Path = OUTPUT_DIR / "classification",
                       folds: int = 5, make_plots: bool = True,
                       tune_xgb: bool = False, tune_iterations: int = 8) -> ClassificationResult:
    X, _, y_price, _ = load_lr02_split()
    X = X.reset_index(drop=True)
    y = (pd.Series(y_price).reset_index(drop=True) >= PRICE_THRESHOLD).astype(int)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, fold_df, oof = compare_classifiers(X, y, folds=folds, return_oof=True)
    summary_path, folds_path, oof_path = out / "cv_summary.csv", out / "cv_folds.csv", out / "oof_models.csv"
    summary.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    oof.to_csv(oof_path, index=False)
    # Сохраняем старый формат OOF для обратной совместимости.
    oof[["ActualClass", "PredictedClass__LogisticRegression", "ProbabilityClass1__LogisticRegression"]].rename(
        columns={"PredictedClass__LogisticRegression": "PredictedClass",
                 "ProbabilityClass1__LogisticRegression": "ProbabilityClass1"}
    ).to_csv(out / "oof_logistic.csv", index=False)

    # Финальный fit кандидата на всем внешнем train. Внешний test не трогаем.
    winner = str(summary.iloc[0]["Model"])
    best_pipe = build_classification_pipeline(X, starter_models()[winner])
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, out / "best_classification_pipeline.joblib")
    (out / "best_model.json").write_text(
        json.dumps({"model": winner, "selection_metric": "mean 5-fold F1",
                    "external_test_used": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # MLP: отдельное полное обучение для графика, независимо от CV.
    mlp = build_classification_pipeline(X, starter_models()["MLPClassifier"])
    mlp.fit(X, y)
    mlp_history, mlp_info = save_mlp_history(mlp, out)
    if make_plots:
        visual_data = prepare_classification_plot_data(y, oof, summary)
        figures = create_classification_figures(
            visual_data, summary, out / "figures", mlp_history, mlp_info
        )
    else:
        figures = {}
    tuned = tune_xgboost(X, y, out, iterations=tune_iterations) if tune_xgb else None
    if tuned is None and (out / "xgboost_tuning.json").is_file():
        tuned = json.loads((out / "xgboost_tuning.json").read_text(encoding="utf-8"))
    report_path = out / "classification_summary.md"
    report_path.write_text(_report(X, y, summary, oof, mlp_info, tuned) + "\n", encoding="utf-8")
    return ClassificationResult(summary_path, folds_path, oof_path, report_path, figures)


def tune_only(output_dir: str | Path = OUTPUT_DIR / "classification", iterations: int = 8):
    """Только настройка XGBoost, без повторного расчёта всех моделей CV."""
    X, _, y_price, _ = load_lr02_split()
    X = X.reset_index(drop=True)
    y = (pd.Series(y_price).reset_index(drop=True) >= PRICE_THRESHOLD).astype(int)
    out = Path(output_dir)
    tuning = tune_xgboost(X, y, out, iterations=iterations)
    summary_file, oof_file, mlp_file = (out / "cv_summary.csv", out / "oof_models.csv",
                                           out / "mlp_training_summary.json")
    if all(path.is_file() for path in (summary_file, oof_file, mlp_file)):
        summary = pd.read_csv(summary_file)
        oof = pd.read_csv(oof_file)
        mlp_info = json.loads(mlp_file.read_text(encoding="utf-8"))
        (out / "classification_summary.md").write_text(
            _report(X, y, summary, oof, mlp_info, tuning) + "\n", encoding="utf-8"
        )
    return tuning
