"""Оценка ценового сегмента недвижимости: CV, сравнение моделей и диагностика.

Все предсказания OOF формируются обучением полного Pipeline на иных строках.
Внешняя тестовая выборка не используется для сравнения или подбора моделей.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

PRICE_THRESHOLD = 200_000
METRICS = ("Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "ROC_AUC")
COLORS = {
    "DummyClassifier": "#8d969f",
    "LogisticRegression": "#427bb2",
    "RandomForestClassifier": "#399079",
    "XGBoostClassifier": "#cd8d37",
    "MLPClassifier": "#8c75b4",
}


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


def _save(fig: plt.Figure, path: Path) -> None:
    fig.set_facecolor("white")
    fig.savefig(path, dpi=185, bbox_inches="tight", pad_inches=.24, facecolor="white")
    plt.close(fig)


def _figure() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(8.27, 11.69), facecolor="white")


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


def create_classification_figures(y: pd.Series, oof: pd.DataFrame,
                                  summary: pd.DataFrame, output_dir: Path,
                                  mlp_history: pd.DataFrame | None = None,
                                  mlp_details: dict | None = None) -> dict[str, Path]:
    """Шесть рисунков A4: состав выборки, сравнение, матрицы, ROC, PR, MLP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    models = [name for name in summary["Model"] if name != "DummyClassifier"]
    y_true = np.asarray(y, dtype=int)

    fig, ax = _figure()
    counts = pd.Series(y_true).value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["Ниже $200 000", "От $200 000"], counts,
                  color=["#427bb2", "#cd8d37"], width=.58)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{count} ({count/len(y_true):.1%})", ha="center", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set_ylabel("Число объектов")
    ax.grid(axis="y", alpha=.18)
    fig.subplots_adjust(left=.17, right=.94, bottom=.13, top=.91)
    figures["balance"] = output_dir / "4_1_class_distribution.png"
    _save(fig, figures["balance"])

    # Три отдельных шкалы, чтобы метрики не смешивались в одной системе координат.
    fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), facecolor="white")
    measure = (("Accuracy_mean", "Accuracy"), ("Precision_mean", "Precision"),
               ("Recall_mean", "Recall"), ("F1_mean", "F1-score"))
    names = summary["Model"].tolist()[::-1]
    for ax, (col, title) in zip(axes, measure):
        vals = summary.set_index("Model").loc[names, col]
        bars = ax.barh(names, vals, color=[COLORS.get(n, "#64748b") for n in names], height=.63)
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
    _save(fig, figures["comparison"])

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69), facecolor="white")
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, models):
        labels = oof[f"PredictedClass__{name}"].to_numpy(dtype=int)
        cm = confusion_matrix(y_true, labels, labels=[0, 1])
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
    _save(fig, figures["confusion"])

    fig, ax = _figure()
    for name in models:
        probs = oof[f"ProbabilityClass1__{name}"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(name), label=f"{name}, AUC={auc:.3f}")
    ax.plot([0,1], [0,1], linestyle="--", color="#9299a3", lw=1.2, label="Случайный ориентир")
    ax.set_xlabel("FPR — доля ложноположительных")
    ax.set_ylabel("TPR — полнота")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=.16)
    ax.legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(left=.17, right=.95, top=.93, bottom=.14)
    figures["roc"] = output_dir / "4_4_roc_comparison.png"
    _save(fig, figures["roc"])

    fig, ax = _figure()
    for name in models:
        probs = oof[f"ProbabilityClass1__{name}"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, lw=2, color=COLORS.get(name), label=f"{name}, AP={ap:.3f}")
    ax.axhline(y_true.mean(), ls="--", lw=1.2, color="#9299a3",
               label=f"Частота класса 1: {y_true.mean():.1%}")
    ax.set_xlabel("Recall — полнота")
    ax.set_ylabel("Precision — точность положительных решений")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=.16)
    ax.legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(left=.17, right=.95, top=.93, bottom=.14)
    figures["pr"] = output_dir / "4_5_precision_recall.png"
    _save(fig, figures["pr"])

    if mlp_history is not None and len(mlp_history):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor="white")
        e = mlp_history["Epoch"]
        ax1.plot(e, mlp_history["TrainingLoss"], color=COLORS["MLPClassifier"], linewidth=2)
        ax1.set_ylabel("Training loss")
        ax1.set_xlabel("Эпоха")
        ax1.grid(alpha=.16)
        ax2.plot(e, mlp_history["ValidationAccuracy"], color=COLORS["LogisticRegression"], linewidth=2)
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
        _save(fig, figures["mlp"])
    return figures


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
    figures = (create_classification_figures(y, oof, summary, out / "figures", mlp_history, mlp_info)
               if make_plots else {})
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Классификация недвижимости и сравнение алгоритмов")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tune-xgb", action="store_true", help="Полный расчёт и подбор XGBoost")
    group.add_argument("--tune-only", action="store_true", help="Только подбор XGBoost без повторной CV")
    p.add_argument("--tune-iterations", type=int, default=8)
    args = p.parse_args()
    if args.tune_only:
        tuning = tune_only(iterations=args.tune_iterations)
        print(f"Подбор завершён. Лучший средний CV F1: {tuning['search_cv_f1']:.4f}")
    else:
        result = run_classification(tune_xgb=args.tune_xgb, tune_iterations=args.tune_iterations)
        print("\nСравнение классификаторов завершено.")
        print(f"Метрики: {result.summary_path}")
        print(f"OOF: {result.oof_path}")
        print(f"Отчёт: {result.report_path}")
        print(f"Рисунков: {len(result.figures)}")
