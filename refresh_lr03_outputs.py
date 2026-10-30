"""Обновить рисунки и историю MLP, не повторяя полную 5-Fold CV.

Запуск из корня проекта: python refresh_lr03_outputs.py
Требует уже рассчитанных outputs/regression/*.csv и train.csv из ЛР №2.
"""
from __future__ import annotations

import pandas as pd

from house_prices_ai.config import OUTPUT_DIR
from house_prices_ai.regression import load_lr02_split, make_regression_pipeline, model_catalog
from house_prices_ai.evaluation import create_regression_figures, save_mlp_training_diagnostics


def main() -> None:
    out_dir = OUTPUT_DIR / "regression"
    for name in ("cv_clean_summary.csv", "cv_clean_folds.csv", "oof_predictions.csv"):
        if not (out_dir / name).exists():
            raise FileNotFoundError(f"Не найден {out_dir / name}; сначала выполни python run_lr03.py")
    summary = pd.read_csv(out_dir / "cv_clean_summary.csv")
    folds = pd.read_csv(out_dir / "cv_clean_folds.csv")
    oof = pd.read_csv(out_dir / "oof_predictions.csv")

    X_train, _, y_train, _ = load_lr02_split()
    if len(oof) != len(y_train):
        raise ValueError("Число OOF-прогнозов не совпадает с обучающей выборкой")
    if list(oof["Id"]) != list(X_train.index):
        raise ValueError("Порядок OOF Id отличается от сохранённых train_ids")

    print("Обучаем только MLP на полном внешнем train, без повторения всей CV.")
    mlp = make_regression_pipeline(X_train, model_catalog()["MLP"])
    mlp.fit(X_train, y_train)
    info = save_mlp_training_diagnostics(mlp, out_dir)
    print(f"MLP: лучшая эпоха {info['best_epoch']} / {info['epochs_run']}")

    figures = create_regression_figures(
        summary, folds, y_train.to_numpy(), oof["OOF_Pred"].to_numpy(),
        out_dir / "figures", mlp,
    )
    print(f"Готово: {len(figures)} рисунков, включая диаграмму трёх метрик и MLP.")
    print(f"Данные динамики MLP: {out_dir / 'mlp_training_summary.json'}")


if __name__ == "__main__":
    main()
