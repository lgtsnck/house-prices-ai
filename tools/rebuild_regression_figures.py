"""Перестроить 6 диаграмм регрессии из CSV без обучения моделей.

Работает до рефакторинга (evaluation.py) и после него (visualization.py).
Используется для независимого сравнения макетов PNG в одной среде.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('outputs/regression'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.results
    required = ['cv_clean_summary.csv', 'cv_clean_folds.csv',
                'oof_predictions.csv', 'mlp_learning_history.csv']
    for name in required:
        if not (base / name).is_file():
            parser.error(f'Не найден {base / name}; сначала получи исходные результаты регрессии')

    summary = pd.read_csv(base / 'cv_clean_summary.csv')
    folds = pd.read_csv(base / 'cv_clean_folds.csv')
    oof = pd.read_csv(base / 'oof_predictions.csv')
    history = pd.read_csv(base / 'mlp_learning_history.csv')

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from house_prices_ai import evaluation, visualization
    if hasattr(visualization, 'create_regression_figures'):
        # После переноса используются только подготовленные числовые данные.
        plots = visualization.create_regression_figures(
            summary, folds, oof['SalePrice'].to_numpy(),
            oof['OOF_Pred'].to_numpy(), args.output, history,
        )
    else:
        # До переноса использовался адаптер истории для прежнего интерфейса.
        reg = SimpleNamespace(
            loss_curve_=history['Training_Loss'].to_numpy(),
            validation_scores_=history['Validation_R2'].to_numpy(),
        )
        pipeline = SimpleNamespace(named_steps={'model': SimpleNamespace(regressor_=reg)})
        plots = evaluation.create_regression_figures(
            summary, folds, oof['SalePrice'].to_numpy(),
            oof['OOF_Pred'].to_numpy(), args.output, pipeline,
        )
    print(f'Построено {len(plots)} рисунков: {args.output}')
    if len(plots) != 6:
        raise SystemExit(f'Ожидалось 6 графиков, получено {len(plots)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
