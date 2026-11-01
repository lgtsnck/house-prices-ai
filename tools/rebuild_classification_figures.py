"""Перестроить шесть рисунков классификации из сохранённых CSV без переобучения."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('outputs/classification'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.results
    required = ('cv_summary.csv', 'oof_models.csv',
                'mlp_learning_history.csv', 'mlp_training_summary.json')
    for file in required:
        if not (base / file).is_file():
            parser.error(f'Нет {base/file}; сначала создай результаты классификации')
    summary = pd.read_csv(base / 'cv_summary.csv')
    oof = pd.read_csv(base / 'oof_models.csv')
    history = pd.read_csv(base / 'mlp_learning_history.csv')
    details = json.loads((base / 'mlp_training_summary.json').read_text(encoding='utf-8'))

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from house_prices_ai import classification, visualization
    y = oof['ActualClass'].astype(int)
    if hasattr(classification, 'prepare_classification_plot_data'):
        data = classification.prepare_classification_plot_data(y, oof, summary)
        figures = visualization.create_classification_figures(
            data, summary, args.output, history, details,
        )
    else:
        figures = classification.create_classification_figures(
            y, oof, summary, args.output, history, details,
        )
    print(f'Графиков: {len(figures)}; каталог: {args.output}')
    if len(figures) != 6:
        raise ValueError(f'Ожидалось шесть графиков, получено {len(figures)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
