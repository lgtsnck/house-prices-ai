"""Независимая точка запуска ЛР №3 без изменений существующего main.py."""
from __future__ import annotations

import argparse

from house_prices_ai.regression import run_regression


def main() -> int:
    parser = argparse.ArgumentParser(description="ЛР №3: сравнение регрессионных моделей")
    parser.add_argument("--quick", action="store_true", help="Пробный запуск: 3 фолда вместо 5")
    parser.add_argument("--tune-xgb", action="store_true", help="Дополнительно RandomizedSearchCV для XGBoost")
    parser.add_argument("--tuning-iterations", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_regression(
        folds=3 if args.quick else 5,
        tune=args.tune_xgb,
        tuning_iterations=args.tuning_iterations,
        make_plots=not args.no_plots,
    )
    print("\nЛР №3: черновой запуск завершён.")
    print(f"Лидер CV по MAE: {result.winner}")
    print(f"CV-таблица: {result.summary_path}")
    print(f"Отчёт: {result.report_path}")
    print(f"Диаграмм: {len(result.figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
