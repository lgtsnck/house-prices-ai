"""Единая точка входа House Prices AI: все этапы из одного CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from house_prices_ai.config import OUTPUT_DIR, PROJECT_ROOT, RAW_TRAIN_FILE

STAGES = {
    "eda": "Разведочный анализ данных: распределения, пропуски и корреляции",
    "preprocessing": "Подготовка и инженерия признаков",
    "regression": "Прогнозирование стоимости недвижимости",
    "classification": "Определение ценового сегмента",
    "explainability": "Оценка важности признаков и объяснение прогнозов",
    "clustering": "Поиск групп недвижимости без использования цен",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="House Prices AI — исследование данных и машинное обучение")
    parser.add_argument("command", nargs="?", default="info",
                        choices=["info", "plan", *STAGES])
    parser.add_argument("--data", type=Path, default=RAW_TRAIN_FILE,
                        help="Исходный train.csv (EDA / предобработка / регрессия)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Переопределить папку артефактов запуска")
    parser.add_argument("--no-plots", action="store_true",
                        help="Пропустить построение изображений, где поддерживается")
    parser.add_argument("--folds", type=int, default=5,
                        help="Число разбиений CV для регрессии или классификации")
    parser.add_argument("--tune-xgb", action="store_true",
                        help="Подобрать параметры XGBoost при запуске моделирования")
    parser.add_argument("--tune-only", action="store_true",
                        help="Подобрать только XGBoost без полного сравнения")
    parser.add_argument("--tune-iterations", type=int, default=8)
    parser.add_argument("--refresh-plots", action="store_true",
                        help="Обновить изображения и динамику MLP регрессии без повторной CV")
    parser.add_argument("--compare-rf-shap", action="store_true",
                        help="Дополнительное сравнение SHAP Random Forest и XGBoost")
    parser.add_argument("--sample-size", type=int, default=180)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--dbscan", action="store_true")
    parser.add_argument("--dbscan-eps", type=float, default=1.8)
    args = parser.parse_args(argv)

    if args.command == "info":
        print("House Prices AI — моделирование и исследование недвижимости")
        print(f"Каталог проекта: {PROJECT_ROOT}")
        print(f"Ожидаемый файл исходных данных: {RAW_TRAIN_FILE}")
        print("Команды: " + ", ".join(STAGES))
        return 0
    if args.command == "plan":
        for command, description in STAGES.items():
            print(f"{command:16} {description}")
        return 0

    out = args.output or OUTPUT_DIR / args.command
    try:
        if args.command == "eda":
            from house_prices_ai.eda import run_eda
            result = run_eda(args.data, out, make_plots=not args.no_plots)
            print(f"EDA завершён, отчёт: {result.report_path}, изображений: {len(result.plots)}")
        elif args.command == "preprocessing":
            from house_prices_ai.preprocessing import run_preprocessing
            result = run_preprocessing(args.data, output_dir=out, make_plots=not args.no_plots)
            print(f"Подготовлено: train={result.train_rows}, test={result.test_rows}, признаков={result.features_count}")
        elif args.command == "regression":
            if args.refresh_plots:
                if args.output is not None:
                    parser.error("--output не поддерживается для regression --refresh-plots")
                from refresh_lr03_outputs import main as refresh_regression_figures
                refresh_regression_figures()
            elif args.tune_only:
                from house_prices_ai.regression import load_lr02_split, tune_xgboost
                X_train, _, y_train, _ = load_lr02_split(raw_path=args.data)
                out.mkdir(parents=True, exist_ok=True)
                result = tune_xgboost(X_train, y_train, out, iterations=args.tune_iterations)
                print("Подбор XGBoost завершён:", result)
            else:
                from house_prices_ai.regression import run_regression
                result = run_regression(raw_path=args.data, out_dir=out,
                                        folds=args.folds, tune=args.tune_xgb,
                                        tuning_iterations=args.tune_iterations,
                                        make_plots=not args.no_plots)
                print(f"Регрессия завершена, отчёт: {result.report_path}")
        elif args.command == "classification":
            from house_prices_ai.classification import run_classification, tune_only
            if args.tune_only:
                result = tune_only(output_dir=out, iterations=args.tune_iterations)
                print("Подбор XGBoost завершён:", result)
            else:
                result = run_classification(output_dir=out, folds=args.folds,
                                            make_plots=not args.no_plots,
                                            tune_xgb=args.tune_xgb,
                                            tune_iterations=args.tune_iterations)
                print(f"Классификация завершена, отчёт: {result.report_path}")
        elif args.command == "explainability":
            from house_prices_ai.explainability import run_explainability
            result = run_explainability(output_dir=out,
                                        sample_size=args.sample_size,
                                        compare_rf_shap=args.compare_rf_shap)
            print(f"Интерпретация завершена, отчёт: {result.report_path}")
        elif args.command == "clustering":
            from house_prices_ai.clustering import run_clustering
            result = run_clustering(out_dir=out, max_k=args.max_k,
                                    make_plots=not args.no_plots,
                                    tsne=args.tsne, dbscan=args.dbscan,
                                    dbscan_eps=args.dbscan_eps)
            print(f"Кластеризация завершена, отчёт: {result.report_path}")
    except (OSError, ValueError) as exc:
        print(f"Ошибка выполнения: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
