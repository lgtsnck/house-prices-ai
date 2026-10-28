"""Единая точка запуска курсового проекта House Prices AI."""

from __future__ import annotations

import argparse

from house_prices_ai.config import OUTPUT_DIR, PROJECT_ROOT, RAW_TRAIN_FILE

STAGES: tuple[tuple[int, str, str], ...] = (
    (1, "eda", "Разведочный анализ данных и постановка задач"),
    (2, "preprocessing", "Предобработка данных и инженерия признаков"),
    (3, "regression", "Построение и оценка регрессионных моделей"),
    (4, "classification", "Построение и оценка моделей классификации"),
    (5, "explainability", "Оценка важности признаков и SHAP-анализ"),
    (6, "clustering", "Кластеризация данных и визуализация результатов"),
)


def main(argv: list[str] | None = None) -> int:
    """Выполнить команду программы и вернуть код завершения."""
    parser = argparse.ArgumentParser(
        description="Курсовой проект по дисциплине «Системы искусственного интеллекта»"
    )
    parser.add_argument(
        "command", nargs="?", default="info", choices=("info", "plan", "eda", "preprocessing"),
        help="info — сведения; plan — этапы; eda — анализ данных; preprocessing — подготовка данных",
    )
    parser.add_argument("--data", default=str(RAW_TRAIN_FILE), help="Путь к train.csv (для eda)")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "eda"), help="Каталог результатов (для eda)")
    parser.add_argument("--no-plots", action="store_true", help="Не строить графики при тестовом запуске EDA")
    args = parser.parse_args(argv)

    if args.command == "info":
        print("House Prices AI — курсовой проект по СИИ")
        print(f"Каталог проекта: {PROJECT_ROOT}")
        print(f"Ожидаемый файл обучающих данных Kaggle: {RAW_TRAIN_FILE}")
        print("Доступны команды info, plan и eda. Остальные лабораторные будут добавлены далее.")
    elif args.command == "plan":
        print("План выполнения лабораторных работ:")
        for number, command, description in STAGES:
            print(f"  {number}. {command:16} {description}")
    elif args.command == "eda":
        from house_prices_ai.eda import run_eda

        try:
            result = run_eda(args.data, args.output, make_plots=not args.no_plots)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Ошибка выполнения ЛР №1: {exc}")
            return 2
        print("ЛР №1 — разведочный анализ завершён.")
        print(f"Размер: {result.observations} строк × {result.columns} столбцов")
        print(f"Пропущенных значений: {result.missing_total}")
        print(f"Отчёт: {result.report_path}")
        print(f"Создано графиков: {len(result.plots)}")
    elif args.command == "preprocessing":
        from house_prices_ai.preprocessing import run_preprocessing

        try:
            result = run_preprocessing(args.data)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Ошибка предобработки: {exc}")
            return 2
        print("Предобработка и инженерия признаков завершены.")
        print(f"Train: {result.train_rows} строк, test: {result.test_rows} строк")
        print(f"Признаков после кодирования: {result.features_count}")
        print(f"Отчёт: {result.report_path}")
        print(f"Сформировано графиков: {len(result.generated_plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
