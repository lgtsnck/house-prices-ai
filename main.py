"""Точка входа в курсовой проект House Prices AI.

На этапе создания архитектуры доступны сведения о проекте и план работ.
Команды запуска лабораторных будут добавляться по мере реализации.
"""

from __future__ import annotations

import argparse

from house_prices_ai.config import PROJECT_ROOT, RAW_TRAIN_FILE

STAGES: tuple[tuple[int, str, str], ...] = (
    (1, "eda", "Разведочный анализ данных и постановка задач"),
    (2, "preprocessing", "Предобработка данных и инженерия признаков"),
    (3, "regression", "Построение и оценка регрессионных моделей"),
    (4, "classification", "Построение и оценка моделей классификации"),
    (5, "explainability", "Оценка важности признаков и SHAP-анализ"),
    (6, "clustering", "Кластеризация данных и визуализация результатов"),
)


def main(argv: list[str] | None = None) -> int:
    """Выполнить одну из доступных команд проекта."""
    parser = argparse.ArgumentParser(
        description="Курсовой проект по дисциплине «Системы искусственного интеллекта»"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="info",
        choices=("info", "plan"),
        metavar="КОМАНДА",
        help="info — сведения о проекте; plan — план шести лабораторных работ",
    )
    args = parser.parse_args(argv)

    if args.command == "info":
        print("House Prices AI — каркас курсового проекта")
        print(f"Каталог проекта: {PROJECT_ROOT}")
        print(f"Ожидаемый файл обучающих данных Kaggle: {RAW_TRAIN_FILE}")
        print("Команды выполнения лабораторных будут добавляться постепенно.")
    else:
        print("План выполнения лабораторных работ:")
        for number, command, description in STAGES:
            print(f"  {number}. {command:16} {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
