# Локальные результаты выполнения

Папка `outputs/` создаётся при запуске проекта. Она содержит CSV-таблицы, Markdown-отчёты, PNG-графики, обученные модели Joblib, JSON с настройками и HTML-объяснения SHAP. Всё содержимое результатов исключено из Git и должно воспроизводиться локально из исходного `train.csv`.

Основные подпапки соответствуют командам `main.py`.

| Подпапка | Основной отчёт |
|---|---|
| `eda/` | `lr01_eda_summary.md` |
| `preprocessing/` | `preprocessing_summary.md` |
| `regression/` | `lr03_summary.md` |
| `classification/` | `classification_summary.md` |
| `explainability/` | `explainability_summary.md` |
| `clustering/` | `clustering_summary.md` |

Путь `outputs/<этап>/figures/` используется для изображений там, где их создаёт соответствующий модуль. Для воспроизведения и интерпретации результатов смотри [корневой README](../README.md) и [шесть Notebook](../notebooks/README.md).
