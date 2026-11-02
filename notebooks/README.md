# Интерактивное исследование

Здесь находится последовательное исследование House Prices AI из шести разделов. Каждый раздел представлен Jupyter Notebook `.ipynb` с сохранёнными выводами и текстовой копией `.py` в формате Jupytext Percent. Оба файла образуют одну логическую пару и должны поддерживаться согласованно.

| Раздел | Notebook | Содержание |
|---|---|---|
| 1. Разведочный анализ | [`01_exploratory_data_analysis.ipynb`](01_exploratory_data_analysis.ipynb) | Структура данных, пропуски, распределения, связи, PCA |
| 2. Предобработка | [`02_data_preprocessing.ipynb`](02_data_preprocessing.ipynb) | Преобразование признаков, IQR, VIF, кодирование и разделение данных |
| 3. Регрессия | [`03_regression_models.ipynb`](03_regression_models.ipynb) | Сравнение моделей, CV, остатки, подбор XGBoost |
| 4. Классификация | [`04_classification_models.ipynb`](04_classification_models.ipynb) | Метрики, матрицы ошибок, ROC, Precision–Recall, MLP |
| 5. Интерпретация | [`05_model_explainability.ipynb`](05_model_explainability.ipynb) | Feature Importance и SHAP для регрессии и классификации |
| 6. Кластеризация | [`06_clustering.ipynb`](06_clustering.ipynb) | K-means, силуэт, PCA, t-SNE, DBSCAN и профили групп |

## Выполнение

Перед запуском установи библиотеки из корневого `requirements.txt`, помести файл Kaggle в `data/raw/train.csv` и выполни соответствующие команды `python main.py ...` согласно [основной инструкции](../README.md). Notebook читают локальные CSV, PNG и другие артефакты из `outputs/`. Открой `.ipynb` в PyCharm либо JupyterLab и выбери Python из `.venv`.

Выполняй ячейки сверху вниз. Для разделов 3–6 требуется предварительная подготовка данных и результатов соответствующих моделей. Дополнительные SHAP- и кластерные рисунки могут появляться только после запуска команд с необязательными параметрами.

## Синхронизация представлений

После изменения одной версии Notebook можно синхронизировать пару через Jupytext.

```powershell
jupytext --sync notebooks/01_exploratory_data_analysis.py
python tools/verify_notebooks.py
```

Перед синхронизацией сохрани файлы и убедись, что между ними нет независимых правок. При массовой редактуре нужно отдельно проверять, что сохранились изображения, текстовые outputs и метаданные выполнения. Структурная проверка не заменяет повторного выполнения ячеек.

Сводное изложение исследования без дублирования кода находится в [`docs/README.md`](../docs/README.md).
