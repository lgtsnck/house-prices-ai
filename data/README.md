# Данные

Источник: [Kaggle House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

- `raw/train.csv` — обучающие записи с `SalePrice`; скачать из Kaggle отдельно.
- `raw/test.csv` — тестовый файл соревнования Kaggle без `SalePrice` (не использовать для вычисления тестовых метрик модели).
- `processed/` — подготовленные признаки и целевые переменные после ЛР №2.

Данные не публикуются в GitHub. Описание и статистику набора данных добавим в ЛР №1 после фактического EDA. Некоторые `NaN` в Ames Housing обозначают отсутствие характеристики (например, гаража), а не обязательно случайную потерю данных; их следует исследовать перед импутацией.
