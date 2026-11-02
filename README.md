# House Prices AI

**House Prices AI** - единый исследовательский проект по анализу исторических продаж жилой недвижимости в Эймсе, штат Айова. Работа выполнена на наборе данных [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) и охватывает разведочный анализ, подготовку данных, регрессию, классификацию, интерпретацию моделей и кластеризацию.

Проект ориентирован на требования уровня «5» дисциплины «Системы искусственного интеллекта». Расчётная логика находится в пакете `house_prices_ai`, а шесть Jupyter Notebook содержат единое исследование с таблицами, рисунками и выводами. Текстовая версия исследования собрана в [`docs/README.md`](docs/README.md).

## Быстрый старт на Windows

Для проверенной конфигурации используется Python 3.13 и PowerShell. Команды ниже выполняются в корне проекта.

```powershell
git clone https://github.com/lgtsnck/house-prices-ai.git
cd house-prices-ai
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Если PowerShell блокирует активацию, можно запускать `\.venv\Scripts\python.exe` напрямую вместо изменения системной политики выполнения скриптов. В PyCharm следует выбрать интерпретатор `.venv\Scripts\python.exe`.

**Исходные данные не входят в репозиторий.** Для воспроизведения результатов необходимо принять условия соревнования Kaggle, скачать `train.csv` со [страницы данных](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) и разместить его в `data/raw/train.csv`. Это единственный обязательный внешний CSV. Файлы `test.csv` и `data_description.txt` при необходимости можно положить рядом. Подробнее в [`data/README.md`](data/README.md).

Для проверки установки и работы командного интерфейса используются следующие команды.

```powershell
python main.py info
python main.py plan
python -m unittest discover -s tests -q
python tools/verify_notebooks.py
```

## Запуск исследования

Рекомендуется выполнять этапы в указанном порядке. Каждый шаг создаёт локальные результаты, которые используются в последующих исследованиях.

```powershell
python main.py eda
python main.py preprocessing
python main.py regression
python main.py classification
python main.py explainability
python main.py clustering
```

| Этап | Назначение | Основные результаты |
|---|---|---|
| `eda` | Разведочный анализ, описательная статистика, распределения, корреляции и PCA | `outputs/eda/` |
| `preprocessing` | Пропуски, ограничение выбросов, кодирование, инженерия признаков, VIF | `data/processed/`, `outputs/preprocessing/` |
| `regression` | Сопоставление регрессионных моделей по CV и анализ остатков | `outputs/regression/` |
| `classification` | Определение ценовой категории, сравнение моделей и матрицы ошибок | `outputs/classification/` |
| `explainability` | Важность признаков и глобальные и локальные объяснения SHAP | `outputs/explainability/` |
| `clustering` | K-means, силуэт, PCA и описание групп недвижимости | `outputs/clustering/` |

Дополнительные эксперименты запускаются отдельно, чтобы обычный сценарий не выполнял длительный подбор гиперпараметров и расширенные визуализации.

```powershell
# Подбор параметров XGBoost для регрессии
python main.py regression --tune-only

# Подбор параметров XGBoost для классификации
python main.py classification --tune-only

# Дополнительное сравнение SHAP Random Forest и XGBoost
python main.py explainability --compare-rf-shap

# Дополнительные проекции и плотностная кластеризация
python main.py clustering --tsne --dbscan
```

Параметры команд можно посмотреть через `python main.py --help`. Подбор параметров не следует автоматически считать независимой оценкой качества настроенной модели.

## Где посмотреть результаты

**Интерактивное исследование** находится в [`notebooks/`](notebooks/README.md). Файлы `.ipynb` можно открыть в PyCharm или JupyterLab. Для запуска JupyterLab из корня репозитория используется команда `python -m jupyterlab`. Затем необходимо выбрать ядро виртуального окружения. Notebook используют локальные файлы из `outputs/` и `data/processed/`, поэтому сначала выполняются соответствующие команды `main.py`.

**Сводные отчёты** создаются в Markdown. Основные файлы приведены ниже.

| Этап | Локальный отчёт |
|---|---|
| Анализ данных | `outputs/eda/lr01_eda_summary.md` |
| Предобработка | `outputs/preprocessing/preprocessing_summary.md` |
| Регрессия | `outputs/regression/lr03_summary.md` |
| Классификация | `outputs/classification/classification_summary.md` |
| Интерпретация | `outputs/explainability/explainability_summary.md` |
| Кластеризация | `outputs/clustering/clustering_summary.md` |

В подпапках `figures/` сохраняются PNG, рядом могут находиться таблицы CSV, модели Joblib и интерактивные объяснения SHAP в HTML. Папка `outputs/` создаётся при выполнении проекта и исключена из Git. Поэтому отсутствие локальных результатов после `git clone` ожидаемо.

**Единое исследование** доступно в [`docs/README.md`](docs/README.md). В документ включены тексты шести разделов, сохранённые таблицы, иллюстрации, выводы и раскрываемые блоки исходного кода. Для воспроизведения экспериментов используются исходные данные и программные модули проекта.

## Структура каталогов

| Каталог | Назначение |
|---|---|
| [`house_prices_ai/`](house_prices_ai/README.md) | Загрузка данных, подготовка признаков, машинное обучение и расчёты |
| [`house_prices_ai/visualization/`](house_prices_ai/visualization/README.md) | Построение диаграмм отдельно от вычислительной логики |
| [`data/`](data/README.md) | Локальные исходные и подготовленные данные |
| [`notebooks/`](notebooks/README.md) | Шесть исследовательских Notebook в парных форматах `.ipynb` и `.py` |
| [`docs/`](docs/README.md) | Объединённое исследование и материалы документации |
| [`docs/assets/`](docs/assets/README.md) | Иллюстрации, встроенные в единое исследование |
| [`tests/`](tests/README.md) | Автоматические проверки |
| [`tools/`](tools/README.md) | Вспомогательные средства сверки изображений и Notebook |
| [`outputs/`](outputs/README.md) | Локальные результаты запусков, не включаемые в Git |

Файл `main.py` содержит единый командный интерфейс. В `requirements.txt` перечислены прямые зависимости проекта с диапазонами допустимых версий, но это не lock-файл с точными версиями транзитивных пакетов.

## Проверка проекта

```powershell
python -m pip check
python -m compileall -q main.py house_prices_ai tests tools
python -m ruff check house_prices_ai main.py tests tools --select F401,F811,F821,F823,F841
python -m unittest discover -s tests -q
python tools/verify_notebooks.py
```

Unit-тесты проверяют основные компоненты и взаимодействие модулей. Структурная проверка Notebook не означает повторного выполнения всех ячеек или переобучения моделей. Для полной проверки воспроизводимости нужно выполнить команды исследования на исходном `train.csv` и изучить сформированные результаты.

Проверенное окружение на Windows использовало Python 3.13. Проверка на macOS с Apple Silicon запланирована отдельно, поэтому совместимость этой платформы пока не подтверждается.

## Ограничения исследования

Модели анализируют исторические данные и не гарантируют точных прогнозов для новых объектов. Подбор параметров и оценка по CV выполняются на обучающей выборке. Внешний тест ранее использовался для диагностики отдельных домов, поэтому его результаты не представляются как новая полностью независимая проверка. SHAP объясняет предсказания модели, но не доказывает причинное влияние характеристик на стоимость. Группы кластеризации определяются без использования цены и затем сопоставляются с ней только для описательного анализа.
