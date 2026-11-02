# Автоматические проверки

В каталоге хранятся unit-тесты вычислительных модулей, проверки графиков, интерфейса командной строки и взаимодействия компонентов. Основные тесты используют синтетические данные и не должны требовать публикации исходного Kaggle CSV.

Файлы `test_eda*.py`, `test_preprocessing*.py`, `test_regression*.py`, `test_classification*.py`, `test_explainability*.py` и `test_clustering*.py` проверяют соответствующие этапы. `test_project.py` и `test_project_integration.py` проверяют устройство проекта и маршрутизацию команд. `test_visualization_integration.py` и другие тесты визуализации контролируют создание изображений. `test_structural_zeros.py` проверяет обработку структурных нулей.

Запуск из корня репозитория.

```powershell
python -m unittest discover -s tests -q
python -m ruff check house_prices_ai main.py tests tools --select F401,F811,F821,F823,F841
```

Успешные unit-тесты не подтверждают выполнение всех Notebook на исходных данных. Это отдельный этап интеграционной проверки воспроизводимости.
