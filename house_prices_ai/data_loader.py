"""Безопасная локальная загрузка данных соревнования Kaggle."""

from pathlib import Path

import pandas as pd

from house_prices_ai.config import RAW_TRAIN_FILE


def load_training_data(path: Path | str = RAW_TRAIN_FILE) -> pd.DataFrame:
    """Прочитать размеченный набор House Prices без изменения исходного CSV.

    ``Id`` и ``SalePrice`` необходимы для задач исследования.
    ``Id`` используется только для связи записей, не для обучения модели.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл {path} не найден. Получите train.csv на Kaggle после "
            "принятия правил соревнования и поместите его в data/raw/."
        )
    try:
        # В Kaggle значение «None» в MasVnrType представляет допустимую категорию,
        # а не пропуск. Пустое поле и специальный маркер «NA» считаем пропуском.
        data = pd.read_csv(path, keep_default_na=False, na_values=["", "NA"])
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать исходные данные {path}: {exc}") from exc
    required = {"Id", "SalePrice"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные столбцы: {sorted(missing)}")
    if data.empty or data["SalePrice"].isna().all():
        raise ValueError("Датасет пуст или целевой столбец SalePrice целиком пуст.")
    return data
