"""Пути и общие настройки проекта для Windows и macOS."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_TRAIN_FILE = RAW_DATA_DIR / "train.csv"
RAW_TEST_FILE = RAW_DATA_DIR / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
RANDOM_STATE = 42
