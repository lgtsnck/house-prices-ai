"""Сравнить новый вывод EDA с эталоном (без исходных строк датасета)."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from PIL import Image

CSV_NAMES = ('feature_dictionary.csv', 'missing_values.csv', 'numeric_summary.csv',
             'correlated_feature_pairs.csv')
REPORT = 'lr01_eda_summary.md'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    if not args.baseline.is_dir() or not args.current.is_dir():
        parser.error('Каталог эталона или новых результатов отсутствует')
    errors = []
    for name in (*CSV_NAMES, REPORT):
        a, b = args.baseline / name, args.current / name
        if not (a.is_file() and b.is_file()):
            errors.append(f'Отсутствует {name}')
        elif digest(a) != digest(b):
            errors.append(f'Различается содержимое {name}')
    before = {p.name: p for p in args.baseline.glob('*.png')}
    after = {p.name: p for p in args.current.glob('*.png')}
    if set(before) != set(after):
        errors.append(f'Не совпадают имена PNG: удалены {sorted(set(before) - set(after))}, '
                      f'добавлены {sorted(set(after) - set(before))}')
    for name in sorted(set(before) & set(after)):
        with Image.open(before[name]) as a, Image.open(after[name]) as b:
            if a.size != b.size:
                errors.append(f'{name}: изменились размеры {a.size} → {b.size}')
            if b.width >= b.height:
                errors.append(f'{name}: нарушена книжная ориентация')
            if b.convert('RGB').getpixel((0, 0)) != (255, 255, 255):
                errors.append(f'{name}: фон верхнего левого угла не белый')
    print(f'PNG: эталон {len(before)}, текущие {len(after)}')
    print('CSV и отчёт:', 'совпадают побайтово' if not any('содержимое' in v for v in errors) else 'есть расхождения')
    if errors:
        for error in errors:
            print('ОШИБКА:', error)
        return 1
    print('Контроль EDA пройден. Прогнозная и аналитическая логика не менялась.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
