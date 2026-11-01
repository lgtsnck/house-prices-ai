"""Сравнить выгрузки предобработки до/после рефакторинга.

Требуемая структура обоих каталогов:
  preprocessing_summary.md, *.csv, *.png, processed/*.csv
Обученные joblib сравнивать по байтам некорректно; контроль данных и тесты
выполняют отдельную проверку результатов.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageChops


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_extension(folder: Path, suffix: str) -> dict[str, Path]:
    return {str(p.relative_to(folder)).replace('\\', '/'): p
            for p in folder.rglob('*' + suffix) if p.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('--strict-pixels', action='store_true',
                        help='дополнительно сравнить все пиксели изображений')
    args = parser.parse_args()
    before, after = args.baseline, args.current
    if not before.is_dir() or not after.is_dir():
        parser.error('Укажи существующие каталоги эталона и новых результатов')
    errors: list[str] = []
    for extension in ('.csv', '.md'):
        old, new = by_extension(before, extension), by_extension(after, extension)
        if old.keys() != new.keys():
            errors.append(f'Состав {extension}: {sorted(old.keys() ^ new.keys())}')
        for name in old.keys() & new.keys():
            if sha(old[name]) != sha(new[name]):
                errors.append(f'Не совпадает содержимое: {name}')
        print(f'{extension}: файлов эталон={len(old)}, новых={len(new)}')
    old, new = by_extension(before, '.png'), by_extension(after, '.png')
    if old.keys() != new.keys():
        errors.append(f'Состав PNG: {sorted(old.keys() ^ new.keys())}')
    for name in sorted(old.keys() & new.keys()):
        with Image.open(old[name]) as im_old, Image.open(new[name]) as im_new:
            if im_old.size != im_new.size:
                errors.append(f'{name}: размеры {im_old.size} → {im_new.size}')
                continue
            if im_new.width >= im_new.height:
                errors.append(f'{name}: не книжная ориентация')
            rgb = im_new.convert('RGB')
            if rgb.getpixel((0, 0)) != (255, 255, 255):
                errors.append(f'{name}: верхний левый угол не белый')
            if args.strict_pixels:
                if ImageChops.difference(im_old.convert('RGB'), rgb).getbbox() is not None:
                    errors.append(f'{name}: отличается изображение')
    print(f'PNG: эталон={len(old)}, новых={len(new)}')
    if errors:
        print('ОШИБКА:')
        for message in errors:
            print(' -', message)
        return 1
    print('Контроль пройден: выходные CSV и Markdown совпадают побайтово, '
          'состав и геометрия PNG сохранены.' +
          (' Пиксели совпадают.' if args.strict_pixels else ' Пиксели не сравнивались.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
