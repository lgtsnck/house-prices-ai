"""Проверить PNG классификации до/после переноса без переобучения."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageChops

EXPECTED = {
    '4_1_class_distribution.png', '4_2_comparison_metrics.png',
    '4_3_confusion_matrices.png', '4_4_roc_comparison.png',
    '4_5_precision_recall.png', '4_6_mlp_training.png',
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('--strict-pixels', action='store_true')
    args = parser.parse_args()
    a = {p.name: p for p in args.baseline.glob('*.png')}
    b = {p.name: p for p in args.current.glob('*.png')}
    errors = []
    if set(a) != EXPECTED or set(b) != EXPECTED:
        errors.append(f'Ожидалось шесть рисунков: исходные {sorted(a)}, новые {sorted(b)}')
    for name in sorted(a.keys() & b.keys()):
        with Image.open(a[name]) as before, Image.open(b[name]) as after:
            if before.size != after.size:
                errors.append(f'{name}: размер {before.size} != {after.size}')
                continue
            if after.width >= after.height:
                errors.append(f'{name}: ориентация не книжная')
            if after.convert('RGB').getpixel((0,0)) != (255,255,255):
                errors.append(f'{name}: не белый фон')
            if args.strict_pixels and ImageChops.difference(
                before.convert('RGB'), after.convert('RGB')
            ).getbbox() is not None:
                errors.append(f'{name}: различаются пиксели')
    print(f'PNG: эталон={len(a)}, новые={len(b)}')
    if errors:
        for problem in errors:
            print('ОШИБКА:', problem)
        return 1
    print('Контроль пройден: состав и геометрия совпали.'
          + (' Пиксели совпадают.' if args.strict_pixels else ' Пиксели не проверялись.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
