"""Сравнение графиков регрессии до и после рефакторинга."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageChops

EXPECTED = {
    '3_1_cv_mae.png', '3_2_cv_r2_folds.png',
    '3_3_oof_actual_vs_pred.png', '3_4_oof_residuals.png',
    '3_5_oof_qq.png', '3_6_mlp_training.png',
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('--strict-pixels', action='store_true')
    args = parser.parse_args()
    errors = []
    before = {p.name:p for p in args.baseline.glob('*.png')}
    after = {p.name:p for p in args.current.glob('*.png')}
    if set(before) != EXPECTED or set(after) != EXPECTED:
        errors.append(f'Ожидались шесть рисунков. Эталон: {sorted(before)}; новые: {sorted(after)}')
    for name in sorted(set(before) & set(after)):
        with Image.open(before[name]) as first, Image.open(after[name]) as second:
            if first.size != second.size:
                errors.append(f'{name}: размер отличается {first.size} / {second.size}')
                continue
            if second.width >= second.height:
                errors.append(f'{name}: изображение не книжное')
            if second.convert('RGB').getpixel((0,0)) != (255,255,255):
                errors.append(f'{name}: фон в углу не белый')
            if args.strict_pixels and ImageChops.difference(
                first.convert('RGB'),second.convert('RGB')
            ).getbbox() is not None:
                errors.append(f'{name}: несовпадение пикселей')
    print(f'PNG: эталон={len(before)}, после={len(after)}')
    if errors:
        for e in errors: print('ОШИБКА:', e)
        return 1
    print('Контроль пройден: имена, размер, книжная ориентация и белый фон совпадают.'
          + (' Пиксели совпадают.' if args.strict_pixels else ' Пиксели не сравнивались.'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
