"""Сравнить численные результаты SHAP и экспорт графики до/после рефакторинга."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops


def compare(before: Path, after: Path, *, strict_pixels: bool = False) -> list[str]:
    errors = []
    for extension in ("csv", "json", "png", "html"):
        left = {p.relative_to(before).as_posix(): p for p in before.rglob(f"*.{extension}")}
        right = {p.relative_to(after).as_posix(): p for p in after.rglob(f"*.{extension}")}
        if set(left) != set(right):
            errors.append(f"{extension}: состав отличается. До: {sorted(left)}. После: {sorted(right)}")
        for name in sorted(set(left) & set(right)):
            first, second = left[name], right[name]
            if extension == "csv":
                a, b = pd.read_csv(first), pd.read_csv(second)
                if list(a.columns) != list(b.columns) or a.shape != b.shape:
                    errors.append(f"{name}: столбцы или размеры различаются")
                    continue
                try:
                    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8, atol=1e-6)
                except AssertionError:
                    errors.append(f"{name}: численные или текстовые значения различаются")
            elif extension == "json":
                if json.loads(first.read_text(encoding="utf-8")) != json.loads(second.read_text(encoding="utf-8")):
                    errors.append(f"{name}: JSON различается")
            elif extension == "png":
                with Image.open(first) as a, Image.open(second) as b:
                    if a.size != b.size:
                        errors.append(f"{name}: размеры различаются: {a.size}, {b.size}")
                        continue
                    if b.width >= b.height:
                        errors.append(f"{name}: нет книжной ориентации")
                    if b.convert("RGB").getpixel((0, 0)) != (255, 255, 255):
                        errors.append(f"{name}: фон не белый")
                    if strict_pixels and ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox():
                        errors.append(f"{name}: пиксели различаются")
            elif extension == "html":
                content = second.read_text(encoding="utf-8")
                if not ("<html" in content.lower() and "shap" in content.lower()):
                    errors.append(f"{name}: не найден полный HTML Force Plot")
        print(f"{extension.upper()}: до {len(left)}, после {len(right)}")
    for name in ("explainability_summary.md",):
        f, g = before / name, after / name
        if f.is_file() and g.is_file() and f.read_bytes() != g.read_bytes():
            errors.append(f"{name}: отчёт различается")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--strict-pixels", action="store_true")
    args = parser.parse_args()
    if not args.baseline.is_dir() or not args.current.is_dir():
        parser.error("Оба каталога результатов должны существовать")
    errors = compare(args.baseline, args.current, strict_pixels=args.strict_pixels)
    for e in errors:
        print("ОШИБКА:", e)
    if errors:
        return 1
    print("Контроль пройден: таблицы SHAP, метаданные и файлы графики согласованы."
          + (" PNG совпадают попиксельно." if args.strict_pixels else " Пиксели не сравнивались."))
    print("HTML проверен по структуре; динамические идентификаторы SHAP не сравнивались побайтово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
