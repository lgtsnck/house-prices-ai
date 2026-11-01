"""Сравнение эталона и результатов кластеризации с учётом округления float.

Использование:
    python compare_clustering_results_fixed.py outputs/clustering_baseline outputs/clustering_refactored --strict-pixels

Числа с плавающей точкой сравниваются с rtol=atol=1e-12;
целочисленные и текстовые поля, структура таблиц и Markdown — строго.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype


RTOL = 1e-12
ATOL = 1e-12


def _files(folder: Path, pattern: str) -> dict[Path, Path]:
    return {p.relative_to(folder): p for p in folder.rglob(pattern)}


def _compare_csv(before: Path, after: Path, relative: Path) -> int:
    a = pd.read_csv(before, float_precision="round_trip")
    b = pd.read_csv(after, float_precision="round_trip")
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise AssertionError(f"Структура CSV изменилась: {relative}: {a.shape} / {b.shape}")
    approximate = 0
    for column in a.columns:
        va, vb = a[column], b[column]
        if not va.isna().equals(vb.isna()):
            raise AssertionError(f"Изменилось расположение пропусков: {relative}, {column}")
        if is_numeric_dtype(va) and is_numeric_dtype(vb):
            if is_integer_dtype(va) and is_integer_dtype(vb):
                if not va.equals(vb):
                    raise AssertionError(f"Изменились целочисленные значения: {relative}, {column}")
            else:
                x = va.to_numpy(dtype=float)
                y = vb.to_numpy(dtype=float)
                mask = np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True)
                if not mask.all():
                    i = int(np.flatnonzero(~mask)[0])
                    raise AssertionError(
                        f"Значимое различие: {relative}, столбец {column}, строка {i + 2}: "
                        f"{x[i]!r} / {y[i]!r}"
                    )
                if not np.array_equal(x, y, equal_nan=True):
                    approximate += 1
        elif not va.equals(vb):
            raise AssertionError(f"Изменились текстовые значения: {relative}, {column}")
    return approximate


def compare(baseline: Path, refactored: Path, strict_pixels: bool = False) -> None:
    for folder in (baseline, refactored):
        if not folder.is_dir():
            raise FileNotFoundError(f"Каталог результатов не найден: {folder}")

    before_csv, after_csv = _files(baseline, "*.csv"), _files(refactored, "*.csv")
    if set(before_csv) != set(after_csv):
        raise AssertionError(f"Состав CSV различается: {set(before_csv) ^ set(after_csv)}")
    adjusted = 0
    for relative in sorted(before_csv):
        adjusted += _compare_csv(before_csv[relative], after_csv[relative], relative)
    print(f"CSV: {len(before_csv)} файлов, данные совпадают (вещественные числа: допуск {RTOL:g})")
    if adjusted:
        print(f"  {adjusted} столбцов содержат лишь допустимые различия последних знаков float")

    before_md, after_md = _files(baseline, "*.md"), _files(refactored, "*.md")
    if set(before_md) != set(after_md):
        raise AssertionError(f"Состав Markdown различается: {set(before_md) ^ set(after_md)}")
    for relative in sorted(before_md):
        if before_md[relative].read_bytes() != after_md[relative].read_bytes():
            raise AssertionError(f"Markdown изменился: {relative}")
    print(f"Markdown: {len(before_md)} файлов совпадают побайтово")

    before_png, after_png = _files(baseline, "*.png"), _files(refactored, "*.png")
    if set(before_png) != set(after_png):
        raise AssertionError(f"Состав PNG различается: {set(before_png) ^ set(after_png)}")
    for relative in sorted(before_png):
        with Image.open(before_png[relative]) as a, Image.open(after_png[relative]) as b:
            if a.size != b.size or a.size[1] <= a.size[0]:
                raise AssertionError(f"Различаются размеры или ориентация: {relative}: {a.size} / {b.size}")
            if strict_pixels and ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox():
                raise AssertionError(f"Различаются пиксели: {relative}")
    suffix = "и пиксели" if strict_pixels else "(пиксели не сравнивались)"
    print(f"PNG: {len(before_png)} изображений, имена, размеры, ориентация {suffix} — OK")
    print("Контроль кластеризации пройден.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("refactored", type=Path)
    parser.add_argument("--strict-pixels", action="store_true")
    args = parser.parse_args()
    compare(args.baseline, args.refactored, args.strict_pixels)
