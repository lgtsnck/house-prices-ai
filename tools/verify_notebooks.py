"""Проверяет шесть Notebook проекта без перезаписи пользовательских результатов.

Примеры:
  python tools/verify_notebooks.py
  python tools/verify_notebooks.py --execute --only 03 04 05 06
  python tools/verify_notebooks.py --execute

Для выполнения необходим nbclient (обычно установлен вместе с JupyterLab).
"""
from __future__ import annotations
import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / 'notebooks'
PERCENT_RE = re.compile(r'^#\s*%{2,4}(?:\s*\[markdown\])?.*$', re.M)


def check_one(pyfile: Path, notebook: Path) -> dict:
    import nbformat

    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    source = pyfile.read_text(encoding='utf-8')
    markers = PERCENT_RE.findall(source)
    if len(markers) != len(nb.cells):
        raise AssertionError(
            f'{pyfile.name}: {len(markers)} ячеек в py, {len(nb.cells)} в ipynb'
        )
    expected_types = ['markdown' if '[markdown]' in line else 'code' for line in markers]
    actual_types = [cell.cell_type for cell in nb.cells]
    if expected_types != actual_types:
        raise AssertionError(f'{notebook.name}: типы ячеек расходятся с парным .py')
    for i, cell in enumerate(nb.cells, 1):
        if cell.cell_type == 'code':
            try:
                ast.parse(cell.source)
            except SyntaxError as exc:
                raise AssertionError(f'{notebook.name}: синтаксис ячейки {i}: {exc}') from exc
    return {'cells': len(nb.cells), 'code_cells': actual_types.count('code')}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='выполнить кодовые ячейки; может занять время')
    parser.add_argument('--only', nargs='+', default=None, help='номера Notebook: 01 02 03 04 05 06')
    args = parser.parse_args(argv)
    if args.execute:
        from nbclient import NotebookClient
        import nbformat

    results = []
    for nb in sorted(NOTEBOOKS.glob('[0-9][0-9]_*.ipynb')):
        if args.only and nb.name[:2] not in args.only:
            continue
        py = nb.with_suffix('.py')
        if not py.exists():
            raise FileNotFoundError(f'Отсутствует парный Jupytext-файл: {py}')
        counts = check_one(py, nb)
        print(f'{nb.name}: разметка OK, ячеек {counts["cells"]}, кодовых {counts["code_cells"]}', flush=True)
        if args.execute:
            notebook = nbformat.read(nb, as_version=4)
            client = NotebookClient(
                notebook, timeout=600, kernel_name='python3',
                resources={'metadata': {'path': str(NOTEBOOKS)}}, allow_errors=False,
            )
            # Выполняет в памяти; существующий .ipynb не перезаписывается.
            client.execute()
            print(f'  Выполнение ячеек OK', flush=True)
        results.append(nb.name)
    if not results:
        parser.error('Не найдено Notebook для проверки')
    print(f'Проверено Notebook: {len(results)}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'Ошибка проверки: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
