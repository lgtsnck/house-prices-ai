"""Начальные проверки структуры проекта (без внешних библиотек)."""

from contextlib import redirect_stdout
from io import StringIO
import unittest

from house_prices_ai.config import PROJECT_ROOT, RAW_TRAIN_FILE
from main import STAGES, main


class ProjectSkeletonTests(unittest.TestCase):
    def test_root_contains_main(self) -> None:
        self.assertTrue((PROJECT_ROOT / "main.py").is_file())

    def test_data_location_is_inside_repo(self) -> None:
        self.assertEqual(RAW_TRAIN_FILE, PROJECT_ROOT / "data" / "raw" / "train.csv")

    def test_six_laboratories_are_planned(self) -> None:
        self.assertEqual(len(STAGES), 6)

    def test_plan_command_works(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["plan"])
        self.assertEqual(exit_code, 0)
        self.assertIn("eda", output.getvalue())
        self.assertIn("clustering", output.getvalue())
        self.assertIn("Разведочный анализ", output.getvalue())

    def test_info_command_uses_russian_descriptions(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["info"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Каталог проекта", output.getvalue())
        self.assertIn("Ожидаемый файл", output.getvalue())


if __name__ == "__main__":
    unittest.main()
