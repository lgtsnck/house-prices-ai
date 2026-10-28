"""Проверки предобработки, исключения утечек и воспроизводимости."""

import tempfile
import warnings
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from house_prices_ai.preprocessing import (
    CLASS_THRESHOLD, HouseFeatures, IQRClipper, RecalculateTotalUsefulArea, run_preprocessing,
    _evaluate_vif, VIF_REDUCED_EXCLUDE,
)


def make_sample(n=50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Id": range(1, n + 1),
        "SalePrice": [95_000, 125_000, 155_000, 220_000, 300_000] * (n // 5),
        "LotArea": rng.integers(1800, 10000, n),
        "LotFrontage": rng.uniform(50, 100, n),
        "OverallQual": rng.integers(3, 10, n),
        "GrLivArea": rng.integers(800, 2900, n),
        "TotalBsmtSF": rng.integers(300, 1100, n),
        "1stFlrSF": rng.integers(500, 1400, n),
        "2ndFlrSF": rng.integers(0, 800, n),
        "GarageArea": rng.integers(0, 700, n),
        "GarageCars": rng.integers(0, 4, n),
        "YearBuilt": rng.integers(1930, 2010, n),
        "YearRemodAdd": rng.integers(1950, 2010, n),
        "YrSold": rng.integers(2006, 2011, n),
        "FullBath": rng.integers(1, 3, n),
        "HalfBath": rng.integers(0, 2, n),
        "BsmtFullBath": rng.integers(0, 2, n),
        "BsmtHalfBath": rng.integers(0, 2, n),
        "MSSubClass": rng.choice([20, 50, 60], n),
        "Neighborhood": rng.choice(["NAmes", "Crawfor"], n),
        "GarageType": rng.choice(["Attchd", "Detchd", "None"], n),
    })


class PreprocessingTests(unittest.TestCase):
    def test_feature_engineering_does_not_change_raw_data(self):
        raw = make_sample(10).drop(columns=["SalePrice", "Id"])
        initial = raw.copy(deep=True)
        engineered = HouseFeatures().fit_transform(raw)
        self.assertEqual(raw.equals(initial), True)
        self.assertIn("TotalUsefulSF", engineered)
        self.assertIn("PropertyAgeAtSale", engineered)
        self.assertIn("MSSubClass", engineered)

    def test_target_and_id_do_not_pass_to_features(self):
        sample = make_sample(10)
        with self.assertRaisesRegex(ValueError, "SalePrice|Id"):
            HouseFeatures().transform(sample)

    def test_iqr_limits_are_fit_only_once(self):
        c = IQRClipper(numeric_columns=("GrLivArea",))
        c.fit(np.array([[1000.0], [1010.0], [1020.0], [1040.0]]))
        upper = c.upper_[0]
        self.assertTrue(upper < 2000)
        result = c.transform(np.array([[1_000_000.0]]))
        self.assertAlmostEqual(result[0, 0], upper)

    def test_total_useful_area_matches_clipped_components(self):
        columns = ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "TotalUsefulSF")
        original = np.array([[200.0, 500.0, 0.0, 700.0],
                             [400.0, 700.0, 250.0, 1350.0],
                             [600.0, 900.0, 400.0, 1900.0],
                             [10000.0, 12000.0, 9000.0, 31000.0]])
        clip = IQRClipper(numeric_columns=columns)
        clip.fit(original[:3])
        trimmed = clip.transform(original)
        result = RecalculateTotalUsefulArea(columns).transform(trimmed)
        np.testing.assert_allclose(result[:, 3], result[:, :3].sum(axis=1))
        self.assertLess(result[-1, 3], original[-1, 3])
        self.assertEqual(result.shape, original.shape)

    def test_outlier_statistics_include_only_actual_changes(self):
        sample = make_sample(50)
        sample.loc[0, "GrLivArea"] = 1_000_000
        sample.loc[1, "LotArea"] = 1_000_000
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "train.csv"
            sample.to_csv(path, index=False)
            run_preprocessing(path, base / "output", base / "processed",
                              make_plots=False, run_diagnostics=False)
            report = pd.read_csv(base / "output" / "outlier_changes.csv")
            self.assertEqual(set(report["Выборка"]), {"train", "test"})
            self.assertEqual(set(report["Признак"]), set((
                "LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea", "1stFlrSF", "2ndFlrSF"
            )))
            self.assertEqual(len(report), 12)
            self.assertTrue((report["Всего изменено"] ==
                             report["Ниже границы"] + report["Выше границы"]).all())
            self.assertGreater(int(report["Всего изменено"].sum()), 0)
            self.assertTrue((report["Доля изменённых, %"] <= 100).all())
            # Для train/test применены одни и те же границы, вычисленные по train.
            for field in ("Нижняя граница", "Верхняя граница"):
                a = report[report["Выборка"] == "train"].set_index("Признак")[field]
                b = report[report["Выборка"] == "test"].set_index("Признак")[field]
                pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), check_names=False)

    def test_end_to_end_split_and_exports(self):
        sample = make_sample(50)
        sample.loc[0, "LotFrontage"] = np.nan
        sample.loc[3, "GarageType"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_file = base / "train.csv"
            sample.to_csv(data_file, index=False)
            result = run_preprocessing(
                data_file, base / "output", base / "processed",
                make_plots=False, run_diagnostics=False,
            )
            self.assertEqual((result.train_rows, result.test_rows), (40, 10))
            train = pd.read_csv(base / "processed" / "X_train_reg.csv")
            test = pd.read_csv(base / "processed" / "X_test_reg.csv")
            self.assertEqual(train.columns.tolist(), test.columns.tolist())
            self.assertEqual(len(train), 40)
            self.assertEqual(len(test), 10)
            self.assertEqual(int(train.isna().sum().sum()), 0)
            self.assertFalse(any("SalePrice" in c or c.endswith("Id") for c in train.columns))
            self.assertEqual(result.class_balance, {0: 30, 1: 20})
            self.assertEqual(len(result.selected_kbest), min(20, len(train.columns)))
            train_ids = pd.read_csv(base / "processed" / "train_ids.csv")["Id"]
            test_ids = pd.read_csv(base / "processed" / "test_ids.csv")["Id"]
            self.assertEqual(len(set(train_ids).intersection(test_ids)), 0)
            loaded = joblib.load(base / "output" / "preprocessing_pipeline.joblib")
            fresh = make_sample(10).drop(columns=["SalePrice", "Id"])
            self.assertEqual(loaded.transform(fresh).shape[1], train.shape[1])
            self.assertTrue((base / "output" / "preprocessing_summary.md").exists())
            assert (base / "output" / "outlier_changes.csv").exists()
            # Проверяем данные до StandardScaler: рассчитанный суммарный признак
            # обязан совпадать с суммой уже ограниченных площадей.
            numerics = loaded.named_steps["encode"].named_transformers_["num"]
            imputed = numerics.named_steps["impute"].transform(
                loaded.named_steps["features"].transform(fresh)[list(numerics.named_steps["clip"].numeric_columns)]
            )
            clipped = numerics.named_steps["clip"].transform(imputed)
            reconciled = numerics.named_steps["reconcile_area"].transform(clipped)
            col_names = list(numerics.named_steps["clip"].numeric_columns)
            ti = col_names.index("TotalUsefulSF")
            indices = [col_names.index(c) for c in ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF")]
            np.testing.assert_allclose(reconciled[:, ti], reconciled[:, indices].sum(axis=1))



    def test_full_and_reduced_vif_are_distinguished(self):
        rng = np.random.default_rng(123)
        values = pd.DataFrame({"num__" + col: rng.normal(size=150) for col in (
            "OverallQual", "GarageCars", "GarageArea", "YearBuilt", "YearRemodAdd",
            "LotArea", "TotalBathrooms", "GrLivArea", "TotalBsmtSF", "1stFlrSF",
            "2ndFlrSF", "PropertyAgeAtSale", "YearsSinceRemodel",
        )})
        values["num__TotalUsefulSF"] = (values["num__TotalBsmtSF"] +
                                         values["num__1stFlrSF"] + values["num__2ndFlrSF"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            full, full_info = _evaluate_vif(values)
            reduced, reduced_info = _evaluate_vif(values, exclude=VIF_REDUCED_EXCLUDE)
        self.assertFalse(caught, [str(w.message) for w in caught])
        self.assertEqual(full_info["Признаков"], 14)
        self.assertEqual(full_info["Ранг"], 13)
        self.assertTrue(full_info["Линейная зависимость"])
        self.assertEqual(int(np.isinf(full["VIF"]).sum()), 4)
        self.assertEqual(reduced_info["Признаков"], 11)
        self.assertEqual(reduced_info["Ранг"], 11)
        self.assertFalse(reduced_info["Линейная зависимость"])
        self.assertTrue(np.isfinite(reduced["VIF"]).all())

    def test_print_ready_plots_and_vif_exports(self):
        sample = make_sample(50)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "sample.csv"
            sample.to_csv(path, index=False)
            run_preprocessing(path, base / "result", base / "processed",
                              make_plots=True, run_diagnostics=True)
            for name in ("vif.csv", "vif_reduced.csv", "vif_diagnostics.csv"):
                self.assertTrue((base / "result" / name).is_file())
            from matplotlib.image import imread
            for name in ("01_missing_values.png", "02_feature_selection.png",
                         "03_pca_preprocessed.png"):
                rgba = imread(base / "result" / name)
                self.assertGreater(rgba.shape[0], rgba.shape[1])
                self.assertTrue(np.allclose(rgba[0, 0, :3], 1.0))
                self.assertTrue(np.allclose(rgba[-1, -1, :3], 1.0))


if __name__ == "__main__":
    unittest.main()
