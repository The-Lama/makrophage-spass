from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from macrophage_analysis.io.catalog import build_image_catalog


class ImageCatalogTests(unittest.TestCase):
    def test_builds_catalog_and_finds_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "raw_data"
            path_a = data_root / "D45" / "M0" / "D45_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tif"
            path_b = data_root / "D47" / "B68KCP2" / "D47_B68KCP2_CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC.tiff"
            path_a.parent.mkdir(parents=True, exist_ok=True)
            path_b.parent.mkdir(parents=True, exist_ok=True)
            path_a.touch()
            path_b.touch()

            catalog = build_image_catalog(data_root)

            self.assertEqual(len(catalog), 2)
            self.assertEqual(
                catalog.find_path("D45", "M0", "CD40_A488_10x_RGB_Qi2 10x", "FITC"),
                path_a.resolve(),
            )
            self.assertEqual(
                catalog.find_path(
                    "D47",
                    "B68KCP2",
                    "CD200R_A488_CD206_A568_10x_RGB_Qi2 10x",
                    "TRITC",
                ),
                path_b.resolve(),
            )

    def test_duplicate_measurements_raise_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "raw_data"
            dup_a = data_root / "D45" / "M0" / "D45_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tif"
            dup_b = data_root / "D45" / "M0" / "D45_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tiff"
            dup_a.parent.mkdir(parents=True, exist_ok=True)
            dup_a.touch()
            dup_b.touch()

            with self.assertRaises(ValueError):
                build_image_catalog(data_root)


if __name__ == "__main__":
    unittest.main()
