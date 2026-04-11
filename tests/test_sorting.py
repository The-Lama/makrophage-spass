from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from macrophage_analysis.io.sorting import (
    parse_donor_condition_from_filename,
    sort_microscopy_images,
)


class SortingTests(unittest.TestCase):
    def test_parse_donor_condition_from_filename(self) -> None:
        self.assertEqual(
            parse_donor_condition_from_filename("D45_B68KCP2_example.tif"),
            ("D45", "B68KCP2"),
        )
        self.assertIsNone(parse_donor_condition_from_filename("badname.tif"))

    def test_sort_microscopy_images_moves_tiffs_into_nested_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "incoming"
            dest_dir = Path(tmpdir) / "raw_data"
            source_dir.mkdir(parents=True, exist_ok=True)
            valid_tif = source_dir / "D45_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tif"
            valid_tiff = source_dir / "D47_B68KCP2_CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC.tiff"
            invalid = source_dir / "invalid_name.tif"
            valid_tif.touch()
            valid_tiff.touch()
            invalid.touch()

            sort_microscopy_images(source_dir, dest_dir)

            self.assertFalse(valid_tif.exists())
            self.assertFalse(valid_tiff.exists())
            self.assertTrue(invalid.exists())
            self.assertTrue(
                (dest_dir / "D45" / "M0" / valid_tif.name).exists()
            )
            self.assertTrue(
                (dest_dir / "D47" / "B68KCP2" / valid_tiff.name).exists()
            )


if __name__ == "__main__":
    unittest.main()
