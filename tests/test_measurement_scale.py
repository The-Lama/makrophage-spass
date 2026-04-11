from __future__ import annotations

import unittest

from macrophage_analysis.config import (
    MEASUREMENT_SCALE_BACKGROUND_RATIO,
    MEASUREMENT_SCALE_RAW_INTENSITY,
    default_measurement_axis_label,
    default_measurement_summary_label,
    default_measurement_title_label,
    validate_measurement_scale,
)


class MeasurementScaleTests(unittest.TestCase):
    def test_default_labels_match_measurement_scale(self) -> None:
        self.assertEqual(
            default_measurement_axis_label(MEASUREMENT_SCALE_RAW_INTENSITY),
            "Per-cell fluorescence intensity",
        )
        self.assertEqual(
            default_measurement_title_label(MEASUREMENT_SCALE_RAW_INTENSITY),
            "single-cell fluorescence",
        )
        self.assertEqual(
            default_measurement_summary_label(
                MEASUREMENT_SCALE_BACKGROUND_RATIO,
                summary_stat="median",
            ),
            "median signal/background ratio",
        )

    def test_validate_measurement_scale_rejects_unknown_values(self) -> None:
        self.assertEqual(
            validate_measurement_scale(MEASUREMENT_SCALE_BACKGROUND_RATIO),
            MEASUREMENT_SCALE_BACKGROUND_RATIO,
        )
        with self.assertRaises(ValueError):
            validate_measurement_scale("something_else")


if __name__ == "__main__":
    unittest.main()
