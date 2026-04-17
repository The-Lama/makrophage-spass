from __future__ import annotations

import unittest

from macrophage_analysis.config import (
    MEASUREMENT_SCALE_BACKGROUND_RATIO,
    MEASUREMENT_SCALE_RAW_INTENSITY,
    MeasurementResult,
)
from macrophage_analysis.labels import (
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

    def test_measurement_result_keeps_raw_and_transformed_values_separate(self) -> None:
        measurement = MeasurementResult(
            path="example.tif",  # type: ignore[arg-type]
            cell_count=2,
            raw_mean_intensities=[10.0, 20.0],  # type: ignore[list-item]
            measurement_values=[2.0, 4.0],  # type: ignore[list-item]
            measurement_mean=3.0,
            measurement_median=3.0,
            measurement_scale=MEASUREMENT_SCALE_BACKGROUND_RATIO,
            background_intensity=5.0,
            background_percentile=50.0,
        )
        self.assertEqual(measurement.raw_mean_intensities, [10.0, 20.0])
        self.assertEqual(measurement.measurement_values, [2.0, 4.0])
        self.assertEqual(measurement.mean_intensities, [2.0, 4.0])
        self.assertEqual(measurement.overall_mean, 3.0)
        self.assertEqual(measurement.overall_median, 3.0)


if __name__ == "__main__":
    unittest.main()
