from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    np = None

from macrophage_analysis.models import (
    AntibodySpec,
    BatchAnalysis,
    MeasurementResult,
    ResultKey,
    MEASUREMENT_SCALE_RAW_INTENSITY,
)

if np is not None:
    from macrophage_analysis.analysis.fluorescence import build_plot_frames, group_plot_values


@unittest.skipIf(np is None, "numpy is not installed in this environment")
class FluorescenceProcessingTests(unittest.TestCase):
    def _build_analysis(self) -> BatchAnalysis:
        return BatchAnalysis(
            donors=["D45"],
            conditions=["M0", "B68KCP2"],
            baseline_condition="M0",
            antibody_order=["CD40"],
            antibody_specs={"CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC")},
            donor_colors={"D45": "#000000"},
            measurement_scale=MEASUREMENT_SCALE_RAW_INTENSITY,
            results={
                ResultKey("CD40", "M0", "D45"): MeasurementResult(
                    path="m0.tif",  # type: ignore[arg-type]
                    cell_count=4,
                    raw_mean_intensities=np.array([0.0, 1.0, np.nan, 4.0]),
                    measurement_values=np.array([0.0, 1.0, np.nan, 4.0]),
                    measurement_mean=1.6666666667,
                    measurement_median=1.0,
                ),
                ResultKey("CD40", "B68KCP2", "D45"): MeasurementResult(
                    path="treated.tif",  # type: ignore[arg-type]
                    cell_count=3,
                    raw_mean_intensities=np.array([2.0, 6.0, 8.0]),
                    measurement_values=np.array([2.0, 6.0, 8.0]),
                    measurement_mean=16.0 / 3.0,
                    measurement_median=6.0,
                ),
            },
        )

    def test_group_plot_values_applies_log_filtering_and_clipping(self) -> None:
        analysis = self._build_analysis()
        grouped = group_plot_values(
            analysis,
            "CD40",
            plot_log_scale=True,
            min_positive_intensity=2.0,
        )
        np.testing.assert_allclose(grouped[("M0", "D45")], np.array([2.0, 4.0]))
        np.testing.assert_allclose(grouped[("B68KCP2", "D45")], np.array([2.0, 6.0, 8.0]))

    def test_build_plot_frames_preserves_expected_categories(self) -> None:
        analysis = self._build_analysis()
        full_df, sample_df, grouped = build_plot_frames(
            analysis,
            "CD40",
            plot_log_scale=False,
            min_positive_intensity=None,
            max_strip_points=1,
            random_seed=7,
        )
        self.assertEqual(list(full_df["condition"].cat.categories), ["M0", "B68KCP2"])
        self.assertEqual(list(full_df["donor"].cat.categories), ["D45"])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(sample_df["condition"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
