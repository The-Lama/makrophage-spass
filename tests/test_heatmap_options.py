from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    np = None
    pd = None

from macrophage_analysis.config import (
    AntibodySpec,
    BatchAnalysis,
    MeasurementResult,
    ResultKey,
    MEASUREMENT_SCALE_RAW_INTENSITY,
)

if np is not None and pd is not None:
    from macrophage_analysis.plotting.fluorescence_summary import plot_summary_heatmap
    from macrophage_analysis.plotting.morphology import plot_morphology_heatmap


@unittest.skipIf(np is None or pd is None, "numpy and pandas are required for these tests")
class HeatmapOptionTests(unittest.TestCase):
    def _build_fluorescence_analysis(self) -> BatchAnalysis:
        def measurement(path: str, values: list[float]) -> MeasurementResult:
            value_array = np.array(values, dtype=float)
            return MeasurementResult(
                path=Path(path),
                cell_count=len(values),
                raw_mean_intensities=value_array,
                measurement_values=value_array,
                measurement_mean=float(np.mean(value_array)),
                measurement_median=float(np.median(value_array)),
            )

        return BatchAnalysis(
            donors=["D45"],
            conditions=["M0", "B68KCP2", "T12CMNepiP4"],
            baseline_condition="M0",
            antibody_order=["CD40"],
            antibody_specs={"CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC")},
            donor_colors={"D45": "#000000"},
            measurement_scale=MEASUREMENT_SCALE_RAW_INTENSITY,
            results={
                ResultKey("CD40", "M0", "D45"): measurement("m0.tif", [2.0, 2.0, 2.0]),
                ResultKey("CD40", "B68KCP2", "D45"): measurement("b68.tif", [4.0, 4.0, 4.0]),
                ResultKey("CD40", "T12CMNepiP4", "D45"): measurement("t12.tif", [8.0, 8.0, 8.0]),
            },
        )

    def _build_fluorescence_stats(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"donor": "D45", "condition": "M0", "antibody": "CD40", "stars": "baseline"},
                {"donor": "D45", "condition": "B68KCP2", "antibody": "CD40", "stars": "**"},
                {"donor": "D45", "condition": "T12CMNepiP4", "antibody": "CD40", "stars": "ns"},
            ]
        )

    def _build_morphology_analysis(self) -> BatchAnalysis:
        def measurement(path: str, intensity_start: float, area_start: float, eccentricity_start: float) -> MeasurementResult:
            intensity_values = np.arange(intensity_start, intensity_start + 8, dtype=float)
            area_values = np.arange(area_start, area_start + 8, dtype=float)
            eccentricity_values = np.arange(eccentricity_start, eccentricity_start + 0.08, 0.01, dtype=float)
            return MeasurementResult(
                path=Path(path),
                cell_count=8,
                raw_mean_intensities=intensity_values,
                measurement_values=intensity_values,
                measurement_mean=float(np.mean(intensity_values)),
                measurement_median=float(np.median(intensity_values)),
                areas=area_values,
                eccentricities=eccentricity_values,
            )

        return BatchAnalysis(
            donors=["D45"],
            conditions=["M0", "B68KCP2", "T12CMNepiP4"],
            baseline_condition="M0",
            antibody_order=["CD206"],
            antibody_specs={"CD206": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "TRITC")},
            donor_colors={"D45": "#000000"},
            measurement_scale=MEASUREMENT_SCALE_RAW_INTENSITY,
            results={
                ResultKey("CD206", "M0", "D45"): measurement("m0.tif", 1.0, 10.0, 0.10),
                ResultKey("CD206", "B68KCP2", "D45"): measurement("b68.tif", 101.0, 110.0, 0.50),
                ResultKey("CD206", "T12CMNepiP4", "D45"): measurement("t12.tif", 201.0, 210.0, 0.70),
            },
        )

    def test_plot_summary_heatmap_can_hide_significance_stars(self) -> None:
        analysis = self._build_fluorescence_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            plot_summary_heatmap(
                analysis,
                stat_results=self._build_fluorescence_stats(),
                show_significance_stars=False,
            )

        annotations = heatmap_mock.call_args.kwargs["annot"]
        self.assertEqual(annotations[1, 0], "1.00")

    def test_plot_summary_heatmap_can_exclude_treatment(self) -> None:
        analysis = self._build_fluorescence_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            matrices = plot_summary_heatmap(
                analysis,
                stat_results=self._build_fluorescence_stats(),
                exclude_treatment="B68KCP2",
            )

        self.assertEqual(matrices["D45"].shape, (2, 1))
        self.assertEqual(
            heatmap_mock.call_args.kwargs["yticklabels"],
            ["M0 (baseline)", "T12 CMN epi P4"],
        )

    def test_plot_summary_heatmap_can_use_descriptive_condition_labels(self) -> None:
        analysis = self._build_fluorescence_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            plot_summary_heatmap(
                analysis,
                condition_label_style="descriptive",
            )

        self.assertEqual(
            heatmap_mock.call_args.kwargs["yticklabels"],
            ["M0 (baseline)", "B68 KCP2", "CMN Melanocytes\nepidermis"],
        )

    def test_plot_morphology_heatmap_can_hide_significance_stars(self) -> None:
        analysis = self._build_morphology_analysis()
        with (
            patch("macrophage_analysis.plotting.morphology.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.morphology.plt.show"),
        ):
            plot_morphology_heatmap(
                analysis,
                per_donor=True,
                show_significance_stars=False,
            )

        annotations = heatmap_mock.call_args.kwargs["annot"]
        self.assertNotIn("\n", annotations[1, 0])

    def test_plot_morphology_heatmap_can_exclude_treatment(self) -> None:
        analysis = self._build_morphology_analysis()
        with (
            patch("macrophage_analysis.plotting.morphology.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.morphology.plt.show"),
        ):
            fold_changes = plot_morphology_heatmap(
                analysis,
                per_donor=True,
                exclude_treatment="B68KCP2",
            )

        self.assertNotIn("B68KCP2", set(fold_changes["condition"].astype(str)))
        self.assertEqual(
            heatmap_mock.call_args.kwargs["yticklabels"],
            ["M0 (baseline)", "T12 CMN epi P4"],
        )

    def test_plot_morphology_heatmap_can_use_descriptive_condition_labels(self) -> None:
        analysis = self._build_morphology_analysis()
        with (
            patch("macrophage_analysis.plotting.morphology.sns.heatmap") as heatmap_mock,
            patch("macrophage_analysis.plotting.morphology.plt.show"),
        ):
            plot_morphology_heatmap(
                analysis,
                per_donor=True,
                condition_label_style="descriptive",
            )

        self.assertEqual(
            heatmap_mock.call_args.kwargs["yticklabels"],
            ["M0 (baseline)", "B68 KCP2", "CMN Melanocytes\nepidermis"],
        )


if __name__ == "__main__":
    unittest.main()
