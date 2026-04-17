from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

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
    from macrophage_analysis.plotting.fluorescence_summary import plot_distributions_with_stats


@unittest.skipIf(np is None, "numpy is not installed in this environment")
class FluorescenceSummaryPlottingTests(unittest.TestCase):
    def _build_analysis(self) -> BatchAnalysis:
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
            conditions=["M0", "B68KCP2"],
            baseline_condition="M0",
            antibody_order=["CD40"],
            antibody_specs={"CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC")},
            donor_colors={"D45": "#000000"},
            measurement_scale=MEASUREMENT_SCALE_RAW_INTENSITY,
            results={
                ResultKey("CD40", "M0", "D45"): measurement("m0.tif", [1.0, 2.0, 3.0]),
                ResultKey("CD40", "B68KCP2", "D45"): measurement("treated.tif", [4.0, 5.0, 6.0]),
            },
        )

    def test_plot_distributions_with_stats_uses_violin_plot_by_default(self) -> None:
        analysis = self._build_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.violinplot") as violin_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.barplot") as bar_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.stripplot") as strip_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            plot_distributions_with_stats(analysis)

        self.assertTrue(violin_mock.called)
        self.assertFalse(bar_mock.called)
        self.assertTrue(strip_mock.called)

    def test_plot_distributions_with_stats_can_render_bar_plots_without_points_by_default(self) -> None:
        analysis = self._build_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.violinplot") as violin_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.barplot") as bar_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.stripplot") as strip_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            plot_distributions_with_stats(
                analysis,
                plot_kind="bar",
                summary_stat="mean",
            )

        self.assertFalse(violin_mock.called)
        self.assertTrue(bar_mock.called)
        self.assertFalse(strip_mock.called)
        self.assertIs(bar_mock.call_args.kwargs["estimator"], np.mean)

    def test_plot_distributions_with_stats_can_show_points_for_bar_plots_when_requested(self) -> None:
        analysis = self._build_analysis()
        with (
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.barplot"),
            patch("macrophage_analysis.plotting.fluorescence_summary.sns.stripplot") as strip_mock,
            patch("macrophage_analysis.plotting.fluorescence_summary.plt.show"),
        ):
            plot_distributions_with_stats(
                analysis,
                plot_kind="bar",
                show_points=True,
            )

        self.assertTrue(strip_mock.called)

    def test_plot_distributions_with_stats_rejects_unknown_plot_kind(self) -> None:
        analysis = self._build_analysis()
        with self.assertRaises(ValueError):
            plot_distributions_with_stats(analysis, plot_kind="scatter")

    def test_plot_distributions_with_stats_rejects_unknown_summary_stat(self) -> None:
        analysis = self._build_analysis()
        with self.assertRaisesRegex(ValueError, "summary_stat"):
            plot_distributions_with_stats(analysis, plot_kind="bar", summary_stat="mode")


if __name__ == "__main__":
    unittest.main()
