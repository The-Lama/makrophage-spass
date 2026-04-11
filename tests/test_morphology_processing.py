from __future__ import annotations

import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - lightweight shell
    pd = None

if pd is not None:
    from macrophage_analysis.morphology_processing import (
        filter_morphology_points,
        prepare_morphology_scatter_points,
        sample_morphology_points,
    )


@unittest.skipIf(pd is None, "pandas is not installed in this environment")
class MorphologyProcessingTests(unittest.TestCase):
    def _build_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "condition": pd.Categorical(
                    ["M0", "M0", "M0", "B68KCP2", "B68KCP2", "B68KCP2"],
                    categories=["M0", "B68KCP2"],
                    ordered=True,
                ),
                "donor": pd.Categorical(
                    ["D45", "D45", "D45", "D45", "D45", "D45"],
                    categories=["D45"],
                    ordered=True,
                ),
                "donor_label": pd.Categorical(
                    ["Donor D45"] * 6,
                    categories=["Donor D45"],
                    ordered=True,
                ),
                "condition_label": pd.Categorical(
                    ["M0", "M0", "M0", "B68KCP2", "B68KCP2", "B68KCP2"],
                    categories=["M0", "B68KCP2"],
                    ordered=True,
                ),
                "intensity": [1.0, 2.0, 100.0, 3.0, 4.0, 5.0],
                "area": [10.0, 11.0, 300.0, 12.0, 13.0, 14.0],
                "eccentricity": [0.1, 0.2, 0.9, 0.3, 0.4, 0.5],
            }
        )

    def test_sample_morphology_points_limits_rows_per_group(self) -> None:
        sampled = sample_morphology_points(
            self._build_df(),
            max_points_per_group=2,
            random_seed=7,
        )
        counts = sampled.groupby(["condition", "donor"], observed=True).size().to_dict()
        self.assertEqual(counts[("M0", "D45")], 2)
        self.assertEqual(counts[("B68KCP2", "D45")], 2)

    def test_filter_morphology_points_applies_percentile_cutoffs(self) -> None:
        filtered = filter_morphology_points(
            self._build_df(),
            x_column="intensity",
            x_upper_percentile=80.0,
            area_upper_percentile=80.0,
        )
        self.assertFalse(((filtered["intensity"] == 100.0) | (filtered["area"] == 300.0)).any())

    def test_prepare_morphology_scatter_points_combines_sampling_and_filtering(self) -> None:
        prepared = prepare_morphology_scatter_points(
            self._build_df(),
            max_points_per_group=2,
            random_seed=7,
            x_column="intensity",
            x_upper_percentile=100.0,
            area_upper_percentile=100.0,
        )
        self.assertEqual(len(prepared), 4)


if __name__ == "__main__":
    unittest.main()
