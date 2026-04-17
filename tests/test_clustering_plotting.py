from __future__ import annotations

import unittest

try:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
except ImportError:  # pragma: no cover - lightweight shell
    pd = None

if pd is not None:
    import numpy as np

    from macrophage_analysis.plotting.clustering import (
        plot_aligned_cluster_composition,
        plot_cluster_feature_pairs,
        plot_kmeans_state_grid,
    )


@unittest.skipIf(pd is None, "plotting dependencies are not installed in this environment")
class ClusteringPlottingTests(unittest.TestCase):
    def test_plot_kmeans_state_grid_returns_cluster_count_lines(self) -> None:
        donor_feature_data = {
            "Donor D45": {
                "display_features": np.asarray([[1.0, 10.0], [2.0, 11.0]], dtype=float),
                "feature_mean": np.asarray([0.0, 0.0], dtype=float),
                "feature_std": np.asarray([1.0, 1.0], dtype=float),
            },
            "Donor D47": {
                "display_features": np.asarray([[1.5, 10.5], [2.5, 11.5]], dtype=float),
                "feature_mean": np.asarray([0.0, 0.0], dtype=float),
                "feature_std": np.asarray([1.0, 1.0], dtype=float),
            },
        }
        state_by_donor = {
            "Donor D45": {
                "iteration": 1,
                "phase": "assignment",
                "centroids": np.asarray([[1.0, 10.0], [2.0, 11.0]], dtype=float),
                "assignments": np.asarray([0, 1], dtype=int),
                "shift": np.nan,
            },
            "Donor D47": {
                "iteration": 1,
                "phase": "assignment",
                "centroids": np.asarray([[1.5, 10.5], [2.5, 11.5]], dtype=float),
                "assignments": np.asarray([0, 1], dtype=int),
                "shift": np.nan,
            },
        }

        count_lines = plot_kmeans_state_grid(
            donor_feature_data,
            state_by_donor,
            antibody="CD206",
            k=2,
        )

        self.assertEqual(
            count_lines,
            [
                "Donor D45: Cluster 1=1, Cluster 2=1",
                "Donor D47: Cluster 1=1, Cluster 2=1",
            ],
        )

    def test_plot_aligned_cluster_composition_accepts_valid_table(self) -> None:
        composition = pd.DataFrame(
            {
                "donor_label": pd.Categorical(
                    ["Donor D45", "Donor D45", "Donor D47", "Donor D47"],
                    categories=["Donor D45", "Donor D47"],
                    ordered=True,
                ),
                "condition_label": pd.Categorical(
                    ["M0 (baseline)", "Treatment", "M0 (baseline)", "Treatment"],
                    categories=["M0 (baseline)", "Treatment"],
                    ordered=True,
                ),
                "aligned_cluster": pd.Categorical(
                    ["Cluster 1", "Cluster 1", "Cluster 1", "Cluster 1"],
                    categories=["Cluster 1"],
                    ordered=True,
                ),
                "cell_count": [6, 6, 6, 6],
                "fraction": [1.0, 1.0, 1.0, 1.0],
            }
        )

        returned = plot_aligned_cluster_composition(
            composition,
            reference_donor_label="Donor D45",
            cluster_count=1,
        )

        self.assertTrue(returned.equals(composition))

    def test_plot_aligned_cluster_composition_requires_fraction_column(self) -> None:
        composition = pd.DataFrame(
            {
                "donor_label": ["Donor D45"],
                "condition_label": ["M0 (baseline)"],
                "aligned_cluster": ["Cluster 1"],
            }
        )

        with self.assertRaises(ValueError):
            plot_aligned_cluster_composition(
                composition,
                reference_donor_label="Donor D45",
            )

    def test_plot_cluster_feature_pairs_accepts_three_feature_table(self) -> None:
        clustered_cells = pd.DataFrame(
            {
                "donor_label": pd.Categorical(
                    ["Donor D45", "Donor D45", "Donor D47", "Donor D47"],
                    categories=["Donor D45", "Donor D47"],
                    ordered=True,
                ),
                "aligned_cluster": pd.Categorical(
                    ["Cluster 1", "Cluster 2", "Cluster 1", "Cluster 2"],
                    categories=["Cluster 1", "Cluster 2"],
                    ordered=True,
                ),
                "intensity": [1.0, 2.0, 1.5, 2.5],
                "area": [10.0, 12.0, 10.5, 12.5],
                "eccentricity": [0.8, 0.2, 0.75, 0.25],
            }
        )

        returned = plot_cluster_feature_pairs(
            clustered_cells,
            feature_columns=("intensity", "area", "eccentricity"),
            feature_labels={
                "intensity": "CD206 brightness",
                "area": "Area",
                "eccentricity": "Eccentricity",
            },
            max_points_per_donor=10,
        )

        self.assertEqual(len(returned), 4)

    def test_plot_cluster_feature_pairs_requires_two_features(self) -> None:
        clustered_cells = pd.DataFrame(
            {
                "donor_label": ["Donor D45"],
                "aligned_cluster": ["Cluster 1"],
                "intensity": [1.0],
            }
        )

        with self.assertRaises(ValueError):
            plot_cluster_feature_pairs(
                clustered_cells,
                feature_columns=("intensity",),
            )


if __name__ == "__main__":
    unittest.main()
