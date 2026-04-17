from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - lightweight shell
    np = None
    pd = None

if np is not None and pd is not None:
    from macrophage_analysis.analysis.clustering import (
        build_aligned_cluster_composition_table,
        build_donor_feature_data,
        run_kmeans_with_history,
        run_split_kmeans,
    )


@unittest.skipIf(np is None or pd is None, "numpy/pandas are not installed in this environment")
class ClusteringAnalysisTests(unittest.TestCase):
    def _build_cells(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add_cluster(
            donor: str,
            donor_label: str,
            condition: str,
            condition_label: str,
            intensity_start: float,
            area_start: float,
        ) -> None:
            for index in range(6):
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": condition_label,
                        "donor": donor,
                        "donor_label": donor_label,
                        "intensity": intensity_start + 0.2 * index,
                        "area": area_start + 0.4 * index,
                    }
                )

        add_cluster("D45", "Donor D45", "M0", "M0 (baseline)", 1.0, 10.0)
        add_cluster("D45", "Donor D45", "Treat", "Treatment", 9.0, 90.0)

        # Reverse the treatment order so donor-local raw cluster numbering can differ.
        add_cluster("D47", "Donor D47", "Treat", "Treatment", 9.0, 90.0)
        add_cluster("D47", "Donor D47", "M0", "M0 (baseline)", 1.0, 10.0)

        return pd.DataFrame(rows).astype(
            {
                "condition": "category",
                "condition_label": "category",
                "donor": "category",
                "donor_label": "category",
            }
        )

    def test_build_donor_feature_data_splits_cells_per_donor(self) -> None:
        donor_feature_data = build_donor_feature_data(self._build_cells())

        self.assertEqual(list(donor_feature_data), ["Donor D45", "Donor D47"])
        self.assertEqual(donor_feature_data["Donor D45"]["display_features"].shape, (12, 2))
        self.assertEqual(donor_feature_data["Donor D47"]["display_features"].shape, (12, 2))

    def test_run_kmeans_with_history_tracks_assignment_and_update_steps(self) -> None:
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [10.0, 10.0],
                [10.2, 10.1],
            ],
            dtype=float,
        )

        centroids, assignments, history = run_kmeans_with_history(
            features,
            k=2,
            random_seed=7,
            max_iterations=10,
        )

        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(assignments.shape, (4,))
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[0]["phase"], "assignment")
        self.assertEqual(history[1]["phase"], "update")

    def test_run_split_kmeans_aligns_non_reference_donor_by_centroid_distance(self) -> None:
        _, donor_runs, clustered_cells, centroid_table = run_split_kmeans(
            self._build_cells(),
            k=2,
            random_seed=3,
            max_iterations=12,
            reference_donor_label="Donor D45",
        )

        self.assertEqual(set(donor_runs), {"Donor D45", "Donor D47"})
        self.assertIn("aligned_cluster", clustered_cells.columns)
        self.assertIn("alignment_distance", centroid_table.columns)
        self.assertIn("reference_donor_label", centroid_table.columns)

        centroid_lookup = centroid_table.set_index(["donor_label", "aligned_cluster"])
        for aligned_cluster in ["Cluster 1", "Cluster 2"]:
            donor_d45 = centroid_lookup.loc[("Donor D45", aligned_cluster)]
            donor_d47 = centroid_lookup.loc[("Donor D47", aligned_cluster)]
            self.assertAlmostEqual(
                donor_d45["cd206_intensity_centroid"],
                donor_d47["cd206_intensity_centroid"],
                places=6,
            )
            self.assertAlmostEqual(
                donor_d45["area_centroid"],
                donor_d47["area_centroid"],
                places=6,
            )

        reference_rows = centroid_table.loc[centroid_table["donor_label"] == "Donor D45"]
        self.assertTrue((reference_rows["alignment_distance"] == 0.0).all())

    def test_run_split_kmeans_rejects_missing_reference_donor(self) -> None:
        with self.assertRaises(ValueError):
            run_split_kmeans(
                self._build_cells(),
                k=2,
                random_seed=3,
                max_iterations=12,
                reference_donor_label="Donor Missing",
            )

    def test_build_aligned_cluster_composition_table_normalizes_per_condition(self) -> None:
        _, _, clustered_cells, _ = run_split_kmeans(
            self._build_cells(),
            k=2,
            random_seed=3,
            max_iterations=12,
            reference_donor_label="Donor D45",
        )

        composition = build_aligned_cluster_composition_table(clustered_cells)

        self.assertEqual(
            set(composition.columns),
            {"donor_label", "condition_label", "aligned_cluster", "cell_count", "fraction"},
        )
        fraction_sums = composition.groupby(
            ["donor_label", "condition_label"],
            observed=True,
        )["fraction"].sum()
        for fraction_sum in fraction_sums.to_numpy(dtype=float):
            self.assertAlmostEqual(float(fraction_sum), 1.0, places=8)

    def test_build_aligned_cluster_composition_table_requires_aligned_cluster(self) -> None:
        with self.assertRaises(ValueError):
            build_aligned_cluster_composition_table(
                self._build_cells()[["donor_label", "condition_label"]]
            )


if __name__ == "__main__":
    unittest.main()
