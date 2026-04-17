from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - lightweight shell
    np = None
    pd = None

if np is not None and pd is not None:
    from macrophage_analysis.analysis.gaussian_mixture import run_split_gaussian_mixture


@unittest.skipIf(np is None or pd is None, "numpy/pandas are not installed in this environment")
class GaussianMixtureAnalysisTests(unittest.TestCase):
    def _build_cells(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add_cluster(
            donor: str,
            donor_label: str,
            condition: str,
            condition_label: str,
            intensity_center: float,
            area_center: float,
            eccentricity_center: float,
        ) -> None:
            for offset in range(8):
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": condition_label,
                        "donor": donor,
                        "donor_label": donor_label,
                        "intensity": intensity_center + 0.2 * offset,
                        "area": area_center + 0.4 * offset,
                        "eccentricity": eccentricity_center + 0.01 * offset,
                    }
                )

        add_cluster("D45", "Donor D45", "M0", "M0 (baseline)", 1.0, 10.0, 0.85)
        add_cluster("D45", "Donor D45", "Treat", "Treatment", 9.0, 90.0, 0.25)
        add_cluster("D47", "Donor D47", "Treat", "Treatment", 8.8, 89.0, 0.28)
        add_cluster("D47", "Donor D47", "M0", "M0 (baseline)", 1.2, 11.0, 0.82)

        return pd.DataFrame(rows).astype(
            {
                "condition": "category",
                "condition_label": "category",
                "donor": "category",
                "donor_label": "category",
            }
        )

    def test_run_split_gaussian_mixture_returns_reference_aligned_clusters(self) -> None:
        donor_feature_data, donor_runs, clustered_cells, component_table = run_split_gaussian_mixture(
            self._build_cells(),
            n_components=2,
            random_seed=7,
            reference_donor_label="Donor D45",
        )

        self.assertEqual(list(donor_feature_data), ["Donor D45", "Donor D47"])
        self.assertEqual(set(donor_runs), {"Donor D45", "Donor D47"})
        self.assertIn("aligned_cluster", clustered_cells.columns)
        self.assertIn("assignment_probability", clustered_cells.columns)
        self.assertIn("reference_weight", component_table.columns)
        self.assertIn("donor_weight", component_table.columns)
        self.assertIn("eccentricity_centroid", component_table.columns)

    def test_run_split_gaussian_mixture_preserves_reference_cluster_labels(self) -> None:
        _, _, clustered_cells, component_table = run_split_gaussian_mixture(
            self._build_cells(),
            n_components=2,
            random_seed=7,
            reference_donor_label="Donor D45",
        )

        self.assertTrue(
            (
                clustered_cells["cluster"].astype(str)
                == clustered_cells["aligned_cluster"].astype(str)
            ).all()
        )
        reference_rows = component_table.loc[component_table["donor_label"] == "Donor D45"]
        self.assertTrue((reference_rows["reference_weight"] > 0.0).all())
        self.assertTrue((reference_rows["donor_weight"] > 0.0).all())

    def test_run_split_gaussian_mixture_rejects_missing_reference_donor(self) -> None:
        with self.assertRaises(ValueError):
            run_split_gaussian_mixture(
                self._build_cells(),
                n_components=2,
                random_seed=7,
                reference_donor_label="Donor Missing",
            )

    def test_run_split_gaussian_mixture_supports_custom_feature_columns(self) -> None:
        _, donor_runs, _, component_table = run_split_gaussian_mixture(
            self._build_cells(),
            n_components=2,
            random_seed=7,
            reference_donor_label="Donor D45",
            feature_columns=("intensity", "area"),
            centroid_column_names=("cd206_intensity_centroid", "area_centroid"),
        )

        self.assertEqual(
            donor_runs["Donor D45"]["feature_columns"],
            ("intensity", "area"),
        )
        self.assertIn("area_centroid", component_table.columns)
        self.assertNotIn("eccentricity_centroid", component_table.columns)

    def test_run_split_gaussian_mixture_supports_independent_non_reference_fit(self) -> None:
        shifted_cells = self._build_cells().copy()
        shifted_cells.loc[shifted_cells["donor_label"] == "Donor D47", "intensity"] += 100.0

        _, donor_runs, clustered_cells, component_table = run_split_gaussian_mixture(
            shifted_cells,
            n_components=2,
            random_seed=7,
            reference_donor_label="Donor D45",
            non_reference_mode="independent",
            max_alignment_distance=1.0,
        )

        self.assertEqual(donor_runs["Donor D47"]["non_reference_mode"], "independent")
        self.assertFalse(
            np.allclose(
                donor_runs["Donor D47"]["component_means"],
                donor_runs["Donor D47"]["reference_means"],
            )
        )
        self.assertIn("Unmatched", set(clustered_cells["aligned_cluster"].astype(str)))
        self.assertIn("alignment_distance", component_table.columns)

    def test_run_split_gaussian_mixture_rejects_invalid_non_reference_mode(self) -> None:
        with self.assertRaises(ValueError):
            run_split_gaussian_mixture(
                self._build_cells(),
                n_components=2,
                random_seed=7,
                reference_donor_label="Donor D45",
                non_reference_mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
