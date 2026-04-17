from __future__ import annotations

import unittest

try:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
except ImportError:  # pragma: no cover - lightweight shell
    pd = None

if pd is not None:
    from macrophage_analysis.plotting.clustering import plot_aligned_cluster_composition


@unittest.skipIf(pd is None, "plotting dependencies are not installed in this environment")
class ClusteringPlottingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
