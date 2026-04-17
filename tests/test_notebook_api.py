from __future__ import annotations

import unittest
import warnings

import macrophage_analysis as ma


class NotebookApiTests(unittest.TestCase):
    def test_top_level_all_focuses_on_notebook_workflow(self) -> None:
        public_names = set(ma.__all__)

        self.assertIn("DEFAULT_DONORS", public_names)
        self.assertIn("extract_single_cell_fluorescence", public_names)
        self.assertIn("plot_summary_heatmap", public_names)
        self.assertIn("display_stat_summary_tables", public_names)

        self.assertNotIn("BatchAnalysis", public_names)
        self.assertNotIn("find_image_path", public_names)
        self.assertNotIn("load_stardist_model", public_names)

    def test_notebook_default_export_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            donors = ma.DEFAULT_DONORS

        self.assertEqual(tuple(donors), ("D45", "D47"))
        self.assertEqual(caught, [])

    def test_compatibility_export_warns_and_resolves(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            batch_analysis = ma.BatchAnalysis

        self.assertEqual(batch_analysis.__name__, "BatchAnalysis")
        self.assertTrue(any(issubclass(warning.category, FutureWarning) for warning in caught))
        self.assertIn("macrophage_analysis.models", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
