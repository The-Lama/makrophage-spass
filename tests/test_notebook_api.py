from __future__ import annotations

from pathlib import Path
import re
import unittest

import macrophage_analysis as ma


class NotebookApiTests(unittest.TestCase):
    def test_top_level_all_focuses_on_notebook_workflow(self) -> None:
        public_names = set(ma.__all__)

        self.assertIn("DEFAULT_DONORS", public_names)
        self.assertIn("extract_single_cell_fluorescence", public_names)
        self.assertIn("plot_summary_heatmap", public_names)
        self.assertIn("display_stat_summary_tables", public_names)
        self.assertIn("plot_distributions_with_stats", public_names)

        self.assertNotIn("BatchAnalysis", public_names)
        self.assertNotIn("find_image_path", public_names)
        self.assertNotIn("load_stardist_model", public_names)
        self.assertNotIn("build_stat_summary_table", public_names)
        self.assertNotIn("build_morphology_fold_change_table", public_names)
        self.assertNotIn("plot_morphology_scatter", public_names)
        self.assertNotIn("sort_microscopy_images", public_names)
        self.assertNotIn("plot_violins_with_stats", public_names)

    def test_notebook_default_export_resolves(self) -> None:
        donors = ma.DEFAULT_DONORS
        self.assertEqual(tuple(donors), ("D39", "D43"))

    def test_non_notebook_export_is_not_available_top_level(self) -> None:
        with self.assertRaises(AttributeError):
            _ = ma.BatchAnalysis

    def test_notebooks_only_use_notebook_facing_top_level_names(self) -> None:
        notebook_api = set(ma.__all__)
        project_root = Path(__file__).resolve().parents[1]
        notebook_paths = sorted((project_root / "notebooks").glob("*.ipynb"))
        used_names: dict[str, set[str]] = {}
        for notebook_path in notebook_paths:
            names = set(
                re.findall(
                    r"\bma\.([A-Za-z_][A-Za-z0-9_]*)",
                    notebook_path.read_text(encoding="utf-8"),
                )
            )
            if names:
                used_names[notebook_path.name] = names

        unexpected = {
            notebook_name: sorted(names - notebook_api)
            for notebook_name, names in used_names.items()
            if names - notebook_api
        }
        self.assertEqual(unexpected, {})
        self.assertEqual(notebook_api, set().union(*used_names.values()))


if __name__ == "__main__":
    unittest.main()
