from __future__ import annotations

import unittest

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - lightweight shell
    widgets = None

if widgets is not None:
    from macrophage_analysis.notebook import (
        build_clustering_feature_labels,
        build_marker_dropdown,
    )


@unittest.skipIf(widgets is None, "ipywidgets is not installed in this environment")
class NotebookHelperTests(unittest.TestCase):
    def test_build_marker_dropdown_uses_requested_default_marker(self) -> None:
        dropdown = build_marker_dropdown(options=("CD40", "CD206"), default_marker="CD206")

        self.assertEqual(tuple(dropdown.options), ("CD40", "CD206"))
        self.assertEqual(dropdown.value, "CD206")
        self.assertEqual(dropdown.description, "Marker")

    def test_build_marker_dropdown_falls_back_to_first_option(self) -> None:
        dropdown = build_marker_dropdown(options=("CD40", "CD86"), default_marker="CD206")

        self.assertEqual(dropdown.value, "CD40")

    def test_build_marker_dropdown_requires_non_empty_options(self) -> None:
        with self.assertRaises(ValueError):
            build_marker_dropdown(options=())

    def test_build_clustering_feature_labels_includes_marker_brightness(self) -> None:
        labels = build_clustering_feature_labels(
            "CD206",
            extra_labels={"area": "Area", "eccentricity": "Eccentricity"},
        )

        self.assertEqual(
            labels,
            {
                "intensity": "CD206 brightness",
                "area": "Area",
                "eccentricity": "Eccentricity",
            },
        )


if __name__ == "__main__":
    unittest.main()
