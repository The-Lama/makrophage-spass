from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from macrophage_analysis.config import AntibodySpec
from macrophage_analysis.planning import (
    build_condition_comparison_requests,
    build_measurement_requests,
)


class PlanningTests(unittest.TestCase):
    def test_build_measurement_requests_uses_catalog_and_preserves_iteration_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "raw_data"
            files = [
                data_root / "D45" / "M0" / "D45_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tif",
                data_root / "D47" / "M0" / "D47_M0_CD40_A488_10x_RGB_Qi2 10x FITC.tif",
                data_root
                / "D45"
                / "M0"
                / "D45_M0_CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC.tif",
                data_root
                / "D47"
                / "M0"
                / "D47_M0_CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC.tif",
            ]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            requests = build_measurement_requests(
                donors=["D45", "D47"],
                conditions=["M0"],
                antibody_specs={
                    "CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC"),
                    "CD206": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "TRITC"),
                },
                antibody_order=["CD40", "CD206"],
                data_root=data_root,
            )

            self.assertEqual(
                [(request.antibody, request.donor, request.condition) for request in requests],
                [
                    ("CD40", "D45", "M0"),
                    ("CD40", "D47", "M0"),
                    ("CD206", "D45", "M0"),
                    ("CD206", "D47", "M0"),
                ],
            )

    def test_build_condition_comparison_requests_creates_one_request_per_donor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "raw_data"
            for donor in ("D45", "D47"):
                path = (
                    data_root
                    / donor
                    / "B68KCP2"
                    / f"{donor}_B68KCP2_CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC.tif"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            requests = build_condition_comparison_requests(
                donors=["D45", "D47"],
                condition="B68KCP2",
                marker_prefix="CD200R_A488_CD206_A568_10x_RGB_Qi2 10x",
                channel="TRITC",
                data_root=data_root,
            )

            self.assertEqual([request.donor for request in requests], ["D45", "D47"])
            self.assertTrue(all(request.condition == "B68KCP2" for request in requests))


if __name__ == "__main__":
    unittest.main()
