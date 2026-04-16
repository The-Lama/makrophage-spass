from __future__ import annotations

import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    pd = None

if pd is not None:
    from macrophage_analysis.analysis.tables import format_stat_summary_tables


@unittest.skipIf(pd is None, "pandas is not installed in this environment")
class StatTableFormattingTests(unittest.TestCase):
    def test_format_stat_summary_tables_applies_labels_and_baseline_text(self) -> None:
        stat_table = pd.DataFrame(
            [
                {
                    "antibody": "CD40",
                    "condition": "M0",
                    "donor": "D45",
                    "cell_count": 10,
                    "mean_intensity": 5.0,
                    "median_intensity": 4.0,
                    "u_stat": float("nan"),
                    "p_value": float("nan"),
                    "stars": "baseline",
                },
                {
                    "antibody": "CD40",
                    "condition": "B68KCP2",
                    "donor": "D45",
                    "cell_count": 9,
                    "mean_intensity": 8.0,
                    "median_intensity": 7.0,
                    "u_stat": 12.0,
                    "p_value": 0.0123,
                    "stars": "*",
                },
            ]
        )

        formatted_tables = format_stat_summary_tables(
            stat_table,
            antibody_order=["CD40"],
            baseline_condition="M0",
            condition_labels={"M0": "Baseline", "B68KCP2": "B68"},
            donor_labels={"D45": "Donor 45"},
        )

        self.assertEqual(len(formatted_tables), 1)
        antibody, formatted = formatted_tables[0]
        self.assertEqual(antibody, "CD40")
        self.assertEqual(list(formatted.columns), ["Condition", "Donor", "Cells", "Median", "Mean", "p-value", "Sig."])
        self.assertEqual(formatted.iloc[0]["Condition"], "Baseline")
        self.assertEqual(formatted.iloc[0]["p-value"], "baseline")
        self.assertEqual(formatted.iloc[1]["Donor"], "Donor 45")
        self.assertEqual(formatted.iloc[1]["p-value"], "0.0123")


if __name__ == "__main__":
    unittest.main()
