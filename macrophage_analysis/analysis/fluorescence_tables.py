from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..defaults import (
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_DONOR_DISPLAY_LABELS,
)
from ..labels import default_condition_display_label
from ..models import BatchAnalysis
from ..stats import mann_whitney_u_test, pvalue_to_stars
from .fluorescence import group_plot_values

__all__ = [
    "build_stat_summary_table",
    "format_stat_summary_tables",
]


def build_stat_summary_table(
    analysis: BatchAnalysis,
    *,
    plot_log_scale: bool = False,
    min_positive_intensity: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for antibody in analysis.antibody_order:
        grouped_values = group_plot_values(
            analysis,
            antibody,
            plot_log_scale=plot_log_scale,
            min_positive_intensity=min_positive_intensity,
        )
        for donor in analysis.donors:
            baseline_plot_values = grouped_values[(analysis.baseline_condition, donor)]
            for condition in analysis.conditions:
                measurement = analysis.get_result(antibody, condition, donor)
                row: dict[str, float | int | str] = {
                    "antibody": antibody,
                    "condition": condition,
                    "donor": donor,
                    "cell_count": measurement.cell_count,
                    "mean_intensity": measurement.overall_mean,
                    "median_intensity": measurement.overall_median,
                    "u_stat": float("nan"),
                    "p_value": float("nan"),
                    "stars": "baseline" if condition == analysis.baseline_condition else "NA",
                }
                if condition != analysis.baseline_condition:
                    current_plot_values = grouped_values[(condition, donor)]
                    u_stat, p_value = mann_whitney_u_test(
                        baseline_plot_values,
                        current_plot_values,
                    )
                    row["u_stat"] = u_stat
                    row["p_value"] = p_value
                    row["stars"] = pvalue_to_stars(p_value)
                rows.append(row)

    stat_table = pd.DataFrame(rows)
    stat_table["condition"] = pd.Categorical(
        stat_table["condition"],
        categories=analysis.conditions,
        ordered=True,
    )
    stat_table["donor"] = pd.Categorical(
        stat_table["donor"],
        categories=analysis.donors,
        ordered=True,
    )
    stat_table["antibody"] = pd.Categorical(
        stat_table["antibody"],
        categories=analysis.antibody_order,
        ordered=True,
    )
    return stat_table.sort_values(["antibody", "condition", "donor"]).reset_index(drop=True)


def format_stat_summary_tables(
    stat_table: pd.DataFrame,
    *,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    condition_labels: Mapping[str, str] | None = None,
    donor_labels: Mapping[str, str] | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    resolved_condition_labels = {
        condition: condition_labels[condition]
        if condition_labels is not None and condition in condition_labels
        else default_condition_display_label(condition)
        for condition in stat_table["condition"].astype(str).unique()
    }
    resolved_donor_labels = {
        donor: donor_labels[donor]
        if donor_labels is not None and donor in donor_labels
        else DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor)
        for donor in stat_table["donor"].astype(str).unique()
    }

    display_columns = [
        "condition",
        "donor",
        "cell_count",
        "median_intensity",
        "mean_intensity",
        "p_value",
        "stars",
    ]
    renamed_columns = {
        "condition": "Condition",
        "donor": "Donor",
        "cell_count": "Cells",
        "median_intensity": "Median",
        "mean_intensity": "Mean",
        "p_value": "p-value",
        "stars": "Sig.",
    }

    formatted_tables: list[tuple[str, pd.DataFrame]] = []
    for antibody in antibody_order:
        antibody_stats = stat_table.loc[
            stat_table["antibody"] == antibody,
            display_columns,
        ].copy()
        if antibody_stats.empty:
            continue
        baseline_mask = antibody_stats["condition"].astype(str).eq(baseline_condition)
        antibody_stats["condition"] = antibody_stats["condition"].astype(str).map(
            lambda condition: resolved_condition_labels.get(condition, condition)
        )
        antibody_stats["donor"] = antibody_stats["donor"].astype(str).map(
            lambda donor: resolved_donor_labels.get(donor, donor)
        )
        antibody_stats["median_intensity"] = antibody_stats["median_intensity"].map(
            lambda value: f"{value:.2f}"
        )
        antibody_stats["mean_intensity"] = antibody_stats["mean_intensity"].map(
            lambda value: f"{value:.2f}"
        )
        antibody_stats["p_value"] = np.where(
            baseline_mask,
            "baseline",
            antibody_stats["p_value"].map(lambda value: f"{value:.3g}"),
        )
        formatted_tables.append((antibody, antibody_stats.rename(columns=renamed_columns)))
    return formatted_tables
