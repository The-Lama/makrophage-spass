from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import (
    BatchAnalysis,
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_DONOR_DISPLAY_LABELS,
    default_condition_display_label,
)
from .stats import mann_whitney_u_test, pvalue_to_stars


def _prepare_plot_values(
    values: np.ndarray,
    *,
    plot_log_scale: bool,
    min_positive_intensity: float | None,
) -> np.ndarray:
    plot_values = np.asarray(values, dtype=float)
    plot_values = plot_values[np.isfinite(plot_values)]
    if plot_log_scale:
        plot_values = plot_values[plot_values > 0]
        if min_positive_intensity is not None:
            plot_values = np.clip(plot_values, min_positive_intensity, None)
    return plot_values


def _group_plot_values(
    analysis: BatchAnalysis,
    antibody: str,
    *,
    plot_log_scale: bool,
    min_positive_intensity: float | None,
) -> dict[tuple[str, str], np.ndarray]:
    grouped_values: dict[tuple[str, str], np.ndarray] = {}

    for condition in analysis.conditions:
        for donor in analysis.donors:
            measurement = analysis.results[(antibody, condition, donor)]
            plot_values = _prepare_plot_values(
                measurement.mean_intensities,
                plot_log_scale=plot_log_scale,
                min_positive_intensity=min_positive_intensity,
            )
            grouped_values[(condition, donor)] = plot_values
    return grouped_values


def build_stat_summary_table(
    analysis: BatchAnalysis,
    *,
    plot_log_scale: bool = False,
    min_positive_intensity: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for antibody in analysis.antibody_order:
        grouped_values = _group_plot_values(
            analysis,
            antibody,
            plot_log_scale=plot_log_scale,
            min_positive_intensity=min_positive_intensity,
        )
        for donor in analysis.donors:
            baseline_plot_values = grouped_values[(analysis.baseline_condition, donor)]
            for condition in analysis.conditions:
                measurement = analysis.results[(antibody, condition, donor)]
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
                    u_stat, p_value = mann_whitney_u_test(baseline_plot_values, current_plot_values)
                    row["u_stat"] = u_stat
                    row["p_value"] = p_value
                    row["stars"] = pvalue_to_stars(p_value)
                rows.append(row)

    stat_table = pd.DataFrame(rows)
    stat_table["condition"] = pd.Categorical(stat_table["condition"], categories=analysis.conditions, ordered=True)
    stat_table["donor"] = pd.Categorical(stat_table["donor"], categories=analysis.donors, ordered=True)
    stat_table["antibody"] = pd.Categorical(
        stat_table["antibody"], categories=analysis.antibody_order, ordered=True
    )
    return stat_table.sort_values(["antibody", "condition", "donor"]).reset_index(drop=True)


def display_stat_summary_tables(
    stat_table: pd.DataFrame,
    *,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    condition_labels: Mapping[str, str] | None = None,
    donor_labels: Mapping[str, str] | None = None,
) -> None:
    from .plots_fluorescence import display_stat_summary_tables as _display_stat_summary_tables

    _display_stat_summary_tables(
        stat_table,
        antibody_order=antibody_order,
        baseline_condition=baseline_condition,
        condition_labels=condition_labels,
        donor_labels=donor_labels,
    )


def build_morphology_table(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    donor_list = list(analysis.donors if donors is None else donors)
    condition_list = list(analysis.conditions if conditions is None else conditions)
    frames: list[pd.DataFrame] = []
    for condition in condition_list:
        for donor in donor_list:
            measurement = analysis.results[(antibody, condition, donor)]
            if measurement.areas is None or measurement.eccentricities is None:
                raise ValueError(
                    "This BatchAnalysis does not include morphology metrics. "
                    "Rerun extract_single_cell_fluorescence after updating the macrophage_analysis package."
                )
            intensities = np.asarray(measurement.mean_intensities, dtype=float)
            areas = np.asarray(measurement.areas, dtype=float)
            eccentricities = np.asarray(measurement.eccentricities, dtype=float)
            if not (intensities.size == areas.size == eccentricities.size):
                raise ValueError(f"Mismatched cell metrics for {antibody} / {condition} / {donor}")
            valid_index = np.flatnonzero(np.isfinite(intensities) & np.isfinite(areas) & np.isfinite(eccentricities))
            if valid_index.size == 0:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "condition_label": default_condition_display_label(condition),
                        "donor": donor,
                        "donor_label": DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor),
                        "cell_index": valid_index,
                        "intensity": intensities[valid_index],
                        "area": areas[valid_index],
                        "eccentricity": eccentricities[valid_index],
                        "path": str(measurement.path),
                        "filename": measurement.path.name,
                    }
                )
            )

    morphology_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if morphology_df.empty:
        return morphology_df
    morphology_df["condition"] = pd.Categorical(morphology_df["condition"], categories=condition_list, ordered=True)
    morphology_df["condition_label"] = pd.Categorical(
        morphology_df["condition_label"],
        categories=[default_condition_display_label(condition) for condition in condition_list],
        ordered=True,
    )
    morphology_df["donor"] = pd.Categorical(morphology_df["donor"], categories=donor_list, ordered=True)
    morphology_df["donor_label"] = pd.Categorical(
        morphology_df["donor_label"],
        categories=[DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor) for donor in donor_list],
        ordered=True,
    )
    return morphology_df


def build_morphology_summary_table(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    morphology_df = build_morphology_table(analysis, antibody=antibody, donors=donors, conditions=conditions)
    if morphology_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "condition_label",
                "donor",
                "donor_label",
                "cell_count",
                "median_intensity",
                "median_area",
                "median_eccentricity",
            ]
        )
    return (
        morphology_df.groupby(["condition", "condition_label", "donor", "donor_label"], observed=True)
        .agg(
            cell_count=("cell_index", "count"),
            median_intensity=("intensity", "median"),
            median_area=("area", "median"),
            median_eccentricity=("eccentricity", "median"),
        )
        .reset_index()
    )


def _morphology_summary_function(summary_stat: str):
    if summary_stat == "median":
        return "median"
    if summary_stat == "mean":
        return "mean"
    raise ValueError("summary_stat must be 'median' or 'mean'")


def _add_morphology_stat_columns(
    summary_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    *,
    baseline_condition: str,
    per_donor: bool,
) -> pd.DataFrame:
    metric_sources = {"area": "area", "eccentricity": "eccentricity", "cd206_brightness": "intensity"}
    grouping_columns = ["condition"] if not per_donor else ["donor", "condition"]
    value_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in morphology_df.groupby(grouping_columns, observed=True)
    }
    stats_rows: list[dict[str, object]] = []
    for group_key, current_group in value_groups.items():
        row: dict[str, object] = {}
        if per_donor:
            donor, condition = group_key
            row["donor"] = donor
            baseline_key = (donor, baseline_condition)
        else:
            (condition,) = group_key
            baseline_key = (baseline_condition,)
        row["condition"] = condition
        baseline_group = value_groups.get(baseline_key)
        for metric, source_column in metric_sources.items():
            row[f"u_stat_{metric}"] = np.nan
            row[f"p_value_{metric}"] = np.nan
            row[f"stars_{metric}"] = "NA"
            if condition == baseline_condition:
                row[f"stars_{metric}"] = "baseline"
                continue
            if baseline_group is None:
                continue
            u_stat, p_value = mann_whitney_u_test(
                baseline_group[source_column].to_numpy(dtype=float),
                current_group[source_column].to_numpy(dtype=float),
            )
            row[f"u_stat_{metric}"] = u_stat
            row[f"p_value_{metric}"] = p_value
            row[f"stars_{metric}"] = pvalue_to_stars(p_value)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    return summary_df.merge(stats_df, on=grouping_columns, how="left")


def build_morphology_fold_change_table(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    baseline_condition: str | None = None,
    summary_stat: str = "median",
    per_donor: bool = False,
) -> pd.DataFrame:
    morphology_df = build_morphology_table(analysis, antibody=antibody, donors=donors, conditions=conditions)
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")
    resolved_baseline_condition = baseline_condition or analysis.baseline_condition
    if resolved_baseline_condition not in set(morphology_df["condition"].astype(str)):
        raise ValueError(
            f"baseline_condition must be present in the morphology table: {resolved_baseline_condition}"
        )
    summary_function = _morphology_summary_function(summary_stat)
    group_columns = ["condition", "condition_label"] if not per_donor else ["donor", "donor_label", "condition", "condition_label"]
    summary_df = (
        morphology_df.groupby(group_columns, observed=True)
        .agg(
            cell_count=("cell_index", "count"),
            cd206_brightness=("intensity", summary_function),
            area=("area", summary_function),
            eccentricity=("eccentricity", summary_function),
        )
        .reset_index()
    )

    metrics = ("area", "eccentricity", "cd206_brightness")
    if per_donor:
        baseline_rows = summary_df.loc[summary_df["condition"].astype(str) == resolved_baseline_condition]
        baseline_donors = set(baseline_rows["donor"].astype(str))
        missing_donors = sorted(set(summary_df["donor"].astype(str)) - baseline_donors)
        if missing_donors:
            raise ValueError(
                "baseline_condition must be present for every donor in the morphology table: "
                + ", ".join(missing_donors)
            )
        for metric in metrics:
            donor_baselines = dict(zip(baseline_rows["donor"].astype(str), baseline_rows[metric].astype(float)))
            baseline_values = summary_df["donor"].astype(str).map(donor_baselines).astype(float)
            metric_values = summary_df[metric].astype(float)
            valid = (
                np.isfinite(metric_values)
                & (metric_values > 0)
                & np.isfinite(baseline_values)
                & (baseline_values > 0)
            )
            summary_df[f"log2_fc_{metric}"] = np.where(
                valid,
                np.log2(metric_values / baseline_values),
                np.nan,
            )
    else:
        baseline_row = summary_df.loc[
            summary_df["condition"].astype(str) == resolved_baseline_condition
        ].iloc[0]
        for metric in metrics:
            baseline_value = float(baseline_row[metric])
            if not np.isfinite(baseline_value) or baseline_value <= 0:
                summary_df[f"log2_fc_{metric}"] = np.nan
            else:
                summary_df[f"log2_fc_{metric}"] = summary_df[metric].map(
                    lambda value: float(np.log2(value / baseline_value))
                    if np.isfinite(value) and value > 0
                    else np.nan
                )
    return _add_morphology_stat_columns(
        summary_df,
        morphology_df,
        baseline_condition=resolved_baseline_condition,
        per_donor=per_donor,
    )
