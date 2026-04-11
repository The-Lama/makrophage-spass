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


def _build_plot_data(
    analysis: BatchAnalysis,
    antibody: str,
    *,
    plot_log_scale: bool,
    min_positive_intensity: float | None,
    max_strip_points: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rng = np.random.default_rng(random_seed)
    full_rows: list[dict[str, float | str]] = []
    sample_rows: list[dict[str, float | str]] = []
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
            if plot_values.size == 0:
                continue

            full_rows.extend(
                {
                    "antibody": antibody,
                    "condition": condition,
                    "donor": donor,
                    "intensity": float(value),
                }
                for value in plot_values
            )
            sampled_values = plot_values
            if sampled_values.size > max_strip_points:
                sampled_values = rng.choice(sampled_values, size=max_strip_points, replace=False)
            sample_rows.extend(
                {
                    "antibody": antibody,
                    "condition": condition,
                    "donor": donor,
                    "intensity": float(value),
                }
                for value in sampled_values
            )

    full_df = pd.DataFrame(full_rows)
    sample_df = pd.DataFrame(sample_rows)
    for frame in (full_df, sample_df):
        if frame.empty:
            continue
        frame["condition"] = pd.Categorical(frame["condition"], categories=analysis.conditions, ordered=True)
        frame["donor"] = pd.Categorical(frame["donor"], categories=analysis.donors, ordered=True)
    return full_df, sample_df, grouped_values


def build_stat_summary_table(
    analysis: BatchAnalysis,
    *,
    plot_log_scale: bool = False,
    min_positive_intensity: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for antibody in analysis.antibody_order:
        _, _, grouped_values = _build_plot_data(
            analysis,
            antibody,
            plot_log_scale=plot_log_scale,
            min_positive_intensity=min_positive_intensity,
            max_strip_points=0,
            random_seed=0,
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

    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown = None
        display = None

    display_columns = ["condition", "donor", "cell_count", "median_intensity", "mean_intensity", "p_value", "stars"]
    renamed_columns = {
        "condition": "Condition",
        "donor": "Donor",
        "cell_count": "Cells",
        "median_intensity": "Median",
        "mean_intensity": "Mean",
        "p_value": "p-value",
        "stars": "Sig.",
    }
    baseline_label = resolved_condition_labels.get(baseline_condition, baseline_condition)

    for antibody in antibody_order:
        antibody_stats = stat_table.loc[stat_table["antibody"] == antibody, display_columns].copy()
        if antibody_stats.empty:
            continue
        antibody_stats["condition"] = antibody_stats["condition"].map(
            lambda condition: resolved_condition_labels.get(str(condition), str(condition))
        )
        antibody_stats["donor"] = antibody_stats["donor"].map(
            lambda donor: resolved_donor_labels.get(str(donor), str(donor))
        )
        antibody_stats["median_intensity"] = antibody_stats["median_intensity"].map(lambda value: f"{value:.2f}")
        antibody_stats["mean_intensity"] = antibody_stats["mean_intensity"].map(lambda value: f"{value:.2f}")
        antibody_stats["p_value"] = antibody_stats.apply(
            lambda row: "baseline" if str(row["condition"]) == baseline_label else f"{row['p_value']:.3g}",
            axis=1,
        )
        formatted_stats = antibody_stats.rename(columns=renamed_columns)
        if Markdown is not None and display is not None:
            display(Markdown(f"**{antibody}**"))
            display(Markdown(f"```text\n{formatted_stats.to_string(index=False)}\n```"))
        else:
            print(f"\n{antibody}")
            print(formatted_stats.to_string(index=False))


def build_morphology_table(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    donor_list = list(analysis.donors if donors is None else donors)
    condition_list = list(analysis.conditions if conditions is None else conditions)
    rows: list[dict[str, float | int | str]] = []
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
            valid = np.isfinite(intensities) & np.isfinite(areas) & np.isfinite(eccentricities)
            rows.extend(
                {
                    "antibody": antibody,
                    "condition": condition,
                    "condition_label": default_condition_display_label(condition),
                    "donor": donor,
                    "donor_label": DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor),
                    "cell_index": int(cell_index),
                    "intensity": float(intensities[cell_index]),
                    "area": float(areas[cell_index]),
                    "eccentricity": float(eccentricities[cell_index]),
                    "path": str(measurement.path),
                    "filename": measurement.path.name,
                }
                for cell_index in np.flatnonzero(valid)
            )

    morphology_df = pd.DataFrame(rows)
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
    for metric in metric_sources:
        summary_df[f"u_stat_{metric}"] = np.nan
        summary_df[f"p_value_{metric}"] = np.nan
        summary_df[f"stars_{metric}"] = "NA"

    grouping_columns = ["condition"] if not per_donor else ["donor", "condition"]
    value_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in morphology_df.groupby(grouping_columns, observed=True)
    }
    for row_index, row in summary_df.iterrows():
        condition = str(row["condition"])
        baseline_key = (baseline_condition,) if not per_donor else (str(row["donor"]), baseline_condition)
        current_key = (condition,) if not per_donor else (str(row["donor"]), condition)
        baseline_group = value_groups.get(baseline_key)
        current_group = value_groups.get(current_key)
        for metric, source_column in metric_sources.items():
            stars_column = f"stars_{metric}"
            if condition == baseline_condition:
                summary_df.at[row_index, stars_column] = "baseline"
                continue
            if baseline_group is None or current_group is None:
                continue
            u_stat, p_value = mann_whitney_u_test(
                baseline_group[source_column].to_numpy(dtype=float),
                current_group[source_column].to_numpy(dtype=float),
            )
            summary_df.at[row_index, f"u_stat_{metric}"] = u_stat
            summary_df.at[row_index, f"p_value_{metric}"] = p_value
            summary_df.at[row_index, stars_column] = pvalue_to_stars(p_value)
    return summary_df


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
            summary_df[f"log2_fc_{metric}"] = summary_df.apply(
                lambda row: float(np.log2(row[metric] / donor_baselines[str(row["donor"])]))
                if (
                    str(row["donor"]) in donor_baselines
                    and np.isfinite(row[metric])
                    and row[metric] > 0
                    and np.isfinite(donor_baselines[str(row["donor"])])
                    and donor_baselines[str(row["donor"])] > 0
                )
                else np.nan,
                axis=1,
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
