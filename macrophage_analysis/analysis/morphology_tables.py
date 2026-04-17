from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..defaults import DEFAULT_DONOR_DISPLAY_LABELS
from ..labels import default_condition_display_label
from ..models import BatchAnalysis, MeasurementResult
from ..stats import mann_whitney_u_test, pvalue_to_stars

__all__ = [
    "build_morphology_fold_change_table",
    "build_morphology_summary_table",
    "build_morphology_table",
]


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
            measurement = analysis.get_result(antibody, condition, donor)
            frame = _build_morphology_measurement_frame(
                measurement,
                antibody=antibody,
                condition=condition,
                donor=donor,
            )
            if frame is not None:
                frames.append(frame)

    morphology_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if morphology_df.empty:
        return morphology_df
    return _categorize_morphology_table(
        morphology_df,
        donor_list=donor_list,
        condition_list=condition_list,
    )


def _build_morphology_measurement_frame(
    measurement: MeasurementResult,
    *,
    antibody: str,
    condition: str,
    donor: str,
) -> pd.DataFrame | None:
    if measurement.areas is None or measurement.eccentricities is None:
        raise ValueError(
            "This BatchAnalysis does not include morphology metrics. "
            "Rerun extract_single_cell_fluorescence after updating the macrophage_analysis package."
        )

    intensities = np.asarray(measurement.measurement_values, dtype=float)
    areas = np.asarray(measurement.areas, dtype=float)
    eccentricities = np.asarray(measurement.eccentricities, dtype=float)
    if not (intensities.size == areas.size == eccentricities.size):
        raise ValueError(f"Mismatched cell metrics for {antibody} / {condition} / {donor}")

    valid_index = np.flatnonzero(
        np.isfinite(intensities) & np.isfinite(areas) & np.isfinite(eccentricities)
    )
    if valid_index.size == 0:
        return None

    return pd.DataFrame(
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


def _categorize_morphology_table(
    morphology_df: pd.DataFrame,
    *,
    donor_list: list[str],
    condition_list: list[str],
) -> pd.DataFrame:
    categorized_df = morphology_df.copy()
    categorized_df["condition"] = pd.Categorical(
        categorized_df["condition"],
        categories=condition_list,
        ordered=True,
    )
    categorized_df["condition_label"] = pd.Categorical(
        categorized_df["condition_label"],
        categories=[default_condition_display_label(condition) for condition in condition_list],
        ordered=True,
    )
    categorized_df["donor"] = pd.Categorical(
        categorized_df["donor"],
        categories=donor_list,
        ordered=True,
    )
    categorized_df["donor_label"] = pd.Categorical(
        categorized_df["donor_label"],
        categories=[DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor) for donor in donor_list],
        ordered=True,
    )
    return categorized_df


def build_morphology_summary_table(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    morphology_df = build_morphology_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
    )
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
        morphology_df.groupby(
            ["condition", "condition_label", "donor", "donor_label"],
            observed=True,
        )
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


MORPHOLOGY_METRIC_SOURCES = {
    "area": "area",
    "eccentricity": "eccentricity",
    "cd206_brightness": "intensity",
}


def _normalize_morphology_group_key(group_key: object) -> tuple[object, ...]:
    return group_key if isinstance(group_key, tuple) else (group_key,)


def _build_morphology_group_lookup(
    morphology_df: pd.DataFrame,
    *,
    grouping_columns: list[str],
) -> dict[tuple[object, ...], pd.DataFrame]:
    return {
        _normalize_morphology_group_key(group_key): group
        for group_key, group in morphology_df.groupby(grouping_columns, observed=True)
    }


def _build_morphology_stat_row(
    group_key: tuple[object, ...],
    current_group: pd.DataFrame,
    group_lookup: dict[tuple[object, ...], pd.DataFrame],
    *,
    baseline_condition: str,
    per_donor: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        f"u_stat_{metric}": np.nan for metric in MORPHOLOGY_METRIC_SOURCES
    }
    row.update({f"p_value_{metric}": np.nan for metric in MORPHOLOGY_METRIC_SOURCES})
    row.update({f"stars_{metric}": "NA" for metric in MORPHOLOGY_METRIC_SOURCES})

    if per_donor:
        donor, condition = group_key
        row["donor"] = donor
        baseline_key = (donor, baseline_condition)
    else:
        (condition,) = group_key
        baseline_key = (baseline_condition,)
    row["condition"] = condition

    if condition == baseline_condition:
        for metric in MORPHOLOGY_METRIC_SOURCES:
            row[f"stars_{metric}"] = "baseline"
        return row

    baseline_group = group_lookup.get(baseline_key)
    if baseline_group is None:
        return row

    for metric, source_column in MORPHOLOGY_METRIC_SOURCES.items():
        u_stat, p_value = mann_whitney_u_test(
            baseline_group[source_column].to_numpy(dtype=float),
            current_group[source_column].to_numpy(dtype=float),
        )
        row[f"u_stat_{metric}"] = u_stat
        row[f"p_value_{metric}"] = p_value
        row[f"stars_{metric}"] = pvalue_to_stars(p_value)
    return row


def _add_morphology_stat_columns(
    summary_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    *,
    baseline_condition: str,
    per_donor: bool,
) -> pd.DataFrame:
    grouping_columns = ["condition"] if not per_donor else ["donor", "condition"]
    group_lookup = _build_morphology_group_lookup(
        morphology_df,
        grouping_columns=grouping_columns,
    )
    stats_df = pd.DataFrame(
        [
            _build_morphology_stat_row(
                group_key,
                current_group,
                group_lookup,
                baseline_condition=baseline_condition,
                per_donor=per_donor,
            )
            for group_key, current_group in group_lookup.items()
        ]
    )
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
    morphology_df = build_morphology_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
    )
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")
    resolved_baseline_condition = baseline_condition or analysis.baseline_condition
    if resolved_baseline_condition not in set(morphology_df["condition"].astype(str)):
        raise ValueError(
            f"baseline_condition must be present in the morphology table: {resolved_baseline_condition}"
        )
    summary_function = _morphology_summary_function(summary_stat)
    group_columns = (
        ["condition", "condition_label"]
        if not per_donor
        else ["donor", "donor_label", "condition", "condition_label"]
    )
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
        baseline_rows = summary_df.loc[
            summary_df["condition"].astype(str) == resolved_baseline_condition
        ]
        baseline_donors = set(baseline_rows["donor"].astype(str))
        missing_donors = sorted(set(summary_df["donor"].astype(str)) - baseline_donors)
        if missing_donors:
            raise ValueError(
                "baseline_condition must be present for every donor in the morphology table: "
                + ", ".join(missing_donors)
            )
        for metric in metrics:
            donor_baselines = dict(
                zip(
                    baseline_rows["donor"].astype(str),
                    baseline_rows[metric].astype(float),
                )
            )
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
