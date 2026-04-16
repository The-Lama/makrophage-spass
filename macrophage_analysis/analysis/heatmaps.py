from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BatchAnalysis, MeasurementResult


@dataclass(frozen=True)
class HeatmapPayload:
    matrix: np.ndarray
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    color_limit: float


def _resolve_summary_value(measurement: MeasurementResult, *, summary_stat: str) -> float:
    if summary_stat == "median":
        return measurement.overall_median
    if summary_stat == "mean":
        return measurement.overall_mean
    raise ValueError(f"Unsupported summary_stat: {summary_stat}")


def build_summary_heatmap_payload(
    analysis: BatchAnalysis,
    *,
    summary_stat: str = "median",
    exclude_treatment: str | None = None,
) -> dict[str, HeatmapPayload]:
    condition_order = list(analysis.conditions)
    antibody_order = list(analysis.antibody_order)
    if exclude_treatment is not None:
        if exclude_treatment not in condition_order:
            raise ValueError(f"exclude_treatment must be one of {condition_order}")
        condition_order = [condition for condition in condition_order if condition != exclude_treatment]
    if not condition_order:
        raise ValueError("At least one treatment must remain in the heatmap")

    donor_matrices: dict[str, np.ndarray] = {}
    all_fold_changes: list[float] = []
    for donor in analysis.donors:
        donor_rows: list[list[float]] = []
        for condition in condition_order:
            row: list[float] = []
            for antibody in antibody_order:
                baseline_measurement = analysis.get_result(antibody, analysis.baseline_condition, donor)
                current_measurement = analysis.get_result(antibody, condition, donor)
                baseline_value = _resolve_summary_value(baseline_measurement, summary_stat=summary_stat)
                current_value = _resolve_summary_value(current_measurement, summary_stat=summary_stat)
                if condition == analysis.baseline_condition and np.isfinite(baseline_value):
                    fold_change = 0.0
                elif (
                    not np.isfinite(baseline_value)
                    or baseline_value <= 0
                    or not np.isfinite(current_value)
                    or current_value <= 0
                ):
                    fold_change = np.nan
                else:
                    fold_change = float(np.log2(current_value / baseline_value))
                row.append(fold_change)
                if np.isfinite(fold_change):
                    all_fold_changes.append(fold_change)
            donor_rows.append(row)
        donor_matrices[donor] = np.array(donor_rows, dtype=float)

    color_limit = max(1.0, float(np.nanmax(np.abs(all_fold_changes)))) if all_fold_changes else 1.0
    return {
        donor: HeatmapPayload(
            matrix=matrix,
            row_labels=tuple(condition_order),
            column_labels=tuple(antibody_order),
            color_limit=color_limit,
        )
        for donor, matrix in donor_matrices.items()
    }
