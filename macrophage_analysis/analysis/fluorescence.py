from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BatchAnalysis


def prepare_plot_values(
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


def group_plot_values(
    analysis: BatchAnalysis,
    antibody: str,
    *,
    plot_log_scale: bool,
    min_positive_intensity: float | None,
) -> dict[tuple[str, str], np.ndarray]:
    grouped_values: dict[tuple[str, str], np.ndarray] = {}
    for condition in analysis.conditions:
        for donor in analysis.donors:
            measurement = analysis.get_result(antibody, condition, donor)
            grouped_values[(condition, donor)] = prepare_plot_values(
                measurement.measurement_values,
                plot_log_scale=plot_log_scale,
                min_positive_intensity=min_positive_intensity,
            )
    return grouped_values


def build_plot_frames(
    analysis: BatchAnalysis,
    antibody: str,
    *,
    plot_log_scale: bool,
    min_positive_intensity: float | None,
    max_strip_points: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rng = np.random.default_rng(random_seed)
    full_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    grouped_values = group_plot_values(
        analysis,
        antibody,
        plot_log_scale=plot_log_scale,
        min_positive_intensity=min_positive_intensity,
    )

    for condition in analysis.conditions:
        for donor in analysis.donors:
            plot_values = grouped_values[(condition, donor)]
            if plot_values.size == 0:
                continue
            full_frames.append(
                pd.DataFrame(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "donor": donor,
                        "intensity": plot_values,
                    }
                )
            )
            sampled_values = plot_values
            if max_strip_points == 0:
                sampled_values = sampled_values[:0]
            elif sampled_values.size > max_strip_points:
                sampled_values = rng.choice(sampled_values, size=max_strip_points, replace=False)
            sample_frames.append(
                pd.DataFrame(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "donor": donor,
                        "intensity": sampled_values,
                    }
                )
            )

    return (
        _categorize_plot_frame(
            pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame(),
            analysis=analysis,
        ),
        _categorize_plot_frame(
            pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(),
            analysis=analysis,
        ),
        grouped_values,
    )


def _categorize_plot_frame(frame: pd.DataFrame, *, analysis: BatchAnalysis) -> pd.DataFrame:
    if frame.empty:
        return frame
    categorized = frame.copy()
    categorized["condition"] = pd.Categorical(
        categorized["condition"],
        categories=analysis.conditions,
        ordered=True,
    )
    categorized["donor"] = pd.Categorical(
        categorized["donor"],
        categories=analysis.donors,
        ordered=True,
    )
    return categorized
