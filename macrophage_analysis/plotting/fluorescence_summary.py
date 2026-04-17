from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.fluorescence import build_plot_frames
from ..analysis.fluorescence_tables import build_stat_summary_table
from ..analysis.heatmaps import build_summary_heatmap_payload
from ..defaults import (
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_DONOR_DISPLAY_LABELS,
    DEFAULT_SEABORN_THEME,
)
from ..labels import (
    default_condition_display_label,
    default_measurement_axis_label,
    default_measurement_summary_label,
    default_measurement_title_label,
    wrap_display_label,
)
from ..models import BatchAnalysis
from ._shared import _build_stat_star_lookup, _deduplicate_legend
from .tables import display_stat_summary_tables as _display_stat_summary_tables
from .utils import build_heatmap_annotation_labels


def display_stat_summary_tables(
    stat_table: pd.DataFrame,
    *,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    condition_labels: Mapping[str, str] | None = None,
    donor_labels: Mapping[str, str] | None = None,
) -> None:
    _display_stat_summary_tables(
        stat_table,
        antibody_order=antibody_order,
        baseline_condition=baseline_condition,
        condition_labels=condition_labels,
        donor_labels=donor_labels,
    )


def plot_violins_with_stats(
    analysis: BatchAnalysis,
    *,
    max_strip_points: int = 250,
    plot_log_scale: bool = False,
    min_positive_intensity: float | None = None,
    upper_display_percentile: float = 99.5,
    random_seed: int = 7,
    y_label: str | None = None,
    title_value_label: str | None = None,
) -> pd.DataFrame:
    sns.set_theme(**DEFAULT_SEABORN_THEME)
    palette = {donor: analysis.donor_colors[donor] for donor in analysis.donors}
    resolved_y_label = y_label or default_measurement_axis_label(analysis.measurement_scale)
    resolved_title_value_label = title_value_label or default_measurement_title_label(analysis.measurement_scale)
    stat_table = build_stat_summary_table(
        analysis,
        plot_log_scale=plot_log_scale,
        min_positive_intensity=min_positive_intensity,
    )
    for antibody in analysis.antibody_order:
        plot_df, sample_df, grouped_values = build_plot_frames(
            analysis,
            antibody,
            plot_log_scale=plot_log_scale,
            min_positive_intensity=min_positive_intensity,
            max_strip_points=max_strip_points,
            random_seed=random_seed,
        )
        if plot_df.empty:
            raise ValueError(f"No valid intensities available for {antibody}")
        fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
        sns.violinplot(
            data=plot_df,
            x="condition",
            y="intensity",
            hue="donor",
            order=analysis.conditions,
            hue_order=analysis.donors,
            palette=palette,
            cut=0,
            inner="quart",
            linewidth=1.0,
            dodge=True,
            ax=ax,
        )
        sns.stripplot(
            data=sample_df,
            x="condition",
            y="intensity",
            hue="donor",
            order=analysis.conditions,
            hue_order=analysis.donors,
            palette=palette,
            dodge=True,
            jitter=0.16,
            alpha=0.18,
            size=2.5,
            linewidth=0,
            ax=ax,
        )
        _deduplicate_legend(ax, analysis.donors)
        if not 0 < upper_display_percentile <= 100:
            raise ValueError("upper_display_percentile must be between 0 and 100")
        display_max = max(float(np.percentile(values, upper_display_percentile)) for values in grouped_values.values() if values.size > 0)
        global_min = float(plot_df["intensity"].min())
        if plot_log_scale:
            log_lower_bound = (
                float(min_positive_intensity)
                if min_positive_intensity is not None
                else max(global_min / 1.5, np.finfo(float).tiny)
            )
            ax.set_yscale("log")
            ax.set_ylim(bottom=log_lower_bound, top=display_max * 2.2)
        else:
            ax.set_ylim(bottom=0, top=display_max * 1.15)
        ax.set_title(f"{antibody} {resolved_title_value_label} by treatment")
        ax.set_xlabel("Treatment")
        ax.set_ylabel(resolved_y_label + (" (log scale)" if plot_log_scale else ""))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)
        plt.show()
    return stat_table


def plot_summary_heatmap(
    analysis: BatchAnalysis,
    *,
    summary_stat: str = "median",
    summary_stat_label: str | None = None,
    stat_results: pd.DataFrame | None = None,
    condition_labels: Mapping[str, str] | None = None,
    antibody_labels: Mapping[str, str] | None = None,
    donor_labels: Mapping[str, str] | None = None,
    show_significance_stars: bool = True,
    exclude_treatment: str | None = None,
    condition_label_style: str = "short",
    condition_label_wrap_width: int | None = None,
) -> dict[str, np.ndarray]:
    sns.set_theme(**DEFAULT_SEABORN_THEME)

    resolved_condition_label_wrap_width = (
        18
        if condition_label_style == "descriptive" and condition_label_wrap_width is None
        else condition_label_wrap_width
    )
    resolved_condition_labels = {
        condition: condition_labels[condition]
        if condition_labels is not None and condition in condition_labels
        else default_condition_display_label(condition, style=condition_label_style)
        for condition in analysis.conditions
    }
    display_condition_labels = {
        condition: default_condition_display_label(
            condition,
            style=condition_label_style,
            wrap_width=resolved_condition_label_wrap_width,
        )
        for condition in analysis.conditions
    }
    if condition_labels is not None:
        for condition, label in condition_labels.items():
            if condition in display_condition_labels:
                display_condition_labels[condition] = wrap_display_label(label, width=resolved_condition_label_wrap_width)
    resolved_antibody_labels = {
        antibody: antibody_labels[antibody]
        if antibody_labels is not None and antibody in antibody_labels
        else antibody
        for antibody in analysis.antibody_order
    }
    resolved_donor_labels = {
        donor: donor_labels[donor]
        if donor_labels is not None and donor in donor_labels
        else DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor)
        for donor in analysis.donors
    }
    baseline_label = resolved_condition_labels.get(analysis.baseline_condition, analysis.baseline_condition)
    resolved_summary_stat_label = summary_stat_label or default_measurement_summary_label(
        analysis.measurement_scale,
        summary_stat=summary_stat,
    )
    heatmap_payloads = build_summary_heatmap_payload(
        analysis,
        summary_stat=summary_stat,
        exclude_treatment=exclude_treatment,
    )
    if not heatmap_payloads:
        raise ValueError("At least one donor is required for the heatmap")
    stat_star_lookup = _build_stat_star_lookup(stat_results)
    fig, axes = plt.subplots(1, len(analysis.donors), figsize=(5.5 * len(analysis.donors), 6), constrained_layout=True)
    if len(analysis.donors) == 1:
        axes = [axes]
    for ax, donor in zip(axes, analysis.donors):
        payload = heatmap_payloads[donor]
        matrix = payload.matrix
        annotation_labels = build_heatmap_annotation_labels(
            matrix,
            row_labels=payload.row_labels,
            column_labels=payload.column_labels,
            star_getter=(
                (
                    lambda row_index, column_index, donor=donor, payload=payload: stat_star_lookup.get(
                        (donor, payload.row_labels[row_index], payload.column_labels[column_index])
                    )
                )
                if show_significance_stars and stat_star_lookup
                else None
            ),
        )
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-payload.color_limit,
            vmax=payload.color_limit,
            annot=annotation_labels,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            xticklabels=[resolved_antibody_labels[antibody] for antibody in payload.column_labels],
            yticklabels=[display_condition_labels[condition] for condition in payload.row_labels],
            cbar=ax is axes[-1],
            cbar_kws={"label": f"Log2 fold change relative to {baseline_label}"},
        )
        ax.set_title(resolved_donor_labels[donor], fontsize=12, fontweight="bold")
        ax.set_xlabel("Marker")
        ax.set_ylabel("Treatment condition" if ax is axes[0] else "")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        sns.despine(ax=ax, left=True, bottom=True)
    fig.suptitle(
        f"Phenotype fingerprint by donor\nLog2 fold change vs {baseline_label} ({resolved_summary_stat_label})",
        fontsize=14,
    )
    plt.show()
    return {donor: payload.matrix for donor, payload in heatmap_payloads.items()}
