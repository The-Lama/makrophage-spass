from __future__ import annotations

import textwrap
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stardist.plot import render_label

from .config import (
    BatchAnalysis,
    ConditionComparison,
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_DONOR_COLORS,
    DEFAULT_DONOR_DISPLAY_LABELS,
    DEFAULT_SEABORN_THEME,
    MeasurementResult,
    default_measurement_axis_label,
    default_measurement_summary_label,
    default_measurement_title_label,
    default_condition_display_label,
)
from .core import load_grayscale_tif
from .tables import build_stat_summary_table


def _wrap_filename(filename: str, *, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(filename, width=width, break_long_words=True))


def _deduplicate_legend(ax: plt.Axes, donors: Iterable[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    donor_set = set(donors)
    for handle, label in zip(handles, labels):
        if label in donor_set and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, title="Donor", frameon=False)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


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
    full_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
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

    full_df = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    for frame in (full_df, sample_df):
        if frame.empty:
            continue
        frame["condition"] = pd.Categorical(frame["condition"], categories=analysis.conditions, ordered=True)
        frame["donor"] = pd.Categorical(frame["donor"], categories=analysis.donors, ordered=True)
    return full_df, sample_df, grouped_values


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

    for antibody in antibody_order:
        antibody_stats = stat_table.loc[stat_table["antibody"] == antibody, display_columns].copy()
        if antibody_stats.empty:
            continue
        baseline_mask = antibody_stats["condition"].astype(str).eq(baseline_condition)
        antibody_stats["condition"] = antibody_stats["condition"].astype(str).map(
            lambda condition: resolved_condition_labels.get(condition, condition)
        )
        antibody_stats["donor"] = antibody_stats["donor"].astype(str).map(
            lambda donor: resolved_donor_labels.get(donor, donor)
        )
        antibody_stats["median_intensity"] = antibody_stats["median_intensity"].map(lambda value: f"{value:.2f}")
        antibody_stats["mean_intensity"] = antibody_stats["mean_intensity"].map(lambda value: f"{value:.2f}")
        antibody_stats["p_value"] = np.where(
            baseline_mask,
            "baseline",
            antibody_stats["p_value"].map(lambda value: f"{value:.3g}"),
        )
        formatted_stats = antibody_stats.rename(columns=renamed_columns)
        if Markdown is not None and display is not None:
            display(Markdown(f"**{antibody}**"))
            display(Markdown(f"```text\n{formatted_stats.to_string(index=False)}\n```"))
        else:
            print(f"\n{antibody}")
            print(formatted_stats.to_string(index=False))


def plot_condition_comparison(comparison: ConditionComparison) -> None:
    fig, axes = plt.subplots(nrows=len(comparison.donors), ncols=2, figsize=(14, 5 * len(comparison.donors)))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"StarDist comparison for condition {comparison.condition} ({comparison.channel})", fontsize=14)
    for row, donor in enumerate(comparison.donors):
        result = comparison.donor_results[donor]
        axes[row, 0].imshow(result.image, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_title(_wrap_filename(result.path.name, width=52), fontsize=9)
        axes[row, 1].imshow(render_label(result.labels, img=result.image))
        axes[row, 1].axis("off")
        axes[row, 1].set_title(f"{donor} detected: {result.cell_count} cells")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def plot_condition_histograms(comparison: ConditionComparison) -> None:
    sns.set_theme(**DEFAULT_SEABORN_THEME)
    non_empty = [
        comparison.donor_results[donor].mean_intensities
        for donor in comparison.donors
        if comparison.donor_results[donor].mean_intensities.size > 0
    ]
    if not non_empty:
        raise ValueError("No cells were detected, so there is nothing to plot.")
    pooled = np.concatenate(non_empty)
    minimum = float(pooled.min())
    maximum = float(pooled.max())
    bins = np.linspace(minimum, minimum + 1.0, 41) if minimum == maximum else np.linspace(minimum, maximum, 41)
    fig, axes = plt.subplots(nrows=len(comparison.donors), ncols=1, figsize=(10, 4 * len(comparison.donors)), sharex=True)
    if len(comparison.donors) == 1:
        axes = [axes]
    for row, donor in enumerate(comparison.donors):
        result = comparison.donor_results[donor]
        sns.histplot(
            result.mean_intensities,
            bins=bins,
            color=DEFAULT_DONOR_COLORS.get(donor, "darkorange"),
            edgecolor="black",
            alpha=0.65,
            ax=axes[row],
        )
        axes[row].axvline(result.overall_mean, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {result.overall_mean:.2f}")
        axes[row].axvline(result.overall_median, color="blue", linestyle="dashed", linewidth=2, label=f"Median: {result.overall_median:.2f}")
        axes[row].set_title(f"{donor} brightness distribution")
        axes[row].set_ylabel("Number of cells")
        axes[row].legend()
        axes[row].grid(axis="y", alpha=0.3)
        sns.despine(ax=axes[row])
    axes[-1].set_xlabel(f"Mean brightness per cell ({comparison.channel})")
    fig.suptitle(f"Cell brightness comparison for condition {comparison.condition}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    for donor in comparison.donors:
        result = comparison.donor_results[donor]
        print(f"{donor} | cells analyzed: {result.cell_count} | mean: {result.overall_mean:.2f} | median: {result.overall_median:.2f}")


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
    resolved_title_value_label = title_value_label or default_measurement_title_label(
        analysis.measurement_scale
    )
    stat_table = build_stat_summary_table(
        analysis,
        plot_log_scale=plot_log_scale,
        min_positive_intensity=min_positive_intensity,
    )
    for antibody in analysis.antibody_order:
        plot_df, sample_df, grouped_values = _build_plot_data(
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
) -> dict[str, np.ndarray]:
    sns.set_theme(**DEFAULT_SEABORN_THEME)

    def summarize(measurement: MeasurementResult) -> float:
        if summary_stat == "median":
            return measurement.overall_median
        if summary_stat == "mean":
            return measurement.overall_mean
        raise ValueError(f"Unsupported summary_stat: {summary_stat}")

    resolved_condition_labels = {
        condition: condition_labels[condition]
        if condition_labels is not None and condition in condition_labels
        else default_condition_display_label(condition)
        for condition in analysis.conditions
    }
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

    heatmap_matrices: dict[str, np.ndarray] = {}
    all_fold_changes: list[float] = []
    for donor in analysis.donors:
        donor_matrix = []
        for condition in analysis.conditions:
            row = []
            for antibody in analysis.antibody_order:
                baseline_measurement = analysis.results[(antibody, analysis.baseline_condition, donor)]
                current_measurement = analysis.results[(antibody, condition, donor)]
                baseline_value = summarize(baseline_measurement)
                current_value = summarize(current_measurement)
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
            donor_matrix.append(row)
        heatmap_matrices[donor] = np.array(donor_matrix, dtype=float)

    color_limit = max(1.0, float(np.nanmax(np.abs(all_fold_changes)))) if all_fold_changes else 1.0
    fig, axes = plt.subplots(1, len(analysis.donors), figsize=(5.5 * len(analysis.donors), 6), constrained_layout=True)
    if len(analysis.donors) == 1:
        axes = [axes]
    for ax, donor in zip(axes, analysis.donors):
        matrix = heatmap_matrices[donor]
        annotation_labels = np.empty(matrix.shape, dtype=object)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                condition = analysis.conditions[row_index]
                antibody = analysis.antibody_order[column_index]
                value = matrix[row_index, column_index]
                label = "NA" if not np.isfinite(value) else f"{value:.2f}"
                if stat_results is not None and condition != analysis.baseline_condition:
                    match = stat_results[
                        (stat_results["donor"] == donor)
                        & (stat_results["condition"] == condition)
                        & (stat_results["antibody"] == antibody)
                    ]
                    if not match.empty:
                        stars = str(match.iloc[0]["stars"])
                        if stars not in {"baseline", "NA", "ns"}:
                            label = f"{label}\n{stars}"
                annotation_labels[row_index, column_index] = label
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-color_limit,
            vmax=color_limit,
            annot=annotation_labels,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            xticklabels=[resolved_antibody_labels[antibody] for antibody in analysis.antibody_order],
            yticklabels=[resolved_condition_labels[condition] for condition in analysis.conditions],
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
    return heatmap_matrices


def plot_condition_brightness_debug(
    analysis: BatchAnalysis,
    condition: str,
    *,
    donors: Iterable[str] | None = None,
    antibody_order: Iterable[str] | None = None,
    image_upper_percentile: float = 99.8,
    histogram_upper_percentile: float = 99.5,
    histogram_bins: int = 40,
    value_label: str | None = None,
    figure_value_label: str | None = None,
    show_background: bool = False,
) -> None:
    if condition not in analysis.conditions:
        raise ValueError(f"condition must be one of {analysis.conditions}")
    if not 0 < image_upper_percentile <= 100:
        raise ValueError("image_upper_percentile must be between 0 and 100")
    if not 0 < histogram_upper_percentile <= 100:
        raise ValueError("histogram_upper_percentile must be between 0 and 100")
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")

    donor_list = list(analysis.donors if donors is None else donors)
    antibody_list = list(analysis.antibody_order if antibody_order is None else antibody_order)
    if not donor_list:
        raise ValueError("At least one donor is required")
    if not antibody_list:
        raise ValueError("At least one antibody is required")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    condition_label = default_condition_display_label(condition)
    resolved_value_label = value_label or default_measurement_axis_label(analysis.measurement_scale)
    resolved_figure_value_label = figure_value_label or default_measurement_title_label(
        analysis.measurement_scale
    )
    histogram_limits: dict[str, tuple[float, float]] = {}
    for antibody in antibody_list:
        antibody_values = []
        for donor in donor_list:
            try:
                measurement = analysis.results[(antibody, condition, donor)]
            except KeyError as exc:
                raise KeyError(
                    f"Missing analysis result for antibody={antibody}, condition={condition}, donor={donor}"
                ) from exc
            values = np.asarray(measurement.mean_intensities, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                antibody_values.append(values)
        if antibody_values:
            pooled_values = np.concatenate(antibody_values)
            lower = min(0.0, float(pooled_values.min()))
            upper = float(np.percentile(pooled_values, histogram_upper_percentile))
            if not np.isfinite(upper) or upper <= lower:
                upper = float(pooled_values.max())
            if not np.isfinite(upper) or upper <= lower:
                upper = lower + 1.0
        else:
            lower, upper = 0.0, 1.0
        histogram_limits[antibody] = (lower, upper)

    fig, axes = plt.subplots(
        nrows=len(donor_list) * 2,
        ncols=len(antibody_list),
        figsize=(3.3 * len(antibody_list), 4.2 * len(donor_list)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{condition_label}: source image and {resolved_figure_value_label} by marker",
        fontsize=14,
    )
    for donor_index, donor in enumerate(donor_list):
        image_row = donor_index * 2
        histogram_row = image_row + 1
        for column_index, antibody in enumerate(antibody_list):
            measurement = analysis.results[(antibody, condition, donor)]
            image = load_grayscale_tif(measurement.path)
            finite_pixels = image[np.isfinite(image)]
            if finite_pixels.size:
                vmin = float(np.percentile(finite_pixels, 1.0))
                vmax = float(np.percentile(finite_pixels, image_upper_percentile))
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmin = vmax = None
            else:
                vmin = vmax = None

            image_ax = axes[image_row, column_index]
            image_ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            image_ax.axis("off")
            image_title = f"{antibody}\n{_wrap_filename(measurement.path.name)}"
            if show_background and measurement.background_intensity is not None:
                image_title = f"{image_title}\nbackground={measurement.background_intensity:.2f}"
            image_ax.set_title(image_title, fontsize=7)

            values = np.asarray(measurement.mean_intensities, dtype=float)
            values = values[np.isfinite(values)]
            histogram_ax = axes[histogram_row, column_index]
            lower, upper = histogram_limits[antibody]
            bins = np.linspace(lower, upper, histogram_bins + 1)
            if values.size:
                sns.histplot(
                    values,
                    bins=bins,
                    color=analysis.donor_colors.get(donor, "darkorange"),
                    edgecolor="black",
                    alpha=0.65,
                    ax=histogram_ax,
                )
                histogram_ax.axvline(measurement.overall_median, color="black", linestyle="--", linewidth=1.2, label="median")
                histogram_ax.axvline(measurement.overall_mean, color="red", linestyle=":", linewidth=1.2, label="mean")
            else:
                histogram_ax.text(0.5, 0.5, "No detected cells", ha="center", va="center", transform=histogram_ax.transAxes)
            histogram_ax.set_xlim(lower, upper)
            histogram_ax.set_title(
                f"n={measurement.cell_count}, median={measurement.overall_median:.2f}\nmean={measurement.overall_mean:.2f}",
                fontsize=9,
            )
            histogram_ax.set_xlabel(resolved_value_label)
            histogram_ax.set_ylabel("Cells" if column_index == 0 else "")
            histogram_ax.tick_params(axis="x", labelrotation=30)
            histogram_ax.grid(axis="y", alpha=0.25)
            sns.despine(ax=histogram_ax)
    plt.show()


def plot_condition_relative_background_debug(
    analysis: BatchAnalysis,
    condition: str,
    *,
    donors: Iterable[str] | None = None,
    antibody_order: Iterable[str] | None = None,
    image_upper_percentile: float = 99.8,
    histogram_upper_percentile: float = 99.5,
    histogram_bins: int = 40,
) -> None:
    plot_condition_brightness_debug(
        analysis,
        condition,
        donors=donors,
        antibody_order=antibody_order,
        image_upper_percentile=image_upper_percentile,
        histogram_upper_percentile=histogram_upper_percentile,
        histogram_bins=histogram_bins,
        value_label="Cell signal / image background",
        figure_value_label="per-cell signal/background ratio",
        show_background=True,
    )
