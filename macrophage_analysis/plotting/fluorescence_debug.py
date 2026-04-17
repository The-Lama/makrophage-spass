from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stardist.plot import render_label

from ..analysis.extraction import load_grayscale_tif
from ..config import (
    BatchAnalysis,
    ConditionComparison,
    DEFAULT_DONOR_COLORS,
    DEFAULT_SEABORN_THEME,
)
from ..labels import (
    default_condition_display_label,
    default_measurement_axis_label,
    default_measurement_title_label,
)
from ._shared import _wrap_filename


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
    resolved_figure_value_label = figure_value_label or default_measurement_title_label(analysis.measurement_scale)
    histogram_limits: dict[str, tuple[float, float]] = {}
    for antibody in antibody_list:
        antibody_values = []
        for donor in donor_list:
            try:
                measurement = analysis.get_result(antibody, condition, donor)
            except KeyError as exc:
                raise KeyError(
                    f"Missing analysis result for antibody={antibody}, condition={condition}, donor={donor}"
                ) from exc
            values = np.asarray(measurement.measurement_values, dtype=float)
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
            measurement = analysis.get_result(antibody, condition, donor)
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

            values = np.asarray(measurement.measurement_values, dtype=float)
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
