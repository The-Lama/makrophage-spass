from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import BatchAnalysis, DEFAULT_SEABORN_THEME, default_condition_display_label
from ..analysis.morphology import prepare_morphology_scatter_points
from .utils import build_heatmap_annotation_labels
from ..analysis.tables import (
    build_morphology_fold_change_table,
    build_morphology_summary_table,
    build_morphology_table,
)


def plot_morphology_heatmap(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    baseline_condition: str | None = None,
    summary_stat: str = "median",
    per_donor: bool = False,
    show_significance_stars: bool = True,
    exclude_treatment: str | None = None,
    condition_label_style: str = "short",
    condition_label_wrap_width: int | None = None,
) -> pd.DataFrame:
    fold_change_df = build_morphology_fold_change_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
        baseline_condition=baseline_condition,
        summary_stat=summary_stat,
        per_donor=per_donor,
    )
    available_conditions = list(fold_change_df["condition"].astype(str).unique())
    if exclude_treatment is not None:
        if exclude_treatment not in available_conditions:
            raise ValueError(f"exclude_treatment must be one of {available_conditions}")
        fold_change_df = fold_change_df.loc[
            fold_change_df["condition"].astype(str) != exclude_treatment
        ].copy()
    if fold_change_df.empty:
        raise ValueError("At least one treatment must remain in the heatmap")
    resolved_condition_label_wrap_width = (
        22
        if condition_label_style == "descriptive" and condition_label_wrap_width is None
        else condition_label_wrap_width
    )
    resolved_baseline_condition = baseline_condition or analysis.baseline_condition
    baseline_label = default_condition_display_label(
        resolved_baseline_condition,
        style=condition_label_style,
    )
    heatmap_columns = {
        "log2_fc_area": "Area",
        "log2_fc_eccentricity": "Eccentricity",
        "log2_fc_cd206_brightness": f"{antibody} brightness",
    }
    heatmap_metrics = {
        "log2_fc_area": "area",
        "log2_fc_eccentricity": "eccentricity",
        "log2_fc_cd206_brightness": "cd206_brightness",
    }
    heatmap_metric_keys = list(heatmap_columns)
    if per_donor:
        comparison_label = f"each donor's {baseline_label}"
        donor_order = [
            donor for donor in list(fold_change_df["donor"].cat.categories) if donor in set(fold_change_df["donor"].astype(str))
        ]
        condition_order = [
            condition
            for condition in list(fold_change_df["condition"].cat.categories)
            if condition in set(fold_change_df["condition"].astype(str))
        ]
        finite_values = fold_change_df[list(heatmap_columns)].to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        color_limit = max(1.0, float(np.max(np.abs(finite_values)))) if finite_values.size else 1.0
        sns.set_theme(**DEFAULT_SEABORN_THEME)
        fig, axes = plt.subplots(
            1,
            len(donor_order),
            figsize=(5.5 * len(donor_order), max(5.5, 0.35 * len(condition_order) + 2)),
            sharey=True,
            constrained_layout=True,
        )
        if len(donor_order) == 1:
            axes = [axes]
        for ax, donor in zip(axes, donor_order):
            donor_df = fold_change_df.loc[fold_change_df["donor"].astype(str) == donor].set_index("condition").reindex(condition_order)
            heatmap_data = donor_df[list(heatmap_columns)].rename(columns=heatmap_columns)
            annotation_labels = build_heatmap_annotation_labels(
                heatmap_data.to_numpy(dtype=float),
                row_labels=condition_order,
                column_labels=heatmap_metric_keys,
                star_getter=(
                    lambda row_index, column_index, donor_df=donor_df, heatmap_metric_keys=heatmap_metric_keys: str(
                        donor_df.iloc[row_index].get(
                            f"stars_{heatmap_metrics[heatmap_metric_keys[column_index]]}",
                            "NA",
                        )
                    )
                    if show_significance_stars
                    else None
                ),
            )
            donor_label = str(donor_df["donor_label"].dropna().iloc[0])
            sns.heatmap(
                heatmap_data,
                ax=ax,
                cmap="vlag",
                center=0,
                vmin=-color_limit,
                vmax=color_limit,
                annot=annotation_labels,
                fmt="",
                linewidths=0.5,
                linecolor="white",
                xticklabels=list(heatmap_columns.values()),
                yticklabels=[
                    default_condition_display_label(
                        condition,
                        style=condition_label_style,
                        wrap_width=resolved_condition_label_wrap_width,
                    )
                    for condition in condition_order
                ],
                cbar=ax is axes[-1],
                cbar_kws={"label": f"Log2 fold change vs {baseline_label}"},
            )
            ax.set_title(donor_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Treatment" if ax is axes[0] else "")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=0)
            sns.despine(ax=ax, left=True, bottom=True)
        fig.suptitle(
            f"Morphology fingerprint by donor\n{summary_stat.capitalize()} per-cell values vs {comparison_label}",
            fontsize=14,
        )
        plt.show()
        return fold_change_df

    heatmap_data = (
        fold_change_df.assign(
            display_condition_label=fold_change_df["condition"].astype(str).map(
                lambda condition: default_condition_display_label(
                    condition,
                    style=condition_label_style,
                    wrap_width=resolved_condition_label_wrap_width,
                )
            )
        )
        .set_index("display_condition_label")[list(heatmap_columns)]
        .rename(columns=heatmap_columns)
    )
    finite_values = heatmap_data.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    color_limit = max(1.0, float(np.max(np.abs(finite_values)))) if finite_values.size else 1.0
    sns.set_theme(**DEFAULT_SEABORN_THEME)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    annotation_labels = build_heatmap_annotation_labels(
        heatmap_data.to_numpy(dtype=float),
        row_labels=list(heatmap_data.index),
        column_labels=list(heatmap_data.columns),
    )
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        annot=annotation_labels,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Log2 fold change vs {baseline_label}"},
    )
    ax.set_title(
        f"Morphology and {antibody} response\n{summary_stat.capitalize()} per-cell values vs {baseline_label}",
        fontsize=14,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Treatment")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.show()
    return fold_change_df


def plot_area_intensity_scatter(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    max_points_per_group: int | None = 1000,
    random_seed: int = 7,
    intensity_upper_percentile: float = 99.0,
    area_upper_percentile: float = 99.0,
    show_donor_style: bool = True,
    split_by_donor: bool = False,
) -> pd.DataFrame:
    if not 0 < intensity_upper_percentile <= 100:
        raise ValueError("intensity_upper_percentile must be between 0 and 100")
    if not 0 < area_upper_percentile <= 100:
        raise ValueError("area_upper_percentile must be between 0 and 100")
    morphology_df = build_morphology_table(analysis, antibody=antibody, donors=donors, conditions=conditions)
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")
    summary_df = build_morphology_summary_table(analysis, antibody=antibody, donors=donors, conditions=conditions)
    plot_df = prepare_morphology_scatter_points(
        morphology_df,
        max_points_per_group=max_points_per_group,
        random_seed=random_seed,
        x_column="intensity",
        x_upper_percentile=intensity_upper_percentile,
        area_upper_percentile=area_upper_percentile,
    )
    if plot_df.empty:
        raise ValueError("No morphology points remain after display percentile filtering")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    condition_label_order = list(summary_df["condition_label"].cat.categories)
    donor_label_order = list(summary_df["donor_label"].cat.categories)
    if split_by_donor:
        donor_label_order = [label for label in donor_label_order if (plot_df["donor_label"] == label).any()]
        fig, axes = plt.subplots(1, len(donor_label_order), figsize=(6.5 * len(donor_label_order), 5.8), sharex=True, sharey=True, constrained_layout=True)
        if len(donor_label_order) == 1:
            axes = [axes]
        for ax, donor_label in zip(axes, donor_label_order):
            donor_plot_df = plot_df.loc[plot_df["donor_label"] == donor_label]
            donor_summary_df = summary_df.loc[summary_df["donor_label"] == donor_label]
            sns.scatterplot(data=donor_plot_df, x="intensity", y="area", hue="condition_label", hue_order=condition_label_order, alpha=0.18, s=14, linewidth=0, legend=False, ax=ax)
            sns.scatterplot(data=donor_summary_df, x="median_intensity", y="median_area", hue="condition_label", hue_order=condition_label_order, alpha=1.0, s=165, linewidth=1.1, edgecolor="black", legend="brief" if ax is axes[-1] else False, ax=ax)
            ax.set_title(str(donor_label), fontsize=12, fontweight="bold")
            ax.set_xlabel(f"Per-cell mean {antibody} fluorescence intensity")
            ax.set_ylabel("Cell area (pixels)" if ax is axes[0] else "")
            ax.grid(alpha=0.25)
            sns.despine(ax=ax)
        legend = axes[-1].get_legend()
        if legend is not None:
            legend.set_title("Median marker: treatment")
        fig.suptitle(f"{antibody} vs Cell Area by Donor", fontsize=14)
        plt.show()
        return morphology_df

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    cloud_kwargs = {"data": plot_df, "x": "intensity", "y": "area", "hue": "condition_label", "hue_order": condition_label_order, "alpha": 0.18, "s": 14, "linewidth": 0, "legend": False, "ax": ax}
    median_kwargs = {"data": summary_df, "x": "median_intensity", "y": "median_area", "hue": "condition_label", "hue_order": condition_label_order, "alpha": 1.0, "s": 165, "linewidth": 1.1, "edgecolor": "black", "ax": ax}
    if show_donor_style:
        cloud_kwargs["style"] = "donor_label"
        cloud_kwargs["style_order"] = donor_label_order
        median_kwargs["style"] = "donor_label"
        median_kwargs["style_order"] = donor_label_order
    sns.scatterplot(**cloud_kwargs)
    sns.scatterplot(**median_kwargs)
    ax.set_title(f"{antibody} vs Cell Area", fontsize=14, pad=12)
    ax.set_xlabel(f"Per-cell mean {antibody} fluorescence intensity")
    ax.set_ylabel("Cell area (pixels)")
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Median marker: treatment" + (" / donor" if show_donor_style else ""))
    plt.show()
    return morphology_df


def plot_morphology_scatter(
    analysis: BatchAnalysis,
    *,
    antibody: str = "CD206",
    donors: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    max_points_per_group: int = 800,
    random_seed: int = 7,
    x_upper_percentile: float = 99.0,
    area_upper_percentile: float = 99.0,
) -> pd.DataFrame:
    if not 0 < x_upper_percentile <= 100:
        raise ValueError("x_upper_percentile must be between 0 and 100")
    if not 0 < area_upper_percentile <= 100:
        raise ValueError("area_upper_percentile must be between 0 and 100")
    morphology_df = build_morphology_table(analysis, antibody=antibody, donors=donors, conditions=conditions)
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")
    plot_df = prepare_morphology_scatter_points(
        morphology_df,
        max_points_per_group=max_points_per_group,
        random_seed=random_seed,
        x_column="intensity",
        x_upper_percentile=x_upper_percentile,
        area_upper_percentile=area_upper_percentile,
    )
    if plot_df.empty:
        raise ValueError("No morphology points remain after display percentile filtering")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    donor_labels = list(plot_df["donor_label"].cat.categories)
    fig, axes = plt.subplots(1, len(donor_labels), figsize=(6.5 * len(donor_labels), 5.5), sharex=True, sharey=True, constrained_layout=True)
    if len(donor_labels) == 1:
        axes = [axes]
    for ax, donor_label in zip(axes, donor_labels):
        donor_plot_df = plot_df.loc[plot_df["donor_label"] == donor_label]
        sns.scatterplot(
            data=donor_plot_df,
            x="intensity",
            y="eccentricity",
            hue="condition_label",
            size="area",
            sizes=(8, 45),
            alpha=0.35,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(str(donor_label), fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{antibody} mean fluorescence intensity")
        ax.set_ylabel("Cell eccentricity")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        sns.despine(ax=ax)
        if ax is not axes[-1]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    legend = axes[-1].get_legend()
    if legend is not None:
        legend.set_title("Treatment / area")
    fig.suptitle(
        f"{antibody} intensity vs cell shape\nPoint size is StarDist cell area; values above p{x_upper_percentile:g} intensity or p{area_upper_percentile:g} area are hidden for display.",
        fontsize=14,
    )
    plt.show()
    return morphology_df
