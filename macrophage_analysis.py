from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from csbdeep.utils import normalize
from scipy.stats import mannwhitneyu
from skimage.measure import regionprops
from stardist.models import StarDist2D
from stardist.plot import render_label

DEFAULT_DATA_ROOT = Path("raw_data")
DEFAULT_BASELINE_CONDITION = "M0"
DEFAULT_DONORS = ("D45", "D47")
DEFAULT_CONDITIONS = (
    "M0",
    "B68KCP2",
    "B75palmFP1",
    "M130227Ni",
    "M130429Np",
    "T12CMNepiP4",
    "T8CMNdermP6",
)
DEFAULT_COMPARISON_MARKER_PREFIX = "CD200R_A488_CD206_A568_10x_RGB_Qi2 10x TRITC"
# DEFAULT_COMPARISON_MARKER_PREFIX = "CD200R_A488_CD206_A568_10x_RGB_Qi2 10x"
DEFAULT_COMPARISON_CHANNEL = "TRITC"
DEFAULT_DONOR_COLORS = {
    "D45": "#0072B2",
    "D47": "#D55E00",
}
DEFAULT_SEABORN_THEME = {"style": "whitegrid"}
DEFAULT_DONOR_DISPLAY_LABELS = {donor: f"Donor {donor}" for donor in DEFAULT_DONORS}
DEFAULT_CONDITION_DISPLAY_LABELS = {
    "M0": "M0 (baseline)",
    "B68KCP2": "B68 KCP2",
    "B75palmFP1": "B75 palm FP1",
    "M130227Ni": "M130227 Ni",
    "M130429Np": "M130429 Np",
    "T12CMNepiP4": "T12 CMN epi P4",
    "T8CMNdermP6": "T8 CMN derm P6",
}


@dataclass(frozen=True)
class AntibodySpec:
    marker_prefix: str
    channel: str


DEFAULT_ANTIBODY_SPECS: dict[str, AntibodySpec] = {
    "CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC"),
    "CD86": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "FITC"),
    "iNOS": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "TRITC"),
    "CD200R": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "FITC"),
    "CD206": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "TRITC"),
}
DEFAULT_ANTIBODY_ORDER = tuple(DEFAULT_ANTIBODY_SPECS.keys())


@dataclass
class ComparisonDonorResult:
    path: Path
    image: np.ndarray
    labels: np.ndarray
    details: dict
    mean_intensities: np.ndarray
    cell_count: int
    overall_mean: float
    overall_median: float


@dataclass
class ConditionComparison:
    condition: str
    donors: list[str]
    marker_prefix: str
    channel: str
    donor_results: dict[str, ComparisonDonorResult]


@dataclass
class MeasurementResult:
    path: Path
    cell_count: int
    mean_intensities: np.ndarray
    overall_mean: float
    overall_median: float


@dataclass
class BatchAnalysis:
    donors: list[str]
    conditions: list[str]
    baseline_condition: str
    antibody_order: list[str]
    antibody_specs: dict[str, AntibodySpec]
    donor_colors: dict[str, str]
    results: dict[tuple[str, str, str], MeasurementResult]


@lru_cache(maxsize=1)
def load_stardist_model(model_name: str = "2D_versatile_fluo") -> StarDist2D:
    return StarDist2D.from_pretrained(model_name)


def find_image_path(
    donor: str,
    condition: str,
    marker_prefix: str,
    channel: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Path:
    condition_dir = Path(data_root) / donor / condition
    matches = sorted(condition_dir.glob(f"{donor}_{condition}_{marker_prefix} {channel}.tif*"))
    if not matches:
        raise FileNotFoundError(
            f"Missing expected image in {condition_dir} for {marker_prefix} {channel}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple matching images for {donor} / {condition} / {marker_prefix} / {channel}: {matches}"
        )
    return matches[0]


def load_grayscale_tif(image_path: str | Path) -> np.ndarray:
    tifffile_logger = logging.getLogger("tifffile")
    original_tifffile_level = tifffile_logger.level

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*OME series cannot handle discontiguous storage.*",
        )
        tifffile_logger.setLevel(logging.ERROR)
        try:
            image = tiff.imread(Path(image_path))
        finally:
            tifffile_logger.setLevel(original_tifffile_level)

    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def _extract_mean_intensities(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    props = regionprops(labels, intensity_image=image)
    return np.array([cell.intensity_mean for cell in props], dtype=float)


def _summarize_intensities(mean_intensities: np.ndarray) -> tuple[int, float, float]:
    count = int(mean_intensities.size)
    overall_mean = float(mean_intensities.mean()) if count else float("nan")
    overall_median = float(np.median(mean_intensities)) if count else float("nan")
    return count, overall_mean, overall_median


def extract_condition_comparison(
    donors: Iterable[str],
    condition: str,
    marker_prefix: str = DEFAULT_COMPARISON_MARKER_PREFIX,
    channel: str = DEFAULT_COMPARISON_CHANNEL,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    model_name: str = "2D_versatile_fluo",
) -> ConditionComparison:
    donor_list = list(donors)
    model = load_stardist_model(model_name)
    donor_results: dict[str, ComparisonDonorResult] = {}

    for donor in donor_list:
        image_path = find_image_path(
            donor=donor,
            condition=condition,
            marker_prefix=marker_prefix,
            channel=channel,
            data_root=data_root,
        )
        image = load_grayscale_tif(image_path)
        normalized_image = normalize(image, 1, 99.8, axis=(0, 1))
        labels, details = model.predict_instances(normalized_image)
        mean_intensities = _extract_mean_intensities(image, labels)
        cell_count, overall_mean, overall_median = _summarize_intensities(mean_intensities)

        donor_results[donor] = ComparisonDonorResult(
            path=image_path,
            image=image,
            labels=labels,
            details=details,
            mean_intensities=mean_intensities,
            cell_count=cell_count,
            overall_mean=overall_mean,
            overall_median=overall_median,
        )

    return ConditionComparison(
        condition=condition,
        donors=donor_list,
        marker_prefix=marker_prefix,
        channel=channel,
        donor_results=donor_results,
    )


def plot_condition_comparison(comparison: ConditionComparison) -> None:
    fig, axes = plt.subplots(nrows=len(comparison.donors), ncols=2, figsize=(14, 5 * len(comparison.donors)))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        f"StarDist comparison for condition {comparison.condition} ({comparison.channel})",
        fontsize=14,
    )

    for row, donor in enumerate(comparison.donors):
        result = comparison.donor_results[donor]
        axes[row, 0].imshow(result.image, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{donor} original")

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
    if minimum == maximum:
        bins = np.linspace(minimum, minimum + 1.0, 41)
    else:
        bins = np.linspace(minimum, maximum, 41)

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
        axes[row].axvline(
            result.overall_mean,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {result.overall_mean:.2f}",
        )
        axes[row].axvline(
            result.overall_median,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {result.overall_median:.2f}",
        )
        axes[row].set_title(f"{donor} brightness distribution")
        axes[row].set_ylabel("Number of cells")
        axes[row].legend()
        axes[row].grid(axis="y", alpha=0.3)
        sns.despine(ax=axes[row])

    axes[-1].set_xlabel(f"Mean brightness per cell ({comparison.channel})")
    fig.suptitle(
        f"Cell brightness comparison for condition {comparison.condition}",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

    for donor in comparison.donors:
        result = comparison.donor_results[donor]
        print(
            f"{donor} | cells analyzed: {result.cell_count} | "
            f"mean: {result.overall_mean:.2f} | median: {result.overall_median:.2f}"
        )


def extract_single_cell_fluorescence(
    donors: Iterable[str],
    conditions: Iterable[str],
    *,
    antibody_specs: Mapping[str, AntibodySpec] = DEFAULT_ANTIBODY_SPECS,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    donor_colors: Mapping[str, str] = DEFAULT_DONOR_COLORS,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    model_name: str = "2D_versatile_fluo",
) -> BatchAnalysis:
    donor_list = list(donors)
    condition_list = list(conditions)
    order_list = list(antibody_order)

    if baseline_condition not in condition_list:
        raise ValueError(f"baseline_condition must be one of {condition_list}")

    model = load_stardist_model(model_name)
    results: dict[tuple[str, str, str], MeasurementResult] = {}

    for antibody in order_list:
        spec = antibody_specs[antibody]
        print(f"Analyzing {antibody}...")
        for condition in condition_list:
            for donor in donor_list:
                image_path = find_image_path(
                    donor=donor,
                    condition=condition,
                    marker_prefix=spec.marker_prefix,
                    channel=spec.channel,
                    data_root=data_root,
                )
                image = load_grayscale_tif(image_path)
                normalized_image = normalize(image, 1, 99.8, axis=(0, 1))
                labels, _ = model.predict_instances(normalized_image)
                mean_intensities = _extract_mean_intensities(image, labels)
                cell_count, overall_mean, overall_median = _summarize_intensities(mean_intensities)

                results[(antibody, condition, donor)] = MeasurementResult(
                    path=image_path,
                    cell_count=cell_count,
                    mean_intensities=mean_intensities,
                    overall_mean=overall_mean,
                    overall_median=overall_median,
                )

    print(f"Finished analyzing {len(results)} donor / treatment / antibody combinations.")
    return BatchAnalysis(
        donors=donor_list,
        conditions=condition_list,
        baseline_condition=baseline_condition,
        antibody_order=order_list,
        antibody_specs=dict(antibody_specs),
        donor_colors=dict(donor_colors),
        results=results,
    )


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

            for value in plot_values:
                full_rows.append(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "donor": donor,
                        "intensity": float(value),
                    }
                )

            sampled_values = plot_values
            if sampled_values.size > max_strip_points:
                sampled_values = rng.choice(sampled_values, size=max_strip_points, replace=False)
            for value in sampled_values:
                sample_rows.append(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "donor": donor,
                        "intensity": float(value),
                    }
                )

    full_df = pd.DataFrame(full_rows)
    sample_df = pd.DataFrame(sample_rows)
    for frame in (full_df, sample_df):
        if frame.empty:
            continue
        frame["condition"] = pd.Categorical(frame["condition"], categories=analysis.conditions, ordered=True)
        frame["donor"] = pd.Categorical(frame["donor"], categories=analysis.donors, ordered=True)

    return full_df, sample_df, grouped_values


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


def mann_whitney_u_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(u_stat), float(p_value)


def _pvalue_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def _default_condition_display_label(condition: str) -> str:
    if condition in DEFAULT_CONDITION_DISPLAY_LABELS:
        return DEFAULT_CONDITION_DISPLAY_LABELS[condition]

    label = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", condition)
    label = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", label)
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return label


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
                    u_stat, p_value = mann_whitney_u_test(
                        baseline_plot_values,
                        current_plot_values,
                    )
                    row["u_stat"] = u_stat
                    row["p_value"] = p_value
                    row["stars"] = _pvalue_to_stars(p_value)

                rows.append(row)

    stat_table = pd.DataFrame(rows)
    stat_table["condition"] = pd.Categorical(
        stat_table["condition"], categories=analysis.conditions, ordered=True
    )
    stat_table["donor"] = pd.Categorical(
        stat_table["donor"], categories=analysis.donors, ordered=True
    )
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
        condition: (
            condition_labels[condition]
            if condition_labels is not None and condition in condition_labels
            else _default_condition_display_label(condition)
        )
        for condition in stat_table["condition"].astype(str).unique()
    }
    resolved_donor_labels = {
        donor: (
            donor_labels[donor]
            if donor_labels is not None and donor in donor_labels
            else DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor)
        )
        for donor in stat_table["donor"].astype(str).unique()
    }

    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown = None
        display = None

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

    for antibody in antibody_order:
        antibody_stats = stat_table.loc[
            stat_table["antibody"] == antibody,
            display_columns,
        ].copy()
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
            lambda row: (
                "baseline"
                if str(row["condition"]) == resolved_condition_labels.get(baseline_condition, baseline_condition)
                else f"{row['p_value']:.3g}"
            ),
            axis=1,
        )
        formatted_stats = antibody_stats.rename(columns=renamed_columns)

        if Markdown is not None and display is not None:
            display(Markdown(f"**{antibody}**"))
            display(Markdown(f"```text\n{formatted_stats.to_string(index=False)}\n```"))
        else:
            print(f"\n{antibody}")
            print(formatted_stats.to_string(index=False))


def plot_violins_with_stats(
    analysis: BatchAnalysis,
    *,
    max_strip_points: int = 250,
    plot_log_scale: bool = False,
    min_positive_intensity: float | None = None,
    upper_display_percentile: float = 99.5,
    random_seed: int = 7,
) -> pd.DataFrame:
    sns.set_theme(**DEFAULT_SEABORN_THEME)
    palette = {donor: analysis.donor_colors[donor] for donor in analysis.donors}
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

        display_max = max(
            float(np.percentile(values, upper_display_percentile))
            for values in grouped_values.values()
            if values.size > 0
        )
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

        ax.set_title(f"{antibody} single-cell fluorescence by treatment")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Per-cell fluorescence intensity" + (" (log scale)" if plot_log_scale else ""))
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
        condition: (
            condition_labels[condition]
            if condition_labels is not None and condition in condition_labels
            else _default_condition_display_label(condition)
        )
        for condition in analysis.conditions
    }
    resolved_antibody_labels = {
        antibody: (
            antibody_labels[antibody]
            if antibody_labels is not None and antibody in antibody_labels
            else antibody
        )
        for antibody in analysis.antibody_order
    }
    resolved_donor_labels = {
        donor: (
            donor_labels[donor]
            if donor_labels is not None and donor in donor_labels
            else DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor)
        )
        for donor in analysis.donors
    }
    baseline_label = resolved_condition_labels.get(analysis.baseline_condition, analysis.baseline_condition)
    summary_stat_label = f"{summary_stat} single-cell intensity"

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
    fig, axes = plt.subplots(
        1,
        len(analysis.donors),
        figsize=(5.5 * len(analysis.donors), 6),
        constrained_layout=True,
    )
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
        f"Phenotype fingerprint by donor\nLog2 fold change vs {baseline_label} ({summary_stat_label})",
        fontsize=14,
    )
    plt.show()
    return heatmap_matrices


__all__ = [
    "AntibodySpec",
    "BatchAnalysis",
    "ComparisonDonorResult",
    "ConditionComparison",
    "DEFAULT_ANTIBODY_ORDER",
    "DEFAULT_ANTIBODY_SPECS",
    "DEFAULT_BASELINE_CONDITION",
    "DEFAULT_COMPARISON_CHANNEL",
    "DEFAULT_COMPARISON_MARKER_PREFIX",
    "DEFAULT_CONDITIONS",
    "DEFAULT_CONDITION_DISPLAY_LABELS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_DONOR_COLORS",
    "DEFAULT_DONORS",
    "DEFAULT_DONOR_DISPLAY_LABELS",
    "MeasurementResult",
    "build_stat_summary_table",
    "display_stat_summary_tables",
    "extract_condition_comparison",
    "extract_single_cell_fluorescence",
    "find_image_path",
    "load_grayscale_tif",
    "load_stardist_model",
    "mann_whitney_u_test",
    "plot_condition_comparison",
    "plot_condition_histograms",
    "plot_summary_heatmap",
    "plot_violins_with_stats",
]
