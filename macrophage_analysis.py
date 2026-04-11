from __future__ import annotations

import logging
import re
import textwrap
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
DEFAULT_COMPARISON_MARKER_PREFIX = "CD200R_A488_CD206_A568_10x_RGB_Qi2 10x"
DEFAULT_COMPARISON_CHANNEL = "TRITC"
DEFAULT_STARDIST_N_TILES = (2, 2)
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
    areas: np.ndarray | None = None
    eccentricities: np.ndarray | None = None


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
    areas: np.ndarray | None = None
    eccentricities: np.ndarray | None = None
    background_intensity: float | None = None
    background_percentile: float | None = None


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


def _extract_cell_measurements(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = regionprops(labels, intensity_image=image)
    mean_intensities = np.array([cell.intensity_mean for cell in props], dtype=float)
    areas = np.array([cell.area for cell in props], dtype=float)
    eccentricities = np.array([cell.eccentricity for cell in props], dtype=float)
    return mean_intensities, areas, eccentricities


def _extract_mean_intensities(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mean_intensities, _, _ = _extract_cell_measurements(image, labels)
    return mean_intensities


def _summarize_intensities(mean_intensities: np.ndarray) -> tuple[int, float, float]:
    count = int(mean_intensities.size)
    finite_intensities = np.asarray(mean_intensities, dtype=float)
    finite_intensities = finite_intensities[np.isfinite(finite_intensities)]
    overall_mean = float(finite_intensities.mean()) if finite_intensities.size else float("nan")
    overall_median = float(np.median(finite_intensities)) if finite_intensities.size else float("nan")
    return count, overall_mean, overall_median


def _estimate_background_intensity(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    background_percentile: float,
) -> float:
    background_pixels = np.asarray(image[labels == 0], dtype=float)
    background_pixels = background_pixels[np.isfinite(background_pixels)]
    if background_pixels.size == 0:
        image_pixels = np.asarray(image, dtype=float)
        background_pixels = image_pixels[np.isfinite(image_pixels)]
    if background_pixels.size == 0:
        return float("nan")
    return float(np.percentile(background_pixels, background_percentile))


def _divide_by_background(values: np.ndarray, background_intensity: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(background_intensity) or background_intensity <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return values / background_intensity


def _wrap_filename(filename: str, *, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(filename, width=width, break_long_words=True))


def extract_condition_comparison(
    donors: Iterable[str],
    condition: str,
    marker_prefix: str = DEFAULT_COMPARISON_MARKER_PREFIX,
    channel: str = DEFAULT_COMPARISON_CHANNEL,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    model_name: str = "2D_versatile_fluo",
    n_tiles: tuple[int, int] | None = DEFAULT_STARDIST_N_TILES,
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
        labels, details = model.predict_instances(normalized_image, n_tiles=n_tiles)
        mean_intensities, areas, eccentricities = _extract_cell_measurements(image, labels)
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
            areas=areas,
            eccentricities=eccentricities,
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
    n_tiles: tuple[int, int] | None = DEFAULT_STARDIST_N_TILES,
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
                labels, _ = model.predict_instances(normalized_image, n_tiles=n_tiles)
                mean_intensities, areas, eccentricities = _extract_cell_measurements(image, labels)
                cell_count, overall_mean, overall_median = _summarize_intensities(mean_intensities)

                results[(antibody, condition, donor)] = MeasurementResult(
                    path=image_path,
                    cell_count=cell_count,
                    mean_intensities=mean_intensities,
                    overall_mean=overall_mean,
                    overall_median=overall_median,
                    areas=areas,
                    eccentricities=eccentricities,
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


def extract_single_cell_relative_fluorescence(
    donors: Iterable[str],
    conditions: Iterable[str],
    *,
    antibody_specs: Mapping[str, AntibodySpec] = DEFAULT_ANTIBODY_SPECS,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    donor_colors: Mapping[str, str] = DEFAULT_DONOR_COLORS,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    model_name: str = "2D_versatile_fluo",
    n_tiles: tuple[int, int] | None = DEFAULT_STARDIST_N_TILES,
    background_percentile: float = 50.0,
) -> BatchAnalysis:
    if not 0 <= background_percentile <= 100:
        raise ValueError("background_percentile must be between 0 and 100")

    donor_list = list(donors)
    condition_list = list(conditions)
    order_list = list(antibody_order)

    if baseline_condition not in condition_list:
        raise ValueError(f"baseline_condition must be one of {condition_list}")

    model = load_stardist_model(model_name)
    results: dict[tuple[str, str, str], MeasurementResult] = {}

    for antibody in order_list:
        spec = antibody_specs[antibody]
        print(f"Analyzing {antibody} relative to image background...")
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
                labels, _ = model.predict_instances(normalized_image, n_tiles=n_tiles)
                mean_intensities, areas, eccentricities = _extract_cell_measurements(image, labels)
                background_intensity = _estimate_background_intensity(
                    image,
                    labels,
                    background_percentile=background_percentile,
                )
                relative_intensities = _divide_by_background(
                    mean_intensities,
                    background_intensity,
                )
                cell_count, overall_mean, overall_median = _summarize_intensities(
                    relative_intensities
                )

                results[(antibody, condition, donor)] = MeasurementResult(
                    path=image_path,
                    cell_count=cell_count,
                    mean_intensities=relative_intensities,
                    overall_mean=overall_mean,
                    overall_median=overall_median,
                    areas=areas,
                    eccentricities=eccentricities,
                    background_intensity=background_intensity,
                    background_percentile=background_percentile,
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
    y_label: str = "Per-cell fluorescence intensity",
    title_value_label: str = "single-cell fluorescence",
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

        ax.set_title(f"{antibody} {title_value_label} by treatment")
        ax.set_xlabel("Treatment")
        ax.set_ylabel(y_label + (" (log scale)" if plot_log_scale else ""))
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
    resolved_summary_stat_label = (
        summary_stat_label if summary_stat_label is not None else f"{summary_stat} single-cell intensity"
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
    value_label: str = "Mean cell brightness",
    figure_value_label: str = "per-cell brightness",
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
    condition_label = _default_condition_display_label(condition)
    histogram_limits: dict[str, tuple[float, float]] = {}

    for antibody in antibody_list:
        antibody_values = []
        for donor in donor_list:
            try:
                measurement = analysis.results[(antibody, condition, donor)]
            except KeyError as exc:
                raise KeyError(
                    f"Missing analysis result for antibody={antibody}, "
                    f"condition={condition}, donor={donor}"
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
        f"{condition_label}: source image and {figure_value_label} by marker",
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
                image_title = (
                    f"{image_title}\n"
                    f"background={measurement.background_intensity:.2f}"
                )
            image_ax.set_title(
                image_title,
                fontsize=7,
            )

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
                histogram_ax.axvline(
                    measurement.overall_median,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="median",
                )
                histogram_ax.axvline(
                    measurement.overall_mean,
                    color="red",
                    linestyle=":",
                    linewidth=1.2,
                    label="mean",
                )
            else:
                histogram_ax.text(
                    0.5,
                    0.5,
                    "No detected cells",
                    ha="center",
                    va="center",
                    transform=histogram_ax.transAxes,
                )

            histogram_ax.set_xlim(lower, upper)
            histogram_ax.set_title(
                f"n={measurement.cell_count}, median={measurement.overall_median:.2f}\n"
                f"mean={measurement.overall_mean:.2f}",
                fontsize=9,
            )
            histogram_ax.set_xlabel(value_label)
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
                    "Rerun extract_single_cell_fluorescence after updating macrophage_analysis.py."
                )

            intensities = np.asarray(measurement.mean_intensities, dtype=float)
            areas = np.asarray(measurement.areas, dtype=float)
            eccentricities = np.asarray(measurement.eccentricities, dtype=float)
            if not (intensities.size == areas.size == eccentricities.size):
                raise ValueError(
                    f"Mismatched cell metrics for {antibody} / {condition} / {donor}"
                )

            valid = np.isfinite(intensities) & np.isfinite(areas) & np.isfinite(eccentricities)
            for cell_index in np.flatnonzero(valid):
                rows.append(
                    {
                        "antibody": antibody,
                        "condition": condition,
                        "condition_label": _default_condition_display_label(condition),
                        "donor": donor,
                        "donor_label": DEFAULT_DONOR_DISPLAY_LABELS.get(donor, donor),
                        "cell_index": int(cell_index),
                        "intensity": float(intensities[cell_index]),
                        "area": float(areas[cell_index]),
                        "eccentricity": float(eccentricities[cell_index]),
                        "path": str(measurement.path),
                        "filename": measurement.path.name,
                    }
                )

    morphology_df = pd.DataFrame(rows)
    if morphology_df.empty:
        return morphology_df

    morphology_df["condition"] = pd.Categorical(
        morphology_df["condition"], categories=condition_list, ordered=True
    )
    morphology_df["condition_label"] = pd.Categorical(
        morphology_df["condition_label"],
        categories=[_default_condition_display_label(condition) for condition in condition_list],
        ordered=True,
    )
    morphology_df["donor"] = pd.Categorical(
        morphology_df["donor"], categories=donor_list, ordered=True
    )
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

    summary_df = (
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
    return summary_df


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
    metric_sources = {
        "area": "area",
        "eccentricity": "eccentricity",
        "cd206_brightness": "intensity",
    }

    for metric in metric_sources:
        summary_df[f"u_stat_{metric}"] = np.nan
        summary_df[f"p_value_{metric}"] = np.nan
        summary_df[f"stars_{metric}"] = "NA"

    grouping_columns = ["condition"]
    if per_donor:
        grouping_columns = ["donor", *grouping_columns]

    value_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in morphology_df.groupby(grouping_columns, observed=True)
    }

    for row_index, row in summary_df.iterrows():
        condition = str(row["condition"])
        baseline_key = (baseline_condition,)
        current_key = (condition,)
        if per_donor:
            donor = str(row["donor"])
            baseline_key = (donor, baseline_condition)
            current_key = (donor, condition)

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
            summary_df.at[row_index, stars_column] = _pvalue_to_stars(p_value)

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
    group_columns = ["condition", "condition_label"]
    if per_donor:
        group_columns = ["donor", "donor_label", *group_columns]

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
                zip(baseline_rows["donor"].astype(str), baseline_rows[metric].astype(float))
            )
            fold_change_column = f"log2_fc_{metric}"
            summary_df[fold_change_column] = summary_df.apply(
                lambda row: (
                    float(np.log2(row[metric] / donor_baselines[str(row["donor"])]))
                    if (
                        str(row["donor"]) in donor_baselines
                        and np.isfinite(row[metric])
                        and row[metric] > 0
                        and np.isfinite(donor_baselines[str(row["donor"])])
                        and donor_baselines[str(row["donor"])] > 0
                    )
                    else np.nan
                ),
                axis=1,
            )
    else:
        baseline_row = summary_df.loc[
            summary_df["condition"].astype(str) == resolved_baseline_condition
        ].iloc[0]
        for metric in metrics:
            baseline_value = float(baseline_row[metric])
            fold_change_column = f"log2_fc_{metric}"
            if not np.isfinite(baseline_value) or baseline_value <= 0:
                summary_df[fold_change_column] = np.nan
            else:
                summary_df[fold_change_column] = summary_df[metric].map(
                    lambda value: (
                        float(np.log2(value / baseline_value))
                        if np.isfinite(value) and value > 0
                        else np.nan
                    )
                )

    return _add_morphology_stat_columns(
        summary_df,
        morphology_df,
        baseline_condition=resolved_baseline_condition,
        per_donor=per_donor,
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
    resolved_baseline_condition = baseline_condition or analysis.baseline_condition
    baseline_label = _default_condition_display_label(resolved_baseline_condition)
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
    if per_donor:
        comparison_label = f"each donor's {baseline_label}"
        donor_order = list(fold_change_df["donor"].cat.categories)
        donor_order = [
            donor
            for donor in donor_order
            if donor in set(fold_change_df["donor"].astype(str))
        ]
        condition_order = list(fold_change_df["condition"].cat.categories)
        condition_order = [
            condition
            for condition in condition_order
            if condition in set(fold_change_df["condition"].astype(str))
        ]
        finite_values = fold_change_df[list(heatmap_columns)].to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        color_limit = (
            max(1.0, float(np.max(np.abs(finite_values))))
            if finite_values.size
            else 1.0
        )

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
            donor_df = fold_change_df.loc[
                fold_change_df["donor"].astype(str) == donor
            ].set_index("condition")
            donor_df = donor_df.reindex(condition_order)
            heatmap_data = donor_df[list(heatmap_columns)].rename(columns=heatmap_columns)
            annotation_labels = np.empty(heatmap_data.shape, dtype=object)

            for row_index, condition in enumerate(condition_order):
                for column_index, fold_change_column in enumerate(heatmap_columns):
                    value = heatmap_data.iloc[row_index, column_index]
                    label = "NA" if not np.isfinite(value) else f"{value:.2f}"
                    metric = heatmap_metrics[fold_change_column]
                    stars = str(donor_df.iloc[row_index].get(f"stars_{metric}", "NA"))
                    if stars not in {"baseline", "NA", "ns"}:
                        label = f"{label}\n{stars}"
                    annotation_labels[row_index, column_index] = label

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
                    _default_condition_display_label(condition)
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
            f"Morphology fingerprint by donor\n"
            f"{summary_stat.capitalize()} per-cell values vs {comparison_label}",
            fontsize=14,
        )
        plt.show()
        return fold_change_df

    else:
        heatmap_data = fold_change_df.set_index("condition_label")[list(heatmap_columns)]
        y_label = "Treatment"
        comparison_label = baseline_label
        figsize = (7, 5.5)
    heatmap_data = heatmap_data.rename(columns=heatmap_columns)

    finite_values = heatmap_data.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    color_limit = max(1.0, float(np.max(np.abs(finite_values)))) if finite_values.size else 1.0

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Log2 fold change vs {comparison_label}"},
    )
    ax.set_title(
        f"Morphology and {antibody} response\n"
        f"{summary_stat.capitalize()} per-cell values vs {comparison_label}",
        fontsize=14,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel(y_label)
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
    if max_points_per_group is not None and max_points_per_group < 1:
        raise ValueError("max_points_per_group must be at least 1 or None")
    if not 0 < intensity_upper_percentile <= 100:
        raise ValueError("intensity_upper_percentile must be between 0 and 100")
    if not 0 < area_upper_percentile <= 100:
        raise ValueError("area_upper_percentile must be between 0 and 100")

    morphology_df = build_morphology_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
    )
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")

    summary_df = build_morphology_summary_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
    )

    rng = np.random.default_rng(random_seed)
    sampled_frames = []
    for _, group in morphology_df.groupby(["condition", "donor"], observed=True):
        if max_points_per_group is not None and len(group) > max_points_per_group:
            sampled_frames.append(
                group.iloc[rng.choice(len(group), size=max_points_per_group, replace=False)]
            )
        else:
            sampled_frames.append(group)
    plot_df = pd.concat(sampled_frames, ignore_index=True)

    intensity_max = float(np.percentile(plot_df["intensity"], intensity_upper_percentile))
    area_max = float(np.percentile(plot_df["area"], area_upper_percentile))
    plot_df = plot_df.loc[
        (plot_df["intensity"] <= intensity_max)
        & (plot_df["area"] <= area_max)
    ].copy()
    if plot_df.empty:
        raise ValueError("No morphology points remain after display percentile filtering")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    condition_label_order = list(summary_df["condition_label"].cat.categories)
    donor_label_order = list(summary_df["donor_label"].cat.categories)

    if split_by_donor:
        donor_label_order = [
            donor_label
            for donor_label in donor_label_order
            if (plot_df["donor_label"] == donor_label).any()
        ]
        fig, axes = plt.subplots(
            1,
            len(donor_label_order),
            figsize=(6.5 * len(donor_label_order), 5.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if len(donor_label_order) == 1:
            axes = [axes]

        for ax, donor_label in zip(axes, donor_label_order):
            donor_plot_df = plot_df.loc[plot_df["donor_label"] == donor_label]
            donor_summary_df = summary_df.loc[summary_df["donor_label"] == donor_label]
            sns.scatterplot(
                data=donor_plot_df,
                x="intensity",
                y="area",
                hue="condition_label",
                hue_order=condition_label_order,
                alpha=0.18,
                s=14,
                linewidth=0,
                legend=False,
                ax=ax,
            )
            sns.scatterplot(
                data=donor_summary_df,
                x="median_intensity",
                y="median_area",
                hue="condition_label",
                hue_order=condition_label_order,
                alpha=1.0,
                s=165,
                linewidth=1.1,
                edgecolor="black",
                legend="brief" if ax is axes[-1] else False,
                ax=ax,
            )
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
    cloud_kwargs = {
        "data": plot_df,
        "x": "intensity",
        "y": "area",
        "hue": "condition_label",
        "hue_order": condition_label_order,
        "alpha": 0.18,
        "s": 14,
        "linewidth": 0,
        "legend": False,
        "ax": ax,
    }
    median_kwargs = {
        "data": summary_df,
        "x": "median_intensity",
        "y": "median_area",
        "hue": "condition_label",
        "hue_order": condition_label_order,
        "alpha": 1.0,
        "s": 165,
        "linewidth": 1.1,
        "edgecolor": "black",
        "ax": ax,
    }
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
    if max_points_per_group < 1:
        raise ValueError("max_points_per_group must be at least 1")
    if not 0 < x_upper_percentile <= 100:
        raise ValueError("x_upper_percentile must be between 0 and 100")
    if not 0 < area_upper_percentile <= 100:
        raise ValueError("area_upper_percentile must be between 0 and 100")

    morphology_df = build_morphology_table(
        analysis,
        antibody=antibody,
        donors=donors,
        conditions=conditions,
    )
    if morphology_df.empty:
        raise ValueError(f"No valid morphology data available for {antibody}")

    rng = np.random.default_rng(random_seed)
    sampled_frames = []
    for _, group in morphology_df.groupby(["condition", "donor"], observed=True):
        if len(group) > max_points_per_group:
            sampled_frames.append(
                group.iloc[rng.choice(len(group), size=max_points_per_group, replace=False)]
            )
        else:
            sampled_frames.append(group)
    plot_df = pd.concat(sampled_frames, ignore_index=True)

    x_max = float(np.percentile(plot_df["intensity"], x_upper_percentile))
    area_max = float(np.percentile(plot_df["area"], area_upper_percentile))
    plot_df = plot_df.loc[
        (plot_df["intensity"] <= x_max)
        & (plot_df["area"] <= area_max)
    ].copy()
    if plot_df.empty:
        raise ValueError("No morphology points remain after display percentile filtering")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    donor_labels = list(plot_df["donor_label"].cat.categories)
    fig, axes = plt.subplots(
        1,
        len(donor_labels),
        figsize=(6.5 * len(donor_labels), 5.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
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
        f"{antibody} intensity vs cell shape\n"
        f"Point size is StarDist cell area; values above p{x_upper_percentile:g} intensity "
        f"or p{area_upper_percentile:g} area are hidden for display.",
        fontsize=14,
    )
    plt.show()

    return morphology_df


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
    "DEFAULT_STARDIST_N_TILES",
    "MeasurementResult",
    "build_morphology_fold_change_table",
    "build_morphology_summary_table",
    "build_morphology_table",
    "build_stat_summary_table",
    "display_stat_summary_tables",
    "extract_condition_comparison",
    "extract_single_cell_fluorescence",
    "extract_single_cell_relative_fluorescence",
    "find_image_path",
    "load_grayscale_tif",
    "load_stardist_model",
    "mann_whitney_u_test",
    "plot_area_intensity_scatter",
    "plot_condition_comparison",
    "plot_condition_histograms",
    "plot_condition_brightness_debug",
    "plot_condition_relative_background_debug",
    "plot_morphology_heatmap",
    "plot_morphology_scatter",
    "plot_summary_heatmap",
    "plot_violins_with_stats",
]
