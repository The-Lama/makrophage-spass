from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .config import (
    AntibodySpec,
    BatchAnalysis,
    ComparisonDonorResult,
    ConditionComparison,
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_ANTIBODY_SPECS,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_COMPARISON_CHANNEL,
    DEFAULT_COMPARISON_MARKER_PREFIX,
    DEFAULT_DATA_ROOT,
    DEFAULT_DONOR_COLORS,
    DEFAULT_STARDIST_N_TILES,
    MeasurementResult,
)
from .core import (
    divide_by_background,
    estimate_background_intensity,
    extract_cell_measurements,
    find_image_path,
    load_grayscale_tif,
    predict_stardist_labels,
    summarize_intensities,
)


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
    donor_results: dict[str, ComparisonDonorResult] = {}

    for donor in donor_list:
        image_path = find_image_path(donor, condition, marker_prefix, channel, data_root=data_root)
        image = load_grayscale_tif(image_path)
        labels, details = predict_stardist_labels(image, model_name=model_name, n_tiles=n_tiles)
        mean_intensities, areas, eccentricities = extract_cell_measurements(image, labels)
        cell_count, overall_mean, overall_median = summarize_intensities(mean_intensities)
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


def _run_batch_analysis(
    donors: Iterable[str],
    conditions: Iterable[str],
    *,
    antibody_specs: Mapping[str, AntibodySpec],
    antibody_order: Iterable[str],
    baseline_condition: str,
    donor_colors: Mapping[str, str],
    data_root: str | Path,
    model_name: str,
    n_tiles: tuple[int, int] | None,
    relative_to_background: bool = False,
    background_percentile: float = 50.0,
) -> BatchAnalysis:
    donor_list = list(donors)
    condition_list = list(conditions)
    order_list = list(antibody_order)
    if baseline_condition not in condition_list:
        raise ValueError(f"baseline_condition must be one of {condition_list}")

    results: dict[tuple[str, str, str], MeasurementResult] = {}
    for antibody in order_list:
        spec = antibody_specs[antibody]
        message = f"Analyzing {antibody}"
        if relative_to_background:
            message += " relative to image background"
        print(f"{message}...")
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
                labels, _ = predict_stardist_labels(image, model_name=model_name, n_tiles=n_tiles)
                mean_intensities, areas, eccentricities = extract_cell_measurements(image, labels)
                background_intensity = None
                result_intensities = mean_intensities
                if relative_to_background:
                    background_intensity = estimate_background_intensity(
                        image,
                        labels,
                        background_percentile=background_percentile,
                    )
                    result_intensities = divide_by_background(mean_intensities, background_intensity)
                cell_count, overall_mean, overall_median = summarize_intensities(result_intensities)
                results[(antibody, condition, donor)] = MeasurementResult(
                    path=image_path,
                    cell_count=cell_count,
                    mean_intensities=result_intensities,
                    overall_mean=overall_mean,
                    overall_median=overall_median,
                    areas=areas,
                    eccentricities=eccentricities,
                    background_intensity=background_intensity,
                    background_percentile=background_percentile if relative_to_background else None,
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
    return _run_batch_analysis(
        donors,
        conditions,
        antibody_specs=antibody_specs,
        antibody_order=antibody_order,
        baseline_condition=baseline_condition,
        donor_colors=donor_colors,
        data_root=data_root,
        model_name=model_name,
        n_tiles=n_tiles,
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
    return _run_batch_analysis(
        donors,
        conditions,
        antibody_specs=antibody_specs,
        antibody_order=antibody_order,
        baseline_condition=baseline_condition,
        donor_colors=donor_colors,
        data_root=data_root,
        model_name=model_name,
        n_tiles=n_tiles,
        relative_to_background=True,
        background_percentile=background_percentile,
    )
