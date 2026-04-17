from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from ..defaults import (
    DEFAULT_ANTIBODY_ORDER,
    DEFAULT_ANTIBODY_SPECS,
    DEFAULT_BASELINE_CONDITION,
    DEFAULT_COMPARISON_CHANNEL,
    DEFAULT_COMPARISON_MARKER_PREFIX,
    DEFAULT_DATA_ROOT,
    DEFAULT_DONOR_COLORS,
    DEFAULT_STARDIST_N_TILES,
)
from ..models import (
    AntibodySpec,
    BatchAnalysis,
    ComparisonDonorResult,
    ConditionComparison,
    MEASUREMENT_SCALE_BACKGROUND_RATIO,
    MEASUREMENT_SCALE_RAW_INTENSITY,
    MeasurementResult,
    ResultKey,
)
from .extraction import (
    divide_by_background,
    estimate_background_intensity,
    extract_cell_measurements,
    load_grayscale_tif,
    predict_stardist_labels,
    summarize_intensities,
)
from .planning import build_condition_comparison_requests, build_measurement_requests


def _segment_and_measure_image(
    image_path: Path,
    *,
    model_name: str,
    n_tiles: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = load_grayscale_tif(image_path)
    labels, _ = predict_stardist_labels(image, model_name=model_name, n_tiles=n_tiles)
    mean_intensities, areas, eccentricities, solidities = extract_cell_measurements(image, labels)
    return image, labels, mean_intensities, areas, eccentricities, solidities


def _normalize_measurement_values(
    mean_intensities: np.ndarray,
    *,
    image: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    relative_to_background: bool,
    background_percentile: float,
) -> tuple[np.ndarray, float | None]:
    if not relative_to_background:
        return mean_intensities, None
    if image is None or labels is None:
        raise ValueError("image and labels are required for background normalization")

    background_intensity = estimate_background_intensity(
        image,
        labels,
        background_percentile=background_percentile,
    )
    return divide_by_background(mean_intensities, background_intensity), background_intensity


def _build_measurement_result(
    image_path: Path,
    *,
    raw_mean_intensities: np.ndarray,
    measurement_values: np.ndarray,
    areas: np.ndarray,
    eccentricities: np.ndarray,
    solidities: np.ndarray,
    measurement_scale: str,
    background_intensity: float | None,
    background_percentile: float | None,
) -> MeasurementResult:
    cell_count, measurement_mean, measurement_median = summarize_intensities(measurement_values)
    return MeasurementResult(
        path=image_path,
        cell_count=cell_count,
        raw_mean_intensities=raw_mean_intensities,
        measurement_values=measurement_values,
        measurement_mean=measurement_mean,
        measurement_median=measurement_median,
        measurement_scale=measurement_scale,
        areas=areas,
        eccentricities=eccentricities,
        solidities=solidities,
        background_intensity=background_intensity,
        background_percentile=background_percentile,
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
    requests = build_condition_comparison_requests(
        donor_list,
        condition,
        marker_prefix=marker_prefix,
        channel=channel,
        data_root=data_root,
    )

    for request in requests:
        image, labels, mean_intensities, _, _, _ = _segment_and_measure_image(
            request.path,
            model_name=model_name,
            n_tiles=n_tiles,
        )
        cell_count, overall_mean, overall_median = summarize_intensities(mean_intensities)
        donor_results[request.donor] = ComparisonDonorResult(
            path=request.path,
            image=image,
            labels=labels,
            mean_intensities=mean_intensities,
            cell_count=cell_count,
            overall_mean=overall_mean,
            overall_median=overall_median,
        )

    return ConditionComparison(
        condition=condition,
        donors=donor_list,
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
    measurement_scale = (
        MEASUREMENT_SCALE_BACKGROUND_RATIO
        if relative_to_background
        else MEASUREMENT_SCALE_RAW_INTENSITY
    )
    requests = build_measurement_requests(
        donor_list,
        condition_list,
        antibody_specs=antibody_specs,
        antibody_order=order_list,
        data_root=data_root,
    )

    results: dict[ResultKey, MeasurementResult] = {}
    current_antibody = None
    for request in requests:
        if request.antibody != current_antibody:
            current_antibody = request.antibody
            message = f"Analyzing {current_antibody}"
            if relative_to_background:
                message += " relative to image background"
            print(f"{message}...")

        image, labels, mean_intensities, areas, eccentricities, solidities = _segment_and_measure_image(
            request.path,
            model_name=model_name,
            n_tiles=n_tiles,
        )
        result_intensities, background_intensity = _normalize_measurement_values(
            mean_intensities,
            image=image,
            labels=labels,
            relative_to_background=relative_to_background,
            background_percentile=background_percentile,
        )
        results[ResultKey(
            antibody=request.antibody,
            condition=request.condition,
            donor=request.donor,
        )] = _build_measurement_result(
            request.path,
            raw_mean_intensities=mean_intensities,
            measurement_values=result_intensities,
            areas=areas,
            eccentricities=eccentricities,
            solidities=solidities,
            measurement_scale=measurement_scale,
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
        measurement_scale=measurement_scale,
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
