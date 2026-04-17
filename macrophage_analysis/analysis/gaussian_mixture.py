from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .clustering import (
    build_donor_feature_data,
    invert_centroids,
    ordered_present_values,
    run_kmeans_with_history,
)

__all__ = [
    "run_split_gaussian_mixture",
]

_DEFAULT_FEATURE_COLUMNS = ("intensity", "area", "eccentricity")
_DEFAULT_CENTROID_COLUMN_NAMES = (
    "cd206_intensity_centroid",
    "area_centroid",
    "eccentricity_centroid",
)
_METADATA_COLUMNS = ("condition", "condition_label", "donor", "donor_label")
_EPSILON = 1e-12


def run_split_gaussian_mixture(
    cells: pd.DataFrame,
    *,
    n_components: int,
    random_seed: int,
    reference_donor_label: str,
    feature_columns: Sequence[str] = _DEFAULT_FEATURE_COLUMNS,
    centroid_column_names: Sequence[str] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    reg_covar: float = 1e-6,
    adapt_non_reference_weights: bool = True,
    non_reference_mode: str = "reference_transfer",
    max_alignment_distance: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    resolved_feature_columns = _resolve_feature_columns(feature_columns)
    resolved_centroid_column_names = _resolve_centroid_column_names(
        resolved_feature_columns,
        centroid_column_names,
    )
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than 0")
    if reg_covar <= 0:
        raise ValueError("reg_covar must be greater than 0")
    resolved_non_reference_mode = _resolve_non_reference_mode(non_reference_mode)
    resolved_max_alignment_distance = _resolve_max_alignment_distance(max_alignment_distance)

    donor_feature_data = build_donor_feature_data(
        cells,
        feature_columns=resolved_feature_columns,
    )
    donor_label_order = list(donor_feature_data)
    if reference_donor_label not in set(donor_label_order):
        raise ValueError(
            f"reference donor {reference_donor_label!r} is not present in the donor clustering data"
        )

    reference_payload = donor_feature_data[reference_donor_label]
    reference_parameters, reference_history = _fit_gaussian_mixture(
        reference_payload["scaled_features"],
        n_components=n_components,
        random_seed=random_seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
        reg_covar=reg_covar,
    )

    ordered_donor_labels = [reference_donor_label] + [
        donor_label for donor_label in donor_label_order if donor_label != reference_donor_label
    ]
    donor_runs: dict[str, dict[str, Any]] = {}
    clustered_frames: list[pd.DataFrame] = []
    component_tables: list[pd.DataFrame] = []
    cluster_names = [f"Cluster {index + 1}" for index in range(n_components)]
    comparison_mean = np.asarray(reference_payload["feature_mean"], dtype=float)
    comparison_std = np.asarray(reference_payload["feature_std"], dtype=float)

    for donor_label in ordered_donor_labels:
        payload = donor_feature_data[donor_label]
        if donor_label == reference_donor_label:
            donor_parameters = {
                "means": reference_parameters["means"].copy(),
                "covariances": reference_parameters["covariances"].copy(),
                "weights": reference_parameters["weights"].copy(),
            }
            donor_result = _score_gaussian_mixture(
                payload["scaled_features"],
                means=donor_parameters["means"],
                covariances=donor_parameters["covariances"],
                weights=donor_parameters["weights"],
            )
            donor_weights = donor_parameters["weights"].copy()
            donor_history = reference_history
        elif resolved_non_reference_mode == "reference_transfer":
            donor_parameters = {
                "means": reference_parameters["means"].copy(),
                "covariances": reference_parameters["covariances"].copy(),
                "weights": reference_parameters["weights"].copy(),
            }
            donor_result = _score_non_reference_donor(
                payload["scaled_features"],
                means=donor_parameters["means"],
                covariances=donor_parameters["covariances"],
                initial_weights=donor_parameters["weights"],
                max_iterations=max_iterations,
                tolerance=tolerance,
                adapt_weights=adapt_non_reference_weights,
            )
            donor_weights = donor_result["weights"].copy()
            donor_parameters["weights"] = donor_weights.copy()
            donor_history = donor_result["history"]
        else:
            donor_parameters, donor_history = _fit_gaussian_mixture(
                payload["scaled_features"],
                n_components=n_components,
                random_seed=random_seed,
                max_iterations=max_iterations,
                tolerance=tolerance,
                reg_covar=reg_covar,
            )
            donor_result = _score_gaussian_mixture(
                payload["scaled_features"],
                means=donor_parameters["means"],
                covariances=donor_parameters["covariances"],
                weights=donor_parameters["weights"],
            )
            donor_weights = donor_parameters["weights"].copy()

        assignments = donor_result["assignments"]
        responsibilities = donor_result["responsibilities"]
        assignment_probability = responsibilities[np.arange(len(assignments)), assignments]
        clustered_cells = payload["feature_frame"].assign(
            cluster=pd.Categorical(
                [cluster_names[index] for index in assignments],
                categories=cluster_names,
                ordered=True,
            ),
            assignment_probability=assignment_probability.astype(float),
        )
        component_table = _build_component_summary_table(
            clustered_cells,
            donor_label=donor_label,
            reference_donor_label=reference_donor_label,
            cluster_names=cluster_names,
            donor_weights=donor_weights,
            responsibilities=responsibilities,
            donor_parameters=donor_parameters,
            reference_parameters=reference_parameters,
            feature_columns=resolved_feature_columns,
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_std=np.asarray(payload["feature_std"], dtype=float),
            centroid_column_names=resolved_centroid_column_names,
        )

        donor_runs[donor_label] = {
            "feature_columns": resolved_feature_columns,
            "centroid_column_names": resolved_centroid_column_names,
            "history": donor_history,
            "clustered_cells": clustered_cells,
            "component_table": component_table,
            "responsibilities": responsibilities,
            "weights": donor_weights,
            "component_means": donor_parameters["means"].copy(),
            "component_covariances": donor_parameters["covariances"].copy(),
            "reference_means": reference_parameters["means"].copy(),
            "reference_covariances": reference_parameters["covariances"].copy(),
            "non_reference_mode": resolved_non_reference_mode,
        }
        clustered_frames.append(clustered_cells)
        component_tables.append(component_table)

    clustered_cells = pd.concat(clustered_frames, ignore_index=True)
    component_table = pd.concat(component_tables, ignore_index=True)
    if resolved_non_reference_mode == "independent":
        aligned_cells, aligned_component_table = _align_components_to_reference(
            clustered_cells,
            component_table,
            reference_donor_label=reference_donor_label,
            centroid_column_names=resolved_centroid_column_names,
            comparison_mean=comparison_mean,
            comparison_std=comparison_std,
            max_alignment_distance=resolved_max_alignment_distance,
        )
    else:
        aligned_cells, aligned_component_table = _apply_reference_component_labels(
            clustered_cells,
            component_table,
            reference_donor_label=reference_donor_label,
            centroid_column_names=resolved_centroid_column_names,
            comparison_mean=comparison_mean,
            comparison_std=comparison_std,
            max_alignment_distance=resolved_max_alignment_distance,
        )

    return (
        donor_feature_data,
        donor_runs,
        aligned_cells,
        aligned_component_table,
    )


def _fit_gaussian_mixture(
    features: np.ndarray,
    *,
    n_components: int,
    random_seed: int,
    max_iterations: int,
    tolerance: float,
    reg_covar: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    if n_components > len(features):
        raise ValueError("n_components must be less than or equal to the number of cells")

    initial_means, _, _ = run_kmeans_with_history(
        features,
        k=n_components,
        random_seed=random_seed,
        max_iterations=min(max_iterations, 20),
    )
    n_features = features.shape[1]
    global_covariance = np.cov(features, rowvar=False)
    if np.ndim(global_covariance) == 0:
        global_covariance = np.asarray([[float(global_covariance)]], dtype=float)
    global_covariance = np.asarray(global_covariance, dtype=float)
    global_covariance += np.eye(n_features, dtype=float) * reg_covar

    means = initial_means.copy()
    covariances = np.repeat(global_covariance[None, :, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components, dtype=float)
    history: list[dict[str, float]] = []
    previous_log_likelihood: float | None = None

    for iteration in range(1, max_iterations + 1):
        score = _score_gaussian_mixture(
            features,
            means=means,
            covariances=covariances,
            weights=weights,
        )
        responsibilities = score["responsibilities"]
        log_likelihood = float(score["log_likelihood"])
        history.append({"iteration": float(iteration), "log_likelihood": log_likelihood})

        nk = responsibilities.sum(axis=0)
        weights = nk / len(features)
        means = (responsibilities.T @ features) / np.maximum(nk[:, None], _EPSILON)
        covariances = _estimate_covariances(
            features,
            responsibilities,
            means,
            reg_covar=reg_covar,
        )

        if previous_log_likelihood is not None and abs(log_likelihood - previous_log_likelihood) <= tolerance:
            break
        previous_log_likelihood = log_likelihood

    return {
        "means": means,
        "covariances": covariances,
        "weights": weights,
    }, history


def _score_non_reference_donor(
    features: np.ndarray,
    *,
    means: np.ndarray,
    covariances: np.ndarray,
    initial_weights: np.ndarray,
    max_iterations: int,
    tolerance: float,
    adapt_weights: bool,
) -> dict[str, Any]:
    weights = np.asarray(initial_weights, dtype=float).copy()
    history: list[dict[str, float]] = []
    previous_log_likelihood: float | None = None
    if not adapt_weights:
        score = _score_gaussian_mixture(
            features,
            means=means,
            covariances=covariances,
            weights=weights,
        )
        history.append({"iteration": 1.0, "log_likelihood": float(score["log_likelihood"])})
        score["weights"] = weights
        score["history"] = history
        return score

    for iteration in range(1, max_iterations + 1):
        score = _score_gaussian_mixture(
            features,
            means=means,
            covariances=covariances,
            weights=weights,
        )
        responsibilities = score["responsibilities"]
        log_likelihood = float(score["log_likelihood"])
        history.append({"iteration": float(iteration), "log_likelihood": log_likelihood})
        weights = responsibilities.mean(axis=0)

        if previous_log_likelihood is not None and abs(log_likelihood - previous_log_likelihood) <= tolerance:
            break
        previous_log_likelihood = log_likelihood

    final_score = _score_gaussian_mixture(
        features,
        means=means,
        covariances=covariances,
        weights=weights,
    )
    final_score["weights"] = weights
    final_score["history"] = history
    return final_score


def _score_gaussian_mixture(
    features: np.ndarray,
    *,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    log_component_density = _compute_log_component_density(
        features,
        means=means,
        covariances=covariances,
    )
    log_weights = np.log(np.maximum(np.asarray(weights, dtype=float), _EPSILON))
    weighted_log_density = log_component_density + log_weights[None, :]
    log_normalizer = _logsumexp(weighted_log_density, axis=1)
    responsibilities = np.exp(weighted_log_density - log_normalizer[:, None])
    assignments = responsibilities.argmax(axis=1)
    return {
        "responsibilities": responsibilities,
        "assignments": assignments,
        "log_likelihood": float(log_normalizer.sum()),
    }


def _compute_log_component_density(
    features: np.ndarray,
    *,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    n_features = features.shape[1]
    log_density_columns: list[np.ndarray] = []
    for mean, covariance in zip(means, covariances):
        regularized_covariance = np.asarray(covariance, dtype=float)
        sign, logdet = np.linalg.slogdet(regularized_covariance)
        if sign <= 0:
            raise ValueError("Gaussian mixture covariance must be positive definite")
        centered = features - mean
        solved = np.linalg.solve(regularized_covariance, centered.T).T
        mahalanobis = np.sum(centered * solved, axis=1)
        log_density_columns.append(
            -0.5 * (n_features * np.log(2.0 * np.pi) + logdet + mahalanobis)
        )
    return np.column_stack(log_density_columns)


def _estimate_covariances(
    features: np.ndarray,
    responsibilities: np.ndarray,
    means: np.ndarray,
    *,
    reg_covar: float,
) -> np.ndarray:
    n_components = responsibilities.shape[1]
    n_features = features.shape[1]
    covariances = np.zeros((n_components, n_features, n_features), dtype=float)
    identity = np.eye(n_features, dtype=float)
    for component_index in range(n_components):
        diff = features - means[component_index]
        weight = responsibilities[:, component_index][:, None]
        effective_n = max(float(responsibilities[:, component_index].sum()), _EPSILON)
        covariance = (weight * diff).T @ diff / effective_n
        covariances[component_index] = covariance + identity * reg_covar
    return covariances


def _build_component_summary_table(
    clustered_cells: pd.DataFrame,
    *,
    donor_label: str,
    reference_donor_label: str,
    cluster_names: list[str],
    donor_weights: np.ndarray,
    responsibilities: np.ndarray,
    donor_parameters: dict[str, np.ndarray],
    reference_parameters: dict[str, np.ndarray],
    feature_columns: tuple[str, ...],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    centroid_column_names: tuple[str, ...],
) -> pd.DataFrame:
    default_centroids = invert_centroids(
        donor_parameters["means"],
        feature_mean,
        feature_std,
    )
    assignment_counts = (
        clustered_cells.groupby("cluster", observed=True).size().reindex(cluster_names, fill_value=0)
    )
    assignment_fractions = assignment_counts / max(len(clustered_cells), 1)
    summary_rows: list[dict[str, object]] = []
    for component_index, cluster_name in enumerate(cluster_names):
        assigned_mask = clustered_cells["cluster"].astype(str) == cluster_name
        assigned_probabilities = responsibilities[assigned_mask.to_numpy(dtype=bool), component_index]
        assigned_cells = clustered_cells.loc[assigned_mask, list(feature_columns)]
        if len(assigned_cells) > 0:
            display_centroid = assigned_cells.mean(axis=0).to_numpy(dtype=float)
        else:
            display_centroid = default_centroids[component_index]
        summary_row: dict[str, object] = {
            "reference_donor_label": str(reference_donor_label),
            "donor_label": str(donor_label),
            "cluster": cluster_name,
            "reference_weight": float(reference_parameters["weights"][component_index]),
            "donor_weight": float(donor_weights[component_index]),
            "assignment_cell_count": int(assignment_counts.loc[cluster_name]),
            "assignment_fraction": float(assignment_fractions.loc[cluster_name]),
            "mean_assignment_probability": (
                float(assigned_probabilities.mean()) if assigned_probabilities.size else np.nan
            ),
            "alignment_distance": np.nan,
        }
        for column_name, value in zip(centroid_column_names, display_centroid):
            summary_row[column_name] = float(value)
        summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)


def _find_best_component_match(
    reference_centroids: np.ndarray,
    donor_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_centroids) != len(donor_centroids):
        raise ValueError("reference and donor component tables must have the same number of clusters")

    distance_matrix = np.linalg.norm(
        donor_centroids[:, None, :] - reference_centroids[None, :, :],
        axis=2,
    )

    best_permutation: tuple[int, ...] | None = None
    best_score: float | None = None
    for permutation in itertools.permutations(range(len(reference_centroids))):
        score = float(
            sum(
                distance_matrix[donor_index, reference_index]
                for donor_index, reference_index in enumerate(permutation)
            )
        )
        if best_score is None or score < best_score:
            best_permutation = permutation
            best_score = score

    if best_permutation is None:
        raise ValueError("Unable to match Gaussian-mixture components")

    permutation_array = np.asarray(best_permutation, dtype=int)
    matched_distances = distance_matrix[np.arange(len(permutation_array)), permutation_array]
    return permutation_array, matched_distances


def _align_components_to_reference(
    clustered_cells: pd.DataFrame,
    component_table: pd.DataFrame,
    *,
    reference_donor_label: str,
    centroid_column_names: tuple[str, ...],
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    max_alignment_distance: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_label_order = ordered_present_values(component_table["donor_label"])
    if reference_donor_label not in set(donor_label_order):
        raise ValueError(
            f"reference donor {reference_donor_label!r} is not present in the Gaussian-mixture component table"
        )

    reference_rows = component_table.loc[
        component_table["donor_label"] == reference_donor_label
    ].copy()
    reference_clusters = ordered_present_values(reference_rows["cluster"])
    reference_centroids = (
        reference_rows.set_index("cluster")
        .loc[reference_clusters, list(centroid_column_names)]
        .to_numpy(dtype=float)
    )
    reference_centroids = (reference_centroids - comparison_mean) / comparison_std
    aligned_categories = _build_aligned_cluster_categories(
        reference_clusters,
        include_unmatched=max_alignment_distance is not None,
    )

    aligned_cluster_frames: list[pd.DataFrame] = []
    aligned_component_frames: list[pd.DataFrame] = []
    for donor_label in donor_label_order:
        donor_rows = component_table.loc[component_table["donor_label"] == donor_label].copy()
        donor_clusters = ordered_present_values(donor_rows["cluster"])
        donor_centroids = (
            donor_rows.set_index("cluster")
            .loc[donor_clusters, list(centroid_column_names)]
            .to_numpy(dtype=float)
        )
        donor_centroids = (donor_centroids - comparison_mean) / comparison_std

        if donor_label == reference_donor_label:
            matched_reference_indices = np.arange(len(reference_clusters), dtype=int)
            matched_distances = np.zeros(len(reference_clusters), dtype=float)
        else:
            matched_reference_indices, matched_distances = _find_best_component_match(
                reference_centroids,
                donor_centroids,
            )

        aligned_cluster_map = {
            donor_clusters[donor_index]: (
                reference_clusters[reference_index]
                if max_alignment_distance is None
                or matched_distances[donor_index] <= max_alignment_distance
                else "Unmatched"
            )
            for donor_index, reference_index in enumerate(matched_reference_indices)
        }
        distance_map = {
            donor_clusters[donor_index]: float(matched_distances[donor_index])
            for donor_index in range(len(donor_clusters))
        }

        donor_clustered_cells = clustered_cells.loc[
            clustered_cells["donor_label"] == donor_label
        ].copy()
        donor_clustered_cells["aligned_cluster"] = pd.Categorical(
            donor_clustered_cells["cluster"].astype(str).map(aligned_cluster_map),
            categories=aligned_categories,
            ordered=True,
        )
        aligned_cluster_frames.append(donor_clustered_cells)

        donor_rows["aligned_cluster"] = pd.Categorical(
            donor_rows["cluster"].astype(str).map(aligned_cluster_map),
            categories=aligned_categories,
            ordered=True,
        )
        donor_rows["alignment_distance"] = donor_rows["cluster"].astype(str).map(distance_map).astype(float)
        aligned_component_frames.append(donor_rows)

    return (
        pd.concat(aligned_cluster_frames, ignore_index=True),
        pd.concat(aligned_component_frames, ignore_index=True),
    )


def _apply_reference_component_labels(
    clustered_cells: pd.DataFrame,
    component_table: pd.DataFrame,
    *,
    reference_donor_label: str,
    centroid_column_names: tuple[str, ...],
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    max_alignment_distance: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_label_order = ordered_present_values(component_table["donor_label"])
    reference_rows = component_table.loc[
        component_table["donor_label"] == reference_donor_label
    ].copy()
    reference_clusters = ordered_present_values(reference_rows["cluster"])
    reference_centroids = (
        reference_rows.set_index("cluster")
        .loc[reference_clusters, list(centroid_column_names)]
        .to_numpy(dtype=float)
    )
    reference_centroids = (reference_centroids - comparison_mean) / comparison_std
    aligned_categories = _build_aligned_cluster_categories(
        reference_clusters,
        include_unmatched=max_alignment_distance is not None,
    )

    aligned_cluster_frames: list[pd.DataFrame] = []
    aligned_component_frames: list[pd.DataFrame] = []
    for donor_label in donor_label_order:
        donor_rows = component_table.loc[component_table["donor_label"] == donor_label].copy()
        donor_centroids = (
            donor_rows.set_index("cluster")
            .loc[reference_clusters, list(centroid_column_names)]
            .to_numpy(dtype=float)
        )
        donor_centroids = (donor_centroids - comparison_mean) / comparison_std
        alignment_distances = np.linalg.norm(donor_centroids - reference_centroids, axis=1)
        distance_map = {
            cluster_name: float(distance)
            for cluster_name, distance in zip(reference_clusters, alignment_distances)
        }
        aligned_cluster_map = {
            cluster_name: (
                cluster_name
                if max_alignment_distance is None
                or distance_map[cluster_name] <= max_alignment_distance
                else "Unmatched"
            )
            for cluster_name in reference_clusters
        }

        donor_clustered_cells = clustered_cells.loc[
            clustered_cells["donor_label"] == donor_label
        ].copy()
        donor_clustered_cells["aligned_cluster"] = pd.Categorical(
            donor_clustered_cells["cluster"].astype(str).map(aligned_cluster_map),
            categories=aligned_categories,
            ordered=True,
        )
        aligned_cluster_frames.append(donor_clustered_cells)

        donor_rows["aligned_cluster"] = pd.Categorical(
            donor_rows["cluster"].astype(str).map(aligned_cluster_map),
            categories=aligned_categories,
            ordered=True,
        )
        donor_rows["alignment_distance"] = donor_rows["cluster"].astype(str).map(distance_map).astype(float)
        aligned_component_frames.append(donor_rows)

    return (
        pd.concat(aligned_cluster_frames, ignore_index=True),
        pd.concat(aligned_component_frames, ignore_index=True),
    )


def _resolve_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    resolved_feature_columns = tuple(str(column) for column in feature_columns)
    if not resolved_feature_columns:
        raise ValueError("feature_columns must contain at least one feature")
    return resolved_feature_columns


def _resolve_centroid_column_names(
    feature_columns: tuple[str, ...],
    centroid_column_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if centroid_column_names is None:
        if feature_columns == _DEFAULT_FEATURE_COLUMNS:
            return _DEFAULT_CENTROID_COLUMN_NAMES
        return tuple(f"{column}_centroid" for column in feature_columns)

    resolved_centroid_column_names = tuple(str(column) for column in centroid_column_names)
    if len(resolved_centroid_column_names) != len(feature_columns):
        raise ValueError(
            "centroid_column_names must have the same length as feature_columns"
        )
    return resolved_centroid_column_names


def _resolve_non_reference_mode(non_reference_mode: str) -> str:
    resolved_mode = str(non_reference_mode)
    if resolved_mode not in {"reference_transfer", "independent"}:
        raise ValueError("non_reference_mode must be 'reference_transfer' or 'independent'")
    return resolved_mode


def _resolve_max_alignment_distance(max_alignment_distance: float | None) -> float | None:
    if max_alignment_distance is None:
        return None

    resolved_distance = float(max_alignment_distance)
    if not np.isfinite(resolved_distance) or resolved_distance <= 0.0:
        raise ValueError("max_alignment_distance must be greater than 0 when provided")
    return resolved_distance


def _build_aligned_cluster_categories(
    reference_clusters: list[Any],
    *,
    include_unmatched: bool,
) -> list[Any]:
    if not include_unmatched:
        return list(reference_clusters)
    return [*reference_clusters, "Unmatched"]


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    return np.squeeze(max_values, axis=axis) + np.log(np.sum(np.exp(shifted), axis=axis))
