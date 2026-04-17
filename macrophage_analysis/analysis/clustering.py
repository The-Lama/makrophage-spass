from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "build_aligned_cluster_composition_table",
    "build_donor_feature_data",
    "build_split_kmeans_state_sequence",
    "invert_centroids",
    "ordered_present_values",
    "run_kmeans_with_history",
    "run_split_kmeans",
]

_DEFAULT_CLUSTER_FEATURE_COLUMNS = ("intensity", "area")
_DEFAULT_CENTROID_COLUMN_NAMES = ("cd206_intensity_centroid", "area_centroid")
_METADATA_COLUMNS = ("condition", "condition_label", "donor", "donor_label")


def ordered_present_values(series: pd.Series) -> list[Any]:
    non_null = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        present_values = set(non_null.tolist())
        return [value for value in series.cat.categories if value in present_values]
    return list(dict.fromkeys(non_null.tolist()))


def build_donor_feature_data(
    cells: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = _DEFAULT_CLUSTER_FEATURE_COLUMNS,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    resolved_feature_columns = _resolve_feature_columns(feature_columns)
    resolved_feature_mean, resolved_feature_std = _resolve_feature_scaler(
        cells,
        feature_columns=resolved_feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    donor_feature_data: dict[str, dict[str, Any]] = {}
    donor_label_order = ordered_present_values(cells["donor_label"])
    for donor_label in donor_label_order:
        donor_cells = cells.loc[cells["donor_label"] == donor_label].copy()
        (
            feature_frame,
            display_matrix,
            scaled_matrix,
            donor_feature_mean,
            donor_feature_std,
        ) = _build_feature_matrices(
            donor_cells,
            feature_columns=resolved_feature_columns,
            feature_mean=resolved_feature_mean,
            feature_std=resolved_feature_std,
        )
        donor_feature_data[str(donor_label)] = {
            "feature_columns": resolved_feature_columns,
            "feature_frame": feature_frame,
            "display_features": display_matrix,
            "scaled_features": scaled_matrix,
            "feature_mean": donor_feature_mean,
            "feature_std": donor_feature_std,
        }
    return donor_feature_data


def build_split_kmeans_state_sequence(
    donor_runs: dict[str, dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    if not donor_runs:
        return []

    max_history_length = max(len(result["history"]) for result in donor_runs.values())
    return [
        {
            donor_label: result["history"][min(state_index, len(result["history"]) - 1)]
            for donor_label, result in donor_runs.items()
        }
        for state_index in range(max_history_length)
    ]


def build_aligned_cluster_composition_table(clustered_cells: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"donor_label", "condition_label", "aligned_cluster"}
    missing_columns = sorted(required_columns - set(clustered_cells.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"clustered_cells must include aligned donor clustering columns; missing: {missing_text}"
        )

    composition = (
        clustered_cells.groupby(
            ["donor_label", "condition_label", "aligned_cluster"],
            observed=True,
        )
        .size()
        .rename("cell_count")
        .reset_index()
    )
    composition["fraction"] = composition.groupby(
        ["donor_label", "condition_label"],
        observed=True,
    )["cell_count"].transform(lambda values: values / values.sum())
    return composition


def invert_centroids(
    centroids: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    return centroids * feature_std + feature_mean


def run_kmeans_with_history(
    features: np.ndarray,
    *,
    k: int,
    random_seed: int,
    max_iterations: int = 12,
    tolerance: float = 1e-6,
    initial_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    centroids, rng = _initialize_centroids(
        features,
        k=k,
        random_seed=random_seed,
        initial_centroids=initial_centroids,
    )
    history: list[dict[str, Any]] = []
    assignments = np.zeros(len(features), dtype=int)
    for iteration in range(1, max_iterations + 1):
        assignments = _assign_clusters(features, centroids)
        history.append(
            {
                "iteration": iteration,
                "phase": "assignment",
                "centroids": centroids.copy(),
                "assignments": assignments.copy(),
                "shift": np.nan,
            }
        )
        updated_centroids = _recompute_centroids(features, assignments, centroids, rng)
        total_shift = float(np.linalg.norm(updated_centroids - centroids, axis=1).sum())
        history.append(
            {
                "iteration": iteration,
                "phase": "update",
                "centroids": updated_centroids.copy(),
                "assignments": assignments.copy(),
                "shift": total_shift,
            }
        )
        if np.allclose(updated_centroids, centroids, atol=tolerance, rtol=0.0):
            centroids = updated_centroids
            break
        centroids = updated_centroids
    return centroids, assignments, history


def run_split_kmeans(
    cells: pd.DataFrame,
    *,
    k: int,
    random_seed: int,
    max_iterations: int = 25,
    reference_donor_label: str,
    feature_columns: Sequence[str] = _DEFAULT_CLUSTER_FEATURE_COLUMNS,
    centroid_column_names: Sequence[str] | None = None,
    non_reference_mode: str = "warm_start",
    max_alignment_distance: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    resolved_feature_columns = _resolve_feature_columns(feature_columns)
    resolved_centroid_column_names = _resolve_centroid_column_names(
        resolved_feature_columns,
        centroid_column_names,
    )
    resolved_non_reference_mode = _resolve_non_reference_mode(non_reference_mode)
    resolved_max_alignment_distance = _resolve_max_alignment_distance(max_alignment_distance)
    comparison_mean, comparison_std = _resolve_feature_scaler(
        cells,
        feature_columns=resolved_feature_columns,
    )
    donor_feature_data = build_donor_feature_data(
        cells,
        feature_columns=resolved_feature_columns,
        feature_mean=comparison_mean,
        feature_std=comparison_std,
    )
    donor_runs: dict[str, dict[str, Any]] = {}
    clustered_frames: list[pd.DataFrame] = []
    centroid_tables: list[pd.DataFrame] = []
    cluster_names = [f"Cluster {cluster_index + 1}" for cluster_index in range(k)]
    donor_label_order = list(donor_feature_data)
    if reference_donor_label not in set(donor_label_order):
        raise ValueError(
            f"reference donor {reference_donor_label!r} is not present in the donor clustering data"
        )
    ordered_donor_labels = [reference_donor_label] + [
        donor_label
        for donor_label in donor_label_order
        if donor_label != reference_donor_label
    ]
    reference_centroids: np.ndarray | None = None

    for donor_label in ordered_donor_labels:
        payload = donor_feature_data[donor_label]
        initial_centroids = None
        if (
            donor_label != reference_donor_label
            and reference_centroids is not None
            and resolved_non_reference_mode in {"warm_start", "reference_fixed"}
        ):
            initial_centroids = reference_centroids.copy()
        if donor_label == reference_donor_label or resolved_non_reference_mode in {
            "warm_start",
            "independent",
        }:
            final_centroids, final_assignments, history = run_kmeans_with_history(
                payload["scaled_features"],
                k=k,
                random_seed=random_seed,
                max_iterations=max_iterations,
                initial_centroids=initial_centroids,
            )
        else:
            if initial_centroids is None:
                raise ValueError("reference centroids must be available before fixed-reference assignment")
            final_assignments = _assign_clusters(payload["scaled_features"], initial_centroids)
            final_centroids = _summarize_assigned_centroids(
                payload["scaled_features"],
                final_assignments,
                reference_centroids=initial_centroids,
            )
            history = [
                {
                    "iteration": 1,
                    "phase": "assignment",
                    "centroids": initial_centroids.copy(),
                    "assignments": final_assignments.copy(),
                    "shift": np.nan,
                }
            ]
        if donor_label == reference_donor_label:
            reference_centroids = final_centroids.copy()
        clustered_cells = payload["feature_frame"].assign(
            cluster=pd.Categorical(
                [cluster_names[index] for index in final_assignments],
                categories=cluster_names,
                ordered=True,
            )
        )
        centroid_table = (
            pd.DataFrame(
                invert_centroids(
                    final_centroids,
                    payload["feature_mean"],
                    payload["feature_std"],
                ),
                columns=list(resolved_centroid_column_names),
            )
            .assign(donor_label=donor_label, cluster=cluster_names)
            [["donor_label", "cluster", *resolved_centroid_column_names]]
        )
        donor_runs[donor_label] = {
            "history": history,
            "initial_centroids": None if initial_centroids is None else initial_centroids.copy(),
            "clustered_cells": clustered_cells,
            "centroid_table": centroid_table,
            "feature_columns": resolved_feature_columns,
            "centroid_column_names": resolved_centroid_column_names,
            "non_reference_mode": resolved_non_reference_mode,
        }
        clustered_frames.append(clustered_cells)
        centroid_tables.append(centroid_table)

    clustered_cells = pd.concat(clustered_frames, ignore_index=True)
    centroid_table = pd.concat(centroid_tables, ignore_index=True)
    if resolved_non_reference_mode == "reference_fixed":
        aligned_cells, aligned_centroids = _apply_reference_cluster_labels(
            clustered_cells,
            centroid_table,
            reference_donor_label=reference_donor_label,
            centroid_column_names=resolved_centroid_column_names,
            comparison_mean=comparison_mean,
            comparison_std=comparison_std,
            max_alignment_distance=resolved_max_alignment_distance,
        )
    else:
        aligned_cells, aligned_centroids = _align_clusters_to_reference(
            clustered_cells,
            centroid_table,
            reference_donor_label=reference_donor_label,
            centroid_column_names=resolved_centroid_column_names,
            comparison_mean=comparison_mean,
            comparison_std=comparison_std,
            max_alignment_distance=resolved_max_alignment_distance,
        )
    return donor_feature_data, donor_runs, aligned_cells, aligned_centroids


def _build_feature_matrices(
    cells: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_required_columns(
        cells,
        [*_METADATA_COLUMNS, *feature_columns],
        context="clustering feature table",
    )
    feature_frame = cells[[*_METADATA_COLUMNS, *feature_columns]].copy()
    display_matrix = feature_frame[list(feature_columns)].to_numpy(dtype=float)
    resolved_feature_mean = np.asarray(feature_mean, dtype=float).copy()
    resolved_feature_std = np.asarray(feature_std, dtype=float).copy()
    resolved_feature_std[~np.isfinite(resolved_feature_std) | (resolved_feature_std == 0)] = 1.0
    scaled_matrix = (display_matrix - resolved_feature_mean) / resolved_feature_std
    return (
        feature_frame,
        display_matrix,
        scaled_matrix,
        resolved_feature_mean,
        resolved_feature_std,
    )


def _resolve_feature_scaler(
    cells: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (feature_mean is None) != (feature_std is None):
        raise ValueError("feature_mean and feature_std must either both be provided or both be omitted")
    if feature_mean is not None and feature_std is not None:
        return (
            np.asarray(feature_mean, dtype=float).copy(),
            np.asarray(feature_std, dtype=float).copy(),
        )

    _validate_required_columns(
        cells,
        feature_columns,
        context="clustering scaler input",
    )
    display_matrix = cells[list(feature_columns)].to_numpy(dtype=float)
    resolved_feature_mean = display_matrix.mean(axis=0)
    resolved_feature_std = display_matrix.std(axis=0)
    resolved_feature_std[~np.isfinite(resolved_feature_std) | (resolved_feature_std == 0)] = 1.0
    return resolved_feature_mean, resolved_feature_std


def _initialize_centroids(
    features: np.ndarray,
    *,
    k: int,
    random_seed: int,
    initial_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.random.Generator]:
    if k < 1 or k > len(features):
        raise ValueError("k must be between 1 and the number of cells")
    rng = np.random.default_rng(random_seed)
    if initial_centroids is not None:
        initial_centroids = np.asarray(initial_centroids, dtype=float)
        if initial_centroids.shape != (k, features.shape[1]):
            raise ValueError(
                "initial_centroids must have shape (k, n_features) matching the clustering features"
            )
        return initial_centroids.copy(), rng
    centroid_index = rng.choice(len(features), size=k, replace=False)
    return features[centroid_index].copy(), rng


def _assign_clusters(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def _recompute_centroids(
    features: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    updated_centroids = centroids.copy()
    for cluster_index in range(len(centroids)):
        members = features[assignments == cluster_index]
        if len(members) == 0:
            updated_centroids[cluster_index] = features[rng.integers(len(features))]
        else:
            updated_centroids[cluster_index] = members.mean(axis=0)
    return updated_centroids


def _summarize_assigned_centroids(
    features: np.ndarray,
    assignments: np.ndarray,
    *,
    reference_centroids: np.ndarray,
) -> np.ndarray:
    summarized_centroids = reference_centroids.copy()
    for cluster_index in range(len(reference_centroids)):
        members = features[assignments == cluster_index]
        if len(members) > 0:
            summarized_centroids[cluster_index] = members.mean(axis=0)
    return summarized_centroids


def _find_best_cluster_match(
    reference_centroids: np.ndarray,
    donor_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_centroids) != len(donor_centroids):
        raise ValueError("reference and donor centroid tables must have the same number of clusters")

    distance_matrix = np.linalg.norm(
        donor_centroids[:, None, :] - reference_centroids[None, :, :],
        axis=2,
    )

    # The notebook intentionally keeps K small, so exhaustive matching stays simple and exact.
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
        raise ValueError("Unable to match centroid sets")

    permutation_array = np.asarray(best_permutation, dtype=int)
    matched_distances = distance_matrix[np.arange(len(permutation_array)), permutation_array]
    return permutation_array, matched_distances


def _align_clusters_to_reference(
    clustered_cells: pd.DataFrame,
    centroid_table: pd.DataFrame,
    *,
    reference_donor_label: str,
    centroid_column_names: tuple[str, ...],
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    max_alignment_distance: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_label_order = ordered_present_values(centroid_table["donor_label"])
    if reference_donor_label not in set(donor_label_order):
        raise ValueError(
            f"reference donor {reference_donor_label!r} is not present in the centroid table"
        )

    reference_rows = centroid_table.loc[
        centroid_table["donor_label"] == reference_donor_label
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
    aligned_centroid_frames: list[pd.DataFrame] = []
    for donor_label in donor_label_order:
        donor_rows = centroid_table.loc[centroid_table["donor_label"] == donor_label].copy()
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
            matched_reference_indices, matched_distances = _find_best_cluster_match(
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
        alignment_distance_map = {
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
        donor_rows["alignment_distance"] = (
            donor_rows["cluster"].astype(str).map(alignment_distance_map).astype(float)
        )
        donor_rows["reference_donor_label"] = str(reference_donor_label)
        aligned_centroid_frames.append(
            donor_rows[
                [
                    "reference_donor_label",
                    "donor_label",
                    "cluster",
                    "aligned_cluster",
                    "alignment_distance",
                    *centroid_column_names,
                ]
            ]
        )

    aligned_clustered_cells = pd.concat(aligned_cluster_frames, ignore_index=True)
    aligned_centroid_table = pd.concat(aligned_centroid_frames, ignore_index=True)
    return aligned_clustered_cells, aligned_centroid_table


def _apply_reference_cluster_labels(
    clustered_cells: pd.DataFrame,
    centroid_table: pd.DataFrame,
    *,
    reference_donor_label: str,
    centroid_column_names: tuple[str, ...],
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
    max_alignment_distance: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_label_order = ordered_present_values(centroid_table["donor_label"])
    reference_rows = centroid_table.loc[
        centroid_table["donor_label"] == reference_donor_label
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
    aligned_centroid_frames: list[pd.DataFrame] = []
    for donor_label in donor_label_order:
        donor_rows = centroid_table.loc[centroid_table["donor_label"] == donor_label].copy()
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

        donor_rows["alignment_distance"] = donor_rows["cluster"].astype(str).map(distance_map).astype(float)
        donor_rows["reference_donor_label"] = str(reference_donor_label)
        donor_rows["aligned_cluster"] = pd.Categorical(
            donor_rows["cluster"].astype(str).map(aligned_cluster_map),
            categories=aligned_categories,
            ordered=True,
        )
        aligned_centroid_frames.append(
            donor_rows[
                [
                    "reference_donor_label",
                    "donor_label",
                    "cluster",
                    "aligned_cluster",
                    "alignment_distance",
                    *centroid_column_names,
                ]
            ]
        )

    aligned_clustered_cells = pd.concat(aligned_cluster_frames, ignore_index=True)
    aligned_centroid_table = pd.concat(aligned_centroid_frames, ignore_index=True)
    return aligned_clustered_cells, aligned_centroid_table


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
        if feature_columns == _DEFAULT_CLUSTER_FEATURE_COLUMNS:
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
    if resolved_mode not in {"warm_start", "reference_fixed", "independent"}:
        raise ValueError(
            "non_reference_mode must be 'warm_start', 'reference_fixed', or 'independent'"
        )
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


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{context} is missing required columns: {missing_text}")
