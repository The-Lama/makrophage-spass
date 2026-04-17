from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "build_aligned_cluster_composition_table",
    "build_donor_feature_data",
    "invert_centroids",
    "ordered_present_values",
    "run_kmeans_with_history",
    "run_split_kmeans",
]

_FEATURE_COLUMNS = ("intensity", "area")
_METADATA_COLUMNS = ("condition", "condition_label", "donor", "donor_label")
_CENTROID_COLUMNS = ("cd206_intensity_centroid", "area_centroid")


def ordered_present_values(series: pd.Series) -> list[Any]:
    non_null = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        present_values = set(non_null.tolist())
        return [value for value in series.cat.categories if value in present_values]
    return list(dict.fromkeys(non_null.tolist()))


def build_donor_feature_data(cells: pd.DataFrame) -> dict[str, dict[str, Any]]:
    donor_feature_data: dict[str, dict[str, Any]] = {}
    donor_label_order = ordered_present_values(cells["donor_label"])
    for donor_label in donor_label_order:
        donor_cells = cells.loc[cells["donor_label"] == donor_label].copy()
        (
            feature_frame,
            display_matrix,
            scaled_matrix,
            feature_mean,
            feature_std,
        ) = _build_feature_matrices(donor_cells)
        donor_feature_data[str(donor_label)] = {
            "feature_frame": feature_frame,
            "display_features": display_matrix,
            "scaled_features": scaled_matrix,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        }
    return donor_feature_data


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
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    centroids, rng = _initialize_centroids(features, k=k, random_seed=random_seed)
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
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    donor_feature_data = build_donor_feature_data(cells)
    donor_runs: dict[str, dict[str, Any]] = {}
    clustered_frames: list[pd.DataFrame] = []
    centroid_tables: list[pd.DataFrame] = []
    cluster_names = [f"Cluster {cluster_index + 1}" for cluster_index in range(k)]

    for donor_label, payload in donor_feature_data.items():
        final_centroids, final_assignments, history = run_kmeans_with_history(
            payload["scaled_features"],
            k=k,
            random_seed=random_seed,
            max_iterations=max_iterations,
        )
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
                columns=list(_CENTROID_COLUMNS),
            )
            .assign(donor_label=donor_label, cluster=cluster_names)
            [["donor_label", "cluster", *_CENTROID_COLUMNS]]
        )
        donor_runs[donor_label] = {
            "history": history,
            "clustered_cells": clustered_cells,
            "centroid_table": centroid_table,
        }
        clustered_frames.append(clustered_cells)
        centroid_tables.append(centroid_table)

    comparison_mean, comparison_std = _build_global_alignment_scaler(cells)
    clustered_cells = pd.concat(clustered_frames, ignore_index=True)
    centroid_table = pd.concat(centroid_tables, ignore_index=True)
    aligned_cells, aligned_centroids = _align_clusters_to_reference(
        clustered_cells,
        centroid_table,
        reference_donor_label=reference_donor_label,
        comparison_mean=comparison_mean,
        comparison_std=comparison_std,
    )
    return donor_feature_data, donor_runs, aligned_cells, aligned_centroids


def _build_feature_matrices(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_frame = cells[[*_METADATA_COLUMNS, *_FEATURE_COLUMNS]].copy()
    display_matrix = feature_frame[list(_FEATURE_COLUMNS)].to_numpy(dtype=float)
    feature_mean = display_matrix.mean(axis=0)
    feature_std = display_matrix.std(axis=0)
    feature_std[~np.isfinite(feature_std) | (feature_std == 0)] = 1.0
    scaled_matrix = (display_matrix - feature_mean) / feature_std
    return feature_frame, display_matrix, scaled_matrix, feature_mean, feature_std


def _build_global_alignment_scaler(cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    display_matrix = cells[list(_FEATURE_COLUMNS)].to_numpy(dtype=float)
    feature_mean = display_matrix.mean(axis=0)
    feature_std = display_matrix.std(axis=0)
    feature_std[~np.isfinite(feature_std) | (feature_std == 0)] = 1.0
    return feature_mean, feature_std


def _initialize_centroids(
    features: np.ndarray,
    *,
    k: int,
    random_seed: int,
) -> tuple[np.ndarray, np.random.Generator]:
    if k < 1 or k > len(features):
        raise ValueError("k must be between 1 and the number of cells")
    rng = np.random.default_rng(random_seed)
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
    comparison_mean: np.ndarray,
    comparison_std: np.ndarray,
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
        .loc[reference_clusters, list(_CENTROID_COLUMNS)]
        .to_numpy(dtype=float)
    )
    reference_centroids = (reference_centroids - comparison_mean) / comparison_std

    aligned_cluster_frames: list[pd.DataFrame] = []
    aligned_centroid_frames: list[pd.DataFrame] = []
    for donor_label in donor_label_order:
        donor_rows = centroid_table.loc[centroid_table["donor_label"] == donor_label].copy()
        donor_clusters = ordered_present_values(donor_rows["cluster"])
        donor_centroids = (
            donor_rows.set_index("cluster")
            .loc[donor_clusters, list(_CENTROID_COLUMNS)]
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
            donor_clusters[donor_index]: reference_clusters[reference_index]
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
            categories=reference_clusters,
            ordered=True,
        )
        aligned_cluster_frames.append(donor_clustered_cells)

        donor_rows["aligned_cluster"] = pd.Categorical(
            donor_rows["cluster"].astype(str).map(aligned_cluster_map),
            categories=reference_clusters,
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
                    *_CENTROID_COLUMNS,
                ]
            ]
        )

    aligned_clustered_cells = pd.concat(aligned_cluster_frames, ignore_index=True)
    aligned_centroid_table = pd.concat(aligned_centroid_frames, ignore_index=True)
    return aligned_clustered_cells, aligned_centroid_table
