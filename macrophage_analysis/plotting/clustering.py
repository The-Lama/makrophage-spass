from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..defaults import DEFAULT_SEABORN_THEME
from ..analysis.clustering import invert_centroids, ordered_present_values

__all__ = [
    "plot_aligned_cluster_composition",
    "plot_cluster_feature_pairs",
    "plot_kmeans_state_grid",
]


def plot_kmeans_state_grid(
    donor_feature_data: dict[str, dict[str, object]],
    state_by_donor: dict[str, dict[str, object]],
    *,
    antibody: str,
    k: int,
) -> list[str]:
    donor_order = list(donor_feature_data)
    palette = sns.color_palette("deep", n_colors=k)

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    fig, axes = plt.subplots(
        1,
        len(donor_order),
        figsize=(7.0 * len(donor_order), 6.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(donor_order) == 1:
        axes = [axes]

    count_lines: list[str] = []
    for ax, donor_label in zip(axes, donor_order):
        payload = donor_feature_data[donor_label]
        state = state_by_donor[donor_label]
        centroid_display = invert_centroids(
            np.asarray(state["centroids"], dtype=float),
            np.asarray(payload["feature_mean"], dtype=float),
            np.asarray(payload["feature_std"], dtype=float),
        )
        assignments = np.asarray(state["assignments"], dtype=int)
        display_features = np.asarray(payload["display_features"], dtype=float)
        if display_features.shape[1] != 2:
            raise ValueError("plot_kmeans_state_grid requires donor feature data with exactly two display features")

        ax.scatter(
            display_features[:, 0],
            display_features[:, 1],
            c=[palette[index] for index in assignments],
            s=14,
            alpha=0.28,
            linewidth=0,
        )
        ax.scatter(
            centroid_display[:, 0],
            centroid_display[:, 1],
            c=palette,
            s=280,
            marker="X",
            edgecolor="black",
            linewidth=1.2,
        )
        for cluster_index, (x_pos, y_pos) in enumerate(centroid_display, start=1):
            ax.text(
                x_pos,
                y_pos,
                f"  C{cluster_index}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
        phase_label = "Assign cells" if state["phase"] == "assignment" else "Move centroids"
        shift = float(state["shift"])
        shift_label = "" if np.isnan(shift) else f" | shift={shift:.3f}"
        ax.set_title(
            f"{donor_label} Iteration {state['iteration']}: {phase_label}{shift_label}",
            fontsize=12,
        )
        ax.set_xlabel(f"Per-cell mean {antibody} fluorescence intensity")
        ax.set_ylabel("Cell area (pixels)" if ax is axes[0] else "")
        ax.grid(alpha=0.22)
        sns.despine(ax=ax)

        counts = np.bincount(assignments, minlength=k)
        count_lines.append(
            f"{donor_label}: "
            + ", ".join(
                [
                    f"Cluster {cluster_index + 1}={count}"
                    for cluster_index, count in enumerate(counts)
                ]
            )
        )

    plt.show()
    return count_lines


def plot_cluster_feature_pairs(
    clustered_cells: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    cluster_column: str = "aligned_cluster",
    donor_column: str = "donor_label",
    feature_labels: Mapping[str, str] | None = None,
    max_points_per_donor: int | None = 3000,
    random_seed: int = 7,
    height: float = 2.8,
    alpha: float = 0.45,
    corner: bool = True,
) -> pd.DataFrame:
    resolved_feature_columns = tuple(str(column) for column in feature_columns)
    if len(resolved_feature_columns) < 2:
        raise ValueError("feature_columns must contain at least two features for a pair plot")

    required_columns = {donor_column, cluster_column, *resolved_feature_columns}
    missing_columns = sorted(required_columns - set(clustered_cells.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"clustered_cells is missing required columns: {missing_text}")
    if max_points_per_donor is not None and max_points_per_donor < 1:
        raise ValueError("max_points_per_donor must be at least 1 when provided")

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    sampled_frames: list[pd.DataFrame] = []
    donor_order = ordered_present_values(clustered_cells[donor_column])
    cluster_order = ordered_present_values(clustered_cells[cluster_column])
    display_feature_labels = [
        feature_labels.get(column, column) if feature_labels is not None else column
        for column in resolved_feature_columns
    ]

    for donor_label in donor_order:
        donor_df = clustered_cells.loc[clustered_cells[donor_column] == donor_label].copy()
        if max_points_per_donor is not None and len(donor_df) > max_points_per_donor:
            donor_df = donor_df.sample(
                n=max_points_per_donor,
                random_state=random_seed,
            ).sort_index()
        sampled_frames.append(donor_df)

        plot_df = donor_df[[*resolved_feature_columns, cluster_column]].rename(
            columns=dict(zip(resolved_feature_columns, display_feature_labels))
        )
        grid = sns.pairplot(
            plot_df,
            vars=display_feature_labels,
            hue=cluster_column,
            hue_order=cluster_order,
            corner=corner,
            height=height,
            diag_kind="hist",
            plot_kws={"alpha": alpha, "s": 16, "linewidth": 0},
            diag_kws={"alpha": 0.65},
        )
        grid.figure.suptitle(
            f"{donor_label}: reference-aligned clusters",
            fontsize=14,
            y=1.02,
        )
        plt.show()

    return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else clustered_cells.iloc[0:0].copy()


def plot_aligned_cluster_composition(
    cluster_composition: pd.DataFrame,
    *,
    reference_donor_label: str,
    cluster_count: int | None = None,
) -> pd.DataFrame:
    required_columns = {"donor_label", "condition_label", "aligned_cluster", "fraction"}
    missing_columns = sorted(required_columns - set(cluster_composition.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"cluster_composition is missing required columns: {missing_text}")

    donor_label_order = ordered_present_values(cluster_composition["donor_label"])
    condition_label_order = ordered_present_values(cluster_composition["condition_label"])
    aligned_cluster_order = ordered_present_values(cluster_composition["aligned_cluster"])
    resolved_cluster_count = len(aligned_cluster_order) if cluster_count is None else cluster_count

    sns.set_theme(**DEFAULT_SEABORN_THEME)
    fig, axes = plt.subplots(
        1,
        len(donor_label_order),
        figsize=(6.5 * len(donor_label_order), 5.6),
        sharey=True,
        constrained_layout=True,
    )
    if len(donor_label_order) == 1:
        axes = [axes]

    palette = sns.color_palette("deep", n_colors=len(aligned_cluster_order))
    for ax, donor_label in zip(axes, donor_label_order):
        donor_composition = cluster_composition.loc[
            cluster_composition["donor_label"] == donor_label
        ]
        donor_pivot = donor_composition.pivot(
            index="condition_label",
            columns="aligned_cluster",
            values="fraction",
        ).fillna(0.0)
        donor_pivot = donor_pivot.reindex(
            index=condition_label_order,
            columns=aligned_cluster_order,
            fill_value=0.0,
        )
        bottom = np.zeros(len(donor_pivot), dtype=float)
        for cluster_name, color in zip(aligned_cluster_order, palette):
            values = donor_pivot[cluster_name].to_numpy(dtype=float)
            ax.bar(
                donor_pivot.index.astype(str),
                values,
                bottom=bottom,
                color=color,
                label=str(cluster_name),
            )
            bottom += values
        ax.set_title(str(donor_label), fontsize=12, fontweight="bold")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Aligned cluster fraction" if ax is axes[0] else "")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)

    axes[-1].legend(frameon=False, title=f"Aligned to {reference_donor_label}")
    fig.suptitle(
        f"Reference-aligned donor-specific cluster composition by treatment (K={resolved_cluster_count})",
        fontsize=14,
    )
    plt.show()
    return cluster_composition
