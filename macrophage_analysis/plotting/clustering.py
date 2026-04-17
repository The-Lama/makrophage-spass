from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..defaults import DEFAULT_SEABORN_THEME
from ..analysis.clustering import ordered_present_values

__all__ = [
    "plot_aligned_cluster_composition",
]


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
