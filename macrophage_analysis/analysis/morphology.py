from __future__ import annotations

import numpy as np
import pandas as pd


def sample_morphology_points(
    morphology_df: pd.DataFrame,
    *,
    max_points_per_group: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if max_points_per_group is not None and max_points_per_group < 1:
        raise ValueError("max_points_per_group must be at least 1 or None")

    rng = np.random.default_rng(random_seed)
    sampled_frames: list[pd.DataFrame] = []
    for _, group in morphology_df.groupby(["condition", "donor"], observed=True):
        if max_points_per_group is not None and len(group) > max_points_per_group:
            sampled_frames.append(
                group.iloc[rng.choice(len(group), size=max_points_per_group, replace=False)]
            )
        else:
            sampled_frames.append(group)
    return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else morphology_df.iloc[:0].copy()


def filter_morphology_points(
    plot_df: pd.DataFrame,
    *,
    x_column: str,
    x_upper_percentile: float,
    area_upper_percentile: float,
) -> pd.DataFrame:
    if not 0 < x_upper_percentile <= 100:
        raise ValueError("x_upper_percentile must be between 0 and 100")
    if not 0 < area_upper_percentile <= 100:
        raise ValueError("area_upper_percentile must be between 0 and 100")
    x_max = float(np.percentile(plot_df[x_column], x_upper_percentile))
    area_max = float(np.percentile(plot_df["area"], area_upper_percentile))
    return plot_df.loc[(plot_df[x_column] <= x_max) & (plot_df["area"] <= area_max)].copy()


def prepare_morphology_scatter_points(
    morphology_df: pd.DataFrame,
    *,
    max_points_per_group: int | None,
    random_seed: int,
    x_column: str,
    x_upper_percentile: float,
    area_upper_percentile: float,
) -> pd.DataFrame:
    sampled_df = sample_morphology_points(
        morphology_df,
        max_points_per_group=max_points_per_group,
        random_seed=random_seed,
    )
    return filter_morphology_points(
        sampled_df,
        x_column=x_column,
        x_upper_percentile=x_upper_percentile,
        area_upper_percentile=area_upper_percentile,
    )
