from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def build_heatmap_annotation_labels(
    values: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    star_getter: Callable[[int, int], str | None] | None = None,
) -> np.ndarray:
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("Heatmap annotation shape does not match the data shape")

    annotations = np.empty(values.shape, dtype=object)
    for row_index, _ in enumerate(row_labels):
        for column_index, _ in enumerate(column_labels):
            value = values[row_index, column_index]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            if star_getter is not None:
                stars = star_getter(row_index, column_index)
                if stars not in {None, "baseline", "NA", "ns"}:
                    label = f"{label}\n{stars}"
            annotations[row_index, column_index] = label
    return annotations
