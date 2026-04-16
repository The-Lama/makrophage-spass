from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import pandas as pd


def _wrap_filename(filename: str, *, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(filename, width=width, break_long_words=True))


def _deduplicate_legend(ax: plt.Axes, donors: list[str] | tuple[str, ...] | set[str]) -> None:
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


def _build_stat_star_lookup(stat_results: pd.DataFrame | None) -> dict[tuple[str, str, str], str]:
    if stat_results is None or stat_results.empty:
        return {}
    return {
        (str(row.donor), str(row.condition), str(row.antibody)): str(row.stars)
        for row in stat_results.itertuples(index=False)
    }
