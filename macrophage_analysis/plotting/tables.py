from __future__ import annotations

from typing import Iterable, Mapping

from ..defaults import DEFAULT_ANTIBODY_ORDER, DEFAULT_BASELINE_CONDITION

__all__ = ["display_stat_summary_tables"]


def display_stat_summary_tables(
    stat_table,
    *,
    antibody_order: Iterable[str] = DEFAULT_ANTIBODY_ORDER,
    baseline_condition: str = DEFAULT_BASELINE_CONDITION,
    condition_labels: Mapping[str, str] | None = None,
    donor_labels: Mapping[str, str] | None = None,
) -> None:
    from ..analysis.fluorescence_tables import format_stat_summary_tables

    formatted_tables = format_stat_summary_tables(
        stat_table,
        antibody_order=antibody_order,
        baseline_condition=baseline_condition,
        condition_labels=condition_labels,
        donor_labels=donor_labels,
    )
    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown = None
        display = None

    for antibody, formatted_stats in formatted_tables:
        if Markdown is not None and display is not None:
            display(Markdown(f"**{antibody}**"))
            display(Markdown(f"```text\n{formatted_stats.to_string(index=False)}\n```"))
        else:
            print(f"\n{antibody}")
            print(formatted_stats.to_string(index=False))
