from __future__ import annotations

from collections.abc import Mapping, Sequence

from .defaults import DEFAULT_ANTIBODY_ORDER

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - optional notebook dependency
    widgets = None

__all__ = [
    "build_clustering_feature_labels",
    "build_marker_dropdown",
]


def build_marker_dropdown(
    *,
    options: Sequence[str] = DEFAULT_ANTIBODY_ORDER,
    default_marker: str = "CD206",
    description: str = "Marker",
) -> "widgets.Dropdown":
    if widgets is None:
        raise ImportError("ipywidgets is required to build notebook dropdown controls")

    resolved_options = tuple(str(option) for option in options)
    if not resolved_options:
        raise ValueError("options must contain at least one antibody name")

    resolved_default_marker = (
        str(default_marker)
        if str(default_marker) in set(resolved_options)
        else resolved_options[0]
    )
    return widgets.Dropdown(
        options=resolved_options,
        value=resolved_default_marker,
        description=str(description),
    )


def build_clustering_feature_labels(
    antibody: str,
    *,
    extra_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    labels = {
        "intensity": f"{antibody} brightness",
    }
    if extra_labels is not None:
        labels.update({str(key): str(value) for key, value in extra_labels.items()})
    return labels
