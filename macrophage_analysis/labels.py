from __future__ import annotations

import re
import textwrap

from .defaults import (
    DEFAULT_CONDITION_DISPLAY_LABELS,
    DEFAULT_DESCRIPTIVE_CONDITION_DISPLAY_LABELS,
)
from .models import (
    MEASUREMENT_SCALE_BACKGROUND_RATIO,
    MEASUREMENT_SCALE_RAW_INTENSITY,
)

__all__ = [
    "default_condition_display_label",
    "default_measurement_axis_label",
    "default_measurement_summary_label",
    "default_measurement_title_label",
    "validate_measurement_scale",
    "wrap_display_label",
]


def wrap_display_label(label: str, *, width: int | None = None) -> str:
    if width is None:
        return label
    if width < 1:
        raise ValueError("width must be at least 1")
    return textwrap.fill(label, width=width, break_long_words=False)


def default_condition_display_label(
    condition: str,
    *,
    style: str = "short",
    wrap_width: int | None = None,
) -> str:
    if style == "short":
        condition_labels = DEFAULT_CONDITION_DISPLAY_LABELS
    elif style == "descriptive":
        condition_labels = DEFAULT_DESCRIPTIVE_CONDITION_DISPLAY_LABELS
    else:
        raise ValueError("style must be 'short' or 'descriptive'")

    if condition in condition_labels:
        return wrap_display_label(condition_labels[condition], width=wrap_width)
    if condition in DEFAULT_CONDITION_DISPLAY_LABELS:
        return wrap_display_label(DEFAULT_CONDITION_DISPLAY_LABELS[condition], width=wrap_width)

    label = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", condition)
    label = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", label)
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return wrap_display_label(label, width=wrap_width)


def validate_measurement_scale(measurement_scale: str) -> str:
    if measurement_scale not in {
        MEASUREMENT_SCALE_RAW_INTENSITY,
        MEASUREMENT_SCALE_BACKGROUND_RATIO,
    }:
        raise ValueError(f"Unsupported measurement_scale: {measurement_scale}")
    return measurement_scale


def default_measurement_axis_label(measurement_scale: str) -> str:
    validate_measurement_scale(measurement_scale)
    if measurement_scale == MEASUREMENT_SCALE_BACKGROUND_RATIO:
        return "Cell signal / image background"
    return "Per-cell fluorescence intensity"


def default_measurement_title_label(measurement_scale: str) -> str:
    validate_measurement_scale(measurement_scale)
    if measurement_scale == MEASUREMENT_SCALE_BACKGROUND_RATIO:
        return "signal/background ratio"
    return "single-cell fluorescence"


def default_measurement_summary_label(measurement_scale: str, *, summary_stat: str) -> str:
    validate_measurement_scale(measurement_scale)
    if measurement_scale == MEASUREMENT_SCALE_BACKGROUND_RATIO:
        return f"{summary_stat} signal/background ratio"
    return f"{summary_stat} single-cell intensity"
