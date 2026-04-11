from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("raw_data")
DEFAULT_BASELINE_CONDITION = "M0"
MEASUREMENT_SCALE_RAW_INTENSITY = "raw_intensity"
MEASUREMENT_SCALE_BACKGROUND_RATIO = "background_ratio"
DEFAULT_DONORS = ("D45", "D47")
DEFAULT_CONDITIONS = (
    "M0",
    "B68KCP2",
    "B75palmFP1",
    "M130227Ni",
    "M130429Np",
    "T12CMNepiP4",
    "T8CMNdermP6",
)
DEFAULT_COMPARISON_MARKER_PREFIX = "CD200R_A488_CD206_A568_10x_RGB_Qi2 10x"
DEFAULT_COMPARISON_CHANNEL = "TRITC"
DEFAULT_STARDIST_N_TILES = (2, 2)
DEFAULT_DONOR_COLORS = {"D45": "#0072B2", "D47": "#D55E00"}
DEFAULT_SEABORN_THEME = {"style": "whitegrid"}
DEFAULT_DONOR_DISPLAY_LABELS = {donor: f"Donor {donor}" for donor in DEFAULT_DONORS}
DEFAULT_CONDITION_DISPLAY_LABELS = {
    "M0": "M0 (baseline)",
    "B68KCP2": "B68 KCP2",
    "B75palmFP1": "B75 palm FP1",
    "M130227Ni": "M130227 Ni",
    "M130429Np": "M130429 Np",
    "T12CMNepiP4": "T12 CMN epi P4",
    "T8CMNdermP6": "T8 CMN derm P6",
}


@dataclass(frozen=True)
class AntibodySpec:
    marker_prefix: str
    channel: str


DEFAULT_ANTIBODY_SPECS: dict[str, AntibodySpec] = {
    "CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC"),
    "CD86": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "FITC"),
    "iNOS": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "TRITC"),
    "CD200R": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "FITC"),
    "CD206": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "TRITC"),
}
DEFAULT_ANTIBODY_ORDER = tuple(DEFAULT_ANTIBODY_SPECS.keys())


@dataclass
class ComparisonDonorResult:
    path: Path
    image: "np.ndarray"
    labels: "np.ndarray"
    details: dict
    mean_intensities: "np.ndarray"
    cell_count: int
    overall_mean: float
    overall_median: float
    areas: "np.ndarray | None" = None
    eccentricities: "np.ndarray | None" = None


@dataclass
class ConditionComparison:
    condition: str
    donors: list[str]
    marker_prefix: str
    channel: str
    donor_results: dict[str, ComparisonDonorResult]


@dataclass
class MeasurementResult:
    path: Path
    cell_count: int
    raw_mean_intensities: "np.ndarray"
    measurement_values: "np.ndarray"
    measurement_mean: float
    measurement_median: float
    measurement_scale: str = MEASUREMENT_SCALE_RAW_INTENSITY
    areas: "np.ndarray | None" = None
    eccentricities: "np.ndarray | None" = None
    background_intensity: float | None = None
    background_percentile: float | None = None

    @property
    def mean_intensities(self) -> "np.ndarray":
        return self.measurement_values

    @property
    def overall_mean(self) -> float:
        return self.measurement_mean

    @property
    def overall_median(self) -> float:
        return self.measurement_median


@dataclass
class BatchAnalysis:
    donors: list[str]
    conditions: list[str]
    baseline_condition: str
    antibody_order: list[str]
    antibody_specs: dict[str, AntibodySpec]
    donor_colors: dict[str, str]
    measurement_scale: str
    results: dict[tuple[str, str, str], MeasurementResult]


def default_condition_display_label(condition: str) -> str:
    if condition in DEFAULT_CONDITION_DISPLAY_LABELS:
        return DEFAULT_CONDITION_DISPLAY_LABELS[condition]

    label = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", condition)
    label = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", label)
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return label


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
