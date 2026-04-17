from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MEASUREMENT_SCALE_RAW_INTENSITY = "raw_intensity"
MEASUREMENT_SCALE_BACKGROUND_RATIO = "background_ratio"

__all__ = [
    "AntibodySpec",
    "BatchAnalysis",
    "ComparisonDonorResult",
    "ConditionComparison",
    "MEASUREMENT_SCALE_BACKGROUND_RATIO",
    "MEASUREMENT_SCALE_RAW_INTENSITY",
    "MeasurementResult",
    "ResultKey",
]


@dataclass(frozen=True)
class AntibodySpec:
    marker_prefix: str
    channel: str


@dataclass(frozen=True)
class ResultKey:
    antibody: str
    condition: str
    donor: str


@dataclass
class ComparisonDonorResult:
    path: Path
    image: "np.ndarray"
    labels: "np.ndarray"
    mean_intensities: "np.ndarray"
    cell_count: int
    overall_mean: float
    overall_median: float


@dataclass
class ConditionComparison:
    condition: str
    donors: list[str]
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
    solidities: "np.ndarray | None" = None
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
    results: dict[ResultKey, MeasurementResult]

    def get_result(self, antibody: str, condition: str, donor: str) -> MeasurementResult:
        key = ResultKey(antibody=antibody, condition=condition, donor=donor)
        try:
            return self.results[key]
        except KeyError as exc:
            raise KeyError(
                f"Missing analysis result for antibody={antibody}, condition={condition}, donor={donor}"
            ) from exc
