from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..config import AntibodySpec, DEFAULT_DATA_ROOT
from ..io.catalog import build_image_catalog


@dataclass(frozen=True)
class MeasurementRequest:
    antibody: str
    donor: str
    condition: str
    path: Path


def build_measurement_requests(
    donors: Iterable[str],
    conditions: Iterable[str],
    *,
    antibody_specs: Mapping[str, AntibodySpec],
    antibody_order: Iterable[str],
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> list[MeasurementRequest]:
    donor_list = list(donors)
    condition_list = list(conditions)
    order_list = list(antibody_order)
    catalog = build_image_catalog(data_root)
    requests: list[MeasurementRequest] = []
    for antibody in order_list:
        spec = antibody_specs[antibody]
        for condition in condition_list:
            for donor in donor_list:
                requests.append(
                    MeasurementRequest(
                        antibody=antibody,
                        donor=donor,
                        condition=condition,
                        path=catalog.find_path(donor, condition, spec.marker_prefix, spec.channel),
                    )
                )
    return requests


def build_condition_comparison_requests(
    donors: Iterable[str],
    condition: str,
    *,
    marker_prefix: str,
    channel: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> list[MeasurementRequest]:
    donor_list = list(donors)
    catalog = build_image_catalog(data_root)
    return [
        MeasurementRequest(
            antibody=f"{marker_prefix} {channel}",
            donor=donor,
            condition=condition,
            path=catalog.find_path(donor, condition, marker_prefix, channel),
        )
        for donor in donor_list
    ]
