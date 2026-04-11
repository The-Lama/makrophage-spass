from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff")
DEFAULT_DATA_ROOT = Path("raw_data")


@dataclass(frozen=True)
class ImageRecord:
    donor: str
    condition: str
    marker_prefix: str
    channel: str
    path: Path

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.donor, self.condition, self.marker_prefix, self.channel)


@dataclass(frozen=True)
class ImageCatalog:
    data_root: Path
    records: tuple[ImageRecord, ...]
    _records_by_key: dict[tuple[str, str, str, str], ImageRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        records_by_key: dict[tuple[str, str, str, str], ImageRecord] = {}
        for record in self.records:
            existing = records_by_key.get(record.key)
            if existing is not None:
                raise ValueError(
                    "Found duplicate image entries for "
                    f"{record.donor} / {record.condition} / {record.marker_prefix} / {record.channel}: "
                    f"{existing.path} and {record.path}"
                )
            records_by_key[record.key] = record
        object.__setattr__(self, "_records_by_key", records_by_key)

    def __len__(self) -> int:
        return len(self.records)

    def find_path(self, donor: str, condition: str, marker_prefix: str, channel: str) -> Path:
        key = (donor, condition, marker_prefix, channel)
        record = self._records_by_key.get(key)
        if record is None:
            condition_dir = self.data_root / donor / condition
            raise FileNotFoundError(
                f"Missing expected image in {condition_dir} for {marker_prefix} {channel}"
            )
        return record.path

    def filter_records(
        self,
        *,
        donors: Iterable[str] | None = None,
        conditions: Iterable[str] | None = None,
    ) -> tuple[ImageRecord, ...]:
        donor_filter = None if donors is None else set(donors)
        condition_filter = None if conditions is None else set(conditions)
        return tuple(
            record
            for record in self.records
            if (donor_filter is None or record.donor in donor_filter)
            and (condition_filter is None or record.condition in condition_filter)
        )


def clear_image_catalog_cache() -> None:
    _build_image_catalog_cached.cache_clear()


def build_image_catalog(data_root: str | Path = DEFAULT_DATA_ROOT) -> ImageCatalog:
    resolved_data_root = Path(data_root).resolve()
    return _build_image_catalog_cached(str(resolved_data_root))


@lru_cache(maxsize=None)
def _build_image_catalog_cached(data_root: str) -> ImageCatalog:
    data_root_path = Path(data_root)
    records: list[ImageRecord] = []
    for image_path in sorted(data_root_path.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        records.append(_record_from_path(data_root_path, image_path))
    return ImageCatalog(data_root=data_root_path, records=tuple(records))


def _record_from_path(data_root: Path, image_path: Path) -> ImageRecord:
    try:
        relative_path = image_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Image path is outside the data root: {image_path}") from exc

    parts = relative_path.parts
    if len(parts) != 3:
        raise ValueError(
            "Expected microscopy images under raw_data/Donor/Condition/<filename>, "
            f"got {relative_path}"
        )

    donor, condition, filename = parts
    stem = Path(filename).stem
    prefix = f"{donor}_{condition}_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"Filename does not match its donor/condition directory: {relative_path}"
        )

    measurement_part = stem[len(prefix) :]
    try:
        marker_prefix, channel = measurement_part.rsplit(" ", 1)
    except ValueError as exc:
        raise ValueError(
            "Expected filenames to end with '<marker prefix> <channel>' after donor/condition, "
            f"got {relative_path}"
        ) from exc

    if not marker_prefix or not channel:
        raise ValueError(f"Malformed microscopy filename: {relative_path}")

    return ImageRecord(
        donor=donor,
        condition=condition,
        marker_prefix=marker_prefix,
        channel=channel,
        path=image_path,
    )
