from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile as tiff
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from stardist.models import StarDist2D

from ..config import DEFAULT_DATA_ROOT
from ..io.catalog import build_image_catalog


@lru_cache(maxsize=1)
def load_stardist_model(model_name: str = "2D_versatile_fluo") -> StarDist2D:
    return StarDist2D.from_pretrained(model_name)


def find_image_path(
    donor: str,
    condition: str,
    marker_prefix: str,
    channel: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Path:
    catalog = build_image_catalog(data_root=data_root)
    return catalog.find_path(
        donor=donor,
        condition=condition,
        marker_prefix=marker_prefix,
        channel=channel,
    )


def load_grayscale_tif(image_path: str | Path) -> np.ndarray:
    tifffile_logger = logging.getLogger("tifffile")
    original_tifffile_level = tifffile_logger.level

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*OME series cannot handle discontiguous storage.*",
        )
        tifffile_logger.setLevel(logging.ERROR)
        try:
            image = tiff.imread(Path(image_path))
        finally:
            tifffile_logger.setLevel(original_tifffile_level)

    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def predict_stardist_labels(
    image: np.ndarray,
    *,
    model_name: str = "2D_versatile_fluo",
    n_tiles: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    model = load_stardist_model(model_name)
    normalized_image = normalize(image, 1, 99.8, axis=(0, 1))
    return model.predict_instances(normalized_image, n_tiles=n_tiles)


def extract_cell_measurements(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = regionprops_table(
        labels,
        intensity_image=image,
        properties=("intensity_mean", "area", "eccentricity"),
    )
    return (
        np.asarray(props["intensity_mean"], dtype=float),
        np.asarray(props["area"], dtype=float),
        np.asarray(props["eccentricity"], dtype=float),
    )


def summarize_intensities(mean_intensities: np.ndarray) -> tuple[int, float, float]:
    count = int(mean_intensities.size)
    finite_intensities = np.asarray(mean_intensities, dtype=float)
    finite_intensities = finite_intensities[np.isfinite(finite_intensities)]
    overall_mean = float(finite_intensities.mean()) if finite_intensities.size else float("nan")
    overall_median = float(np.median(finite_intensities)) if finite_intensities.size else float("nan")
    return count, overall_mean, overall_median


def estimate_background_intensity(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    background_percentile: float,
) -> float:
    background_pixels = np.asarray(image[labels == 0], dtype=float)
    background_pixels = background_pixels[np.isfinite(background_pixels)]
    if background_pixels.size == 0:
        image_pixels = np.asarray(image, dtype=float)
        background_pixels = image_pixels[np.isfinite(image_pixels)]
    if background_pixels.size == 0:
        return float("nan")
    return float(np.percentile(background_pixels, background_percentile))


def divide_by_background(values: np.ndarray, background_intensity: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(background_intensity) or background_intensity <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return values / background_intensity
