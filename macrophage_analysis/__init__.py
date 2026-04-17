from __future__ import annotations

from importlib import import_module
import warnings

_NOTEBOOK_EXPORT_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        ".defaults",
        "macrophage_analysis.defaults",
        (
            "DEFAULT_ANTIBODY_ORDER",
            "DEFAULT_ANTIBODY_SPECS",
            "DEFAULT_BASELINE_CONDITION",
            "DEFAULT_COMPARISON_CHANNEL",
            "DEFAULT_COMPARISON_MARKER_PREFIX",
            "DEFAULT_CONDITIONS",
            "DEFAULT_DONORS",
        ),
    ),
    (
        ".analysis.fluorescence_tables",
        "macrophage_analysis.analysis.fluorescence_tables",
        (
            "build_stat_summary_table",
        ),
    ),
    (
        ".analysis.morphology_tables",
        "macrophage_analysis.analysis.morphology_tables",
        (
            "build_morphology_fold_change_table",
            "build_morphology_summary_table",
        ),
    ),
    (
        ".analysis.pipelines",
        "macrophage_analysis.analysis.pipelines",
        (
            "extract_condition_comparison",
            "extract_single_cell_fluorescence",
            "extract_single_cell_relative_fluorescence",
        ),
    ),
    (
        ".plotting.tables",
        "macrophage_analysis.plotting.tables",
        (
            "display_stat_summary_tables",
        ),
    ),
    (
        ".plotting.morphology",
        "macrophage_analysis.plotting.morphology",
        (
            "plot_area_intensity_scatter",
            "plot_morphology_heatmap",
            "plot_morphology_scatter",
        ),
    ),
    (
        ".plotting.fluorescence_summary",
        "macrophage_analysis.plotting.fluorescence_summary",
        (
            "plot_summary_heatmap",
            "plot_violins_with_stats",
        ),
    ),
    (
        ".plotting.fluorescence_debug",
        "macrophage_analysis.plotting.fluorescence_debug",
        (
            "plot_condition_brightness_debug",
            "plot_condition_comparison",
            "plot_condition_histograms",
            "plot_condition_relative_background_debug",
        ),
    ),
    (
        ".io",
        "macrophage_analysis.io",
        (
            "sort_microscopy_images",
        ),
    ),
)

_COMPAT_EXPORT_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        ".models",
        "macrophage_analysis.models",
        (
            "AntibodySpec",
            "BatchAnalysis",
            "ComparisonDonorResult",
            "ConditionComparison",
            "MEASUREMENT_SCALE_BACKGROUND_RATIO",
            "MEASUREMENT_SCALE_RAW_INTENSITY",
            "MeasurementResult",
            "ResultKey",
        ),
    ),
    (
        ".defaults",
        "macrophage_analysis.defaults",
        (
            "DEFAULT_CONDITION_DISPLAY_LABELS",
            "DEFAULT_DATA_ROOT",
            "DEFAULT_DONOR_COLORS",
            "DEFAULT_DONOR_DISPLAY_LABELS",
            "DEFAULT_SEABORN_THEME",
            "DEFAULT_STARDIST_N_TILES",
        ),
    ),
    (
        ".io",
        "macrophage_analysis.io",
        (
            "build_image_catalog",
            "clear_image_catalog_cache",
        ),
    ),
    (
        ".analysis.morphology_tables",
        "macrophage_analysis.analysis.morphology_tables",
        (
            "build_morphology_table",
        ),
    ),
    (
        ".analysis.extraction",
        "macrophage_analysis.analysis.extraction",
        (
            "find_image_path",
            "load_grayscale_tif",
            "load_stardist_model",
        ),
    ),
    (
        ".stats",
        "macrophage_analysis.stats",
        (
            "mann_whitney_u_test",
        ),
    ),
)

__all__ = [
    name
    for _, _, exported_names in _NOTEBOOK_EXPORT_GROUPS
    for name in exported_names
]

_NOTEBOOK_EXPORTS: dict[str, tuple[str, str, str]] = {
    name: (module_name, name, public_module)
    for module_name, public_module, exported_names in _NOTEBOOK_EXPORT_GROUPS
    for name in exported_names
}

_COMPAT_EXPORTS: dict[str, tuple[str, str, str]] = {
    name: (module_name, name, public_module)
    for module_name, public_module, exported_names in _COMPAT_EXPORT_GROUPS
    for name in exported_names
}

_WARNED_COMPAT_EXPORTS: set[str] = set()


def __getattr__(name: str):
    if name in _NOTEBOOK_EXPORTS:
        module_name, attribute_name, _ = _NOTEBOOK_EXPORTS[name]
    else:
        try:
            module_name, attribute_name, public_module = _COMPAT_EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        if name not in _WARNED_COMPAT_EXPORTS:
            warnings.warn(
                f"`macrophage_analysis.{name}` is kept for compatibility but is not part of the "
                f"notebook-first top-level API. Import it from `{public_module}` instead.",
                FutureWarning,
                stacklevel=2,
            )
            _WARNED_COMPAT_EXPORTS.add(name)

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
