from __future__ import annotations

from importlib import import_module

_MODULE_EXPORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        ".config",
        (
            "AntibodySpec",
            "BatchAnalysis",
            "ComparisonDonorResult",
            "ConditionComparison",
            "DEFAULT_ANTIBODY_ORDER",
            "DEFAULT_ANTIBODY_SPECS",
            "DEFAULT_BASELINE_CONDITION",
            "DEFAULT_COMPARISON_CHANNEL",
            "DEFAULT_COMPARISON_MARKER_PREFIX",
            "DEFAULT_CONDITIONS",
            "DEFAULT_CONDITION_DISPLAY_LABELS",
            "DEFAULT_DATA_ROOT",
            "DEFAULT_DONOR_COLORS",
            "DEFAULT_DONORS",
            "DEFAULT_DONOR_DISPLAY_LABELS",
            "DEFAULT_SEABORN_THEME",
            "DEFAULT_STARDIST_N_TILES",
            "MEASUREMENT_SCALE_BACKGROUND_RATIO",
            "MEASUREMENT_SCALE_RAW_INTENSITY",
            "MeasurementResult",
            "ResultKey",
        ),
    ),
    (
        ".io",
        (
            "build_image_catalog",
            "clear_image_catalog_cache",
            "sort_microscopy_images",
        ),
    ),
    (
        ".analysis.fluorescence_tables",
        (
            "build_stat_summary_table",
        ),
    ),
    (
        ".analysis.morphology_tables",
        (
            "build_morphology_fold_change_table",
            "build_morphology_summary_table",
            "build_morphology_table",
        ),
    ),
    (
        ".analysis.pipelines",
        (
            "extract_condition_comparison",
            "extract_single_cell_fluorescence",
            "extract_single_cell_relative_fluorescence",
        ),
    ),
    (
        ".analysis.extraction",
        (
            "find_image_path",
            "load_grayscale_tif",
            "load_stardist_model",
        ),
    ),
    (
        ".plotting.tables",
        (
            "display_stat_summary_tables",
        ),
    ),
    (
        ".plotting.morphology",
        (
            "plot_area_intensity_scatter",
            "plot_morphology_heatmap",
            "plot_morphology_scatter",
        ),
    ),
    (
        ".plotting.fluorescence_summary",
        (
            "plot_summary_heatmap",
            "plot_violins_with_stats",
        ),
    ),
    (
        ".plotting.fluorescence_debug",
        (
            "plot_condition_brightness_debug",
            "plot_condition_comparison",
            "plot_condition_histograms",
            "plot_condition_relative_background_debug",
        ),
    ),
    (
        ".stats",
        (
            "mann_whitney_u_test",
        ),
    ),
)

__all__ = [
    name
    for _, exported_names in _MODULE_EXPORTS
    for name in exported_names
]

_EXPORTS: dict[str, tuple[str, str]] = {
    name: (module_name, name)
    for module_name, exported_names in _MODULE_EXPORTS
    for name in exported_names
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
