from __future__ import annotations

from importlib import import_module

__all__ = [
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
    "MEASUREMENT_SCALE_BACKGROUND_RATIO",
    "MEASUREMENT_SCALE_RAW_INTENSITY",
    "DEFAULT_SEABORN_THEME",
    "DEFAULT_STARDIST_N_TILES",
    "MeasurementResult",
    "ResultKey",
    "build_image_catalog",
    "build_morphology_fold_change_table",
    "build_morphology_summary_table",
    "build_morphology_table",
    "build_stat_summary_table",
    "clear_image_catalog_cache",
    "display_stat_summary_tables",
    "extract_condition_comparison",
    "extract_single_cell_fluorescence",
    "extract_single_cell_relative_fluorescence",
    "find_image_path",
    "load_grayscale_tif",
    "load_stardist_model",
    "mann_whitney_u_test",
    "plot_area_intensity_scatter",
    "plot_condition_brightness_debug",
    "plot_condition_comparison",
    "plot_condition_histograms",
    "plot_condition_relative_background_debug",
    "plot_morphology_heatmap",
    "plot_morphology_scatter",
    "plot_summary_heatmap",
    "plot_violins_with_stats",
    "sort_microscopy_images",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AntibodySpec": (".config", "AntibodySpec"),
    "BatchAnalysis": (".config", "BatchAnalysis"),
    "ComparisonDonorResult": (".config", "ComparisonDonorResult"),
    "ConditionComparison": (".config", "ConditionComparison"),
    "DEFAULT_ANTIBODY_ORDER": (".config", "DEFAULT_ANTIBODY_ORDER"),
    "DEFAULT_ANTIBODY_SPECS": (".config", "DEFAULT_ANTIBODY_SPECS"),
    "DEFAULT_BASELINE_CONDITION": (".config", "DEFAULT_BASELINE_CONDITION"),
    "DEFAULT_COMPARISON_CHANNEL": (".config", "DEFAULT_COMPARISON_CHANNEL"),
    "DEFAULT_COMPARISON_MARKER_PREFIX": (".config", "DEFAULT_COMPARISON_MARKER_PREFIX"),
    "DEFAULT_CONDITIONS": (".config", "DEFAULT_CONDITIONS"),
    "DEFAULT_CONDITION_DISPLAY_LABELS": (".config", "DEFAULT_CONDITION_DISPLAY_LABELS"),
    "DEFAULT_DATA_ROOT": (".config", "DEFAULT_DATA_ROOT"),
    "DEFAULT_DONOR_COLORS": (".config", "DEFAULT_DONOR_COLORS"),
    "DEFAULT_DONORS": (".config", "DEFAULT_DONORS"),
    "DEFAULT_DONOR_DISPLAY_LABELS": (".config", "DEFAULT_DONOR_DISPLAY_LABELS"),
    "MEASUREMENT_SCALE_BACKGROUND_RATIO": (".config", "MEASUREMENT_SCALE_BACKGROUND_RATIO"),
    "MEASUREMENT_SCALE_RAW_INTENSITY": (".config", "MEASUREMENT_SCALE_RAW_INTENSITY"),
    "DEFAULT_SEABORN_THEME": (".config", "DEFAULT_SEABORN_THEME"),
    "DEFAULT_STARDIST_N_TILES": (".config", "DEFAULT_STARDIST_N_TILES"),
    "MeasurementResult": (".config", "MeasurementResult"),
    "ResultKey": (".config", "ResultKey"),
    "build_image_catalog": (".io", "build_image_catalog"),
    "build_morphology_fold_change_table": (".analysis.tables", "build_morphology_fold_change_table"),
    "build_morphology_summary_table": (".analysis.tables", "build_morphology_summary_table"),
    "build_morphology_table": (".analysis.tables", "build_morphology_table"),
    "build_stat_summary_table": (".analysis.tables", "build_stat_summary_table"),
    "clear_image_catalog_cache": (".io", "clear_image_catalog_cache"),
    "extract_condition_comparison": (".analysis.pipelines", "extract_condition_comparison"),
    "extract_single_cell_fluorescence": (".analysis.pipelines", "extract_single_cell_fluorescence"),
    "extract_single_cell_relative_fluorescence": (
        ".analysis.pipelines",
        "extract_single_cell_relative_fluorescence",
    ),
    "find_image_path": (".analysis.extraction", "find_image_path"),
    "load_grayscale_tif": (".analysis.extraction", "load_grayscale_tif"),
    "load_stardist_model": (".analysis.extraction", "load_stardist_model"),
    "mann_whitney_u_test": (".stats", "mann_whitney_u_test"),
    "plot_area_intensity_scatter": (".plotting.morphology", "plot_area_intensity_scatter"),
    "plot_morphology_heatmap": (".plotting.morphology", "plot_morphology_heatmap"),
    "plot_morphology_scatter": (".plotting.morphology", "plot_morphology_scatter"),
    "display_stat_summary_tables": (".plotting.fluorescence_summary", "display_stat_summary_tables"),
    "plot_summary_heatmap": (".plotting.fluorescence_summary", "plot_summary_heatmap"),
    "plot_violins_with_stats": (".plotting.fluorescence_summary", "plot_violins_with_stats"),
    "plot_condition_brightness_debug": (
        ".plotting.fluorescence_debug",
        "plot_condition_brightness_debug",
    ),
    "plot_condition_comparison": (".plotting.fluorescence_debug", "plot_condition_comparison"),
    "plot_condition_histograms": (".plotting.fluorescence_debug", "plot_condition_histograms"),
    "plot_condition_relative_background_debug": (
        ".plotting.fluorescence_debug",
        "plot_condition_relative_background_debug",
    ),
    "sort_microscopy_images": (".io", "sort_microscopy_images"),
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
