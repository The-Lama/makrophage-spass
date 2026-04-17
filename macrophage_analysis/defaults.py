from __future__ import annotations

from pathlib import Path

from .models import AntibodySpec

DEFAULT_DATA_ROOT = Path("raw_data")
DEFAULT_BASELINE_CONDITION = "M0"
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
DEFAULT_DESCRIPTIVE_CONDITION_DISPLAY_LABELS = {
    "M0": "M0 (baseline)",
    "B68KCP2": "Keratinocytes",
    "B75palmFP1": "Fibroblasts",
    "M130227Ni": "Mesenchymal NRAS melanoma cells",
    "M130429Np": "Proliferative NRAS melanoma cells",
    "T12CMNepiP4": "CMN Melanocytes epidermis",
    "T8CMNdermP6": "CMN Melanocytes dermis",
}
DEFAULT_ANTIBODY_SPECS: dict[str, AntibodySpec] = {
    "CD40": AntibodySpec("CD40_A488_10x_RGB_Qi2 10x", "FITC"),
    "CD86": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "FITC"),
    "iNOS": AntibodySpec("CD86_A488_iNOS_A568_10x_RGB_Qi2 10x", "TRITC"),
    "CD200R": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "FITC"),
    "CD206": AntibodySpec("CD200R_A488_CD206_A568_10x_RGB_Qi2 10x", "TRITC"),
}
DEFAULT_ANTIBODY_ORDER = tuple(DEFAULT_ANTIBODY_SPECS.keys())

__all__ = [
    "DEFAULT_ANTIBODY_ORDER",
    "DEFAULT_ANTIBODY_SPECS",
    "DEFAULT_BASELINE_CONDITION",
    "DEFAULT_COMPARISON_CHANNEL",
    "DEFAULT_COMPARISON_MARKER_PREFIX",
    "DEFAULT_CONDITIONS",
    "DEFAULT_CONDITION_DISPLAY_LABELS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_DESCRIPTIVE_CONDITION_DISPLAY_LABELS",
    "DEFAULT_DONOR_COLORS",
    "DEFAULT_DONOR_DISPLAY_LABELS",
    "DEFAULT_DONORS",
    "DEFAULT_SEABORN_THEME",
    "DEFAULT_STARDIST_N_TILES",
]
