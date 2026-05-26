"""
Rekarisk QRA — Ignition Probability Models.

Ignition probability estimation for flammable releases. Models
implement Cox/Lees/Amey, TNO Purple Book (CPR 18E), and HSE UK
methodologies.

An ignition event requires:
  - Release of a flammable substance within its flammability range
  - Presence of an ignition source in the vapour cloud
  - Sufficient energy to initiate combustion

References:
  - Cox, Lees & Ang (1990) — Classification of Hazardous Locations
  - TNO Purple Book CPR 18E — Guidelines for QRA
  - HSE UK — Ignition Probability Review & Model Development
  - API RP 581 — Risk-Based Inspection
  - CCPS/AIChE — Guidelines for Determining the Probability of Ignition
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class IgnitionModel(str, Enum):
    """Ignition probability model reference."""
    COX_LEES_AMEY = "cox_lees_amey"
    TNO_PURPLE_BOOK = "tno_purple_book"
    HSE_UK = "hse_uk"
    API_581 = "api_581"
    CUSTOM = "custom"


class SubstanceCategory(str, Enum):
    """Substance categories for ignition probability."""
    HYDROGEN = "hydrogen"
    C1_C4 = "c1_c4"          # Methane to butane, light gases
    C5_PLUS = "c5_plus"      # C5+ liquid hydrocarbons
    H2S = "h2s"               # Hydrogen sulphide
    AROMATICS = "aromatics"   # Benzene, toluene, etc.
    ALCOHOLS = "alcohols"     # Methanol, ethanol, etc.
    OTHER = "other"


class LocationType(str, Enum):
    """Installation/location type classification."""
    ONSHORE_PROCESS = "onshore_process"
    OFFSHORE_PROCESS = "offshore_process"
    STORAGE_TANK_FARM = "storage_tank_farm"
    LOADING_UNLOADING = "loading_unloading"
    PIPELINE = "pipeline"
    REMOTE = "remote"
    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"


class CongestionLevel(str, Enum):
    """Vapour cloud congestion level for explosion probability."""
    LOW = "low"           # Open field, few obstacles
    MEDIUM = "medium"     # Process area, some pipe racks
    HIGH = "high"         # Dense process unit, many obstacles


class ConfinementLevel(str, Enum):
    """Vapour cloud confinement level."""
    NONE = "none"          # No confinement, open area
    PARTIAL_2D = "partial_2d"   # Ground-level partial confinement
    PARTIAL_3D = "partial_3d"   # 3D partial confinement
    FULL = "full"          # Fully enclosed module


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IgnitionData:
    """Default ignition probability data for a substance."""
    substance: str
    category: SubstanceCategory
    p_immediate: float          # Probability of immediate ignition
    p_delayed: float            # Probability of delayed ignition
    p_explosion_base: float     # Base explosion probability (medium congestion)
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# Default substance ignition probabilities
# ──────────────────────────────────────────────────────────────────────

# Sources: TNO Purple Book CPR 18E Table 4.1,
#          Cox/Lees/Amey (1990),
#          HSE UK ignition probability review

DEFAULT_SUBSTANCE_DATA: dict[str, IgnitionData] = {
    "hydrogen": IgnitionData(
        "Hydrogen (H₂)", SubstanceCategory.HYDROGEN,
        p_immediate=0.4, p_delayed=0.2, p_explosion_base=0.3,
        notes="Very wide flammability range (4-75%), low ignition energy (0.02 mJ)",
    ),
    "methane": IgnitionData(
        "Methane (CH₄)", SubstanceCategory.C1_C4,
        p_immediate=0.1, p_delayed=0.1, p_explosion_base=0.2,
        notes="Narrower flammability range (5-15%), higher auto-ignition temp",
    ),
    "ethane": IgnitionData(
        "Ethane (C₂H₆)", SubstanceCategory.C1_C4,
        p_immediate=0.1, p_delayed=0.1, p_explosion_base=0.2,
        notes="Similar to methane",
    ),
    "propane": IgnitionData(
        "Propane (C₃H₈)", SubstanceCategory.C1_C4,
        p_immediate=0.1, p_delayed=0.1, p_explosion_base=0.25,
        notes="Heavier than air at release, higher reactivity",
    ),
    "butane": IgnitionData(
        "Butane (C₄H₁₀)", SubstanceCategory.C1_C4,
        p_immediate=0.1, p_delayed=0.1, p_explosion_base=0.2,
        notes="Similar to propane",
    ),
    "lpg": IgnitionData(
        "LPG (C₃/C₄ mix)", SubstanceCategory.C1_C4,
        p_immediate=0.1, p_delayed=0.15, p_explosion_base=0.25,
        notes="Liquefied petroleum gas; heavier-than-air vapour",
    ),
    "lng": IgnitionData(
        "LNG (Liquefied Natural Gas)", SubstanceCategory.C1_C4,
        p_immediate=0.05, p_delayed=0.1, p_explosion_base=0.15,
        notes="Very cold release, initial dense gas dispersion",
    ),
    "gasoline": IgnitionData(
        "Gasoline / Petrol", SubstanceCategory.C5_PLUS,
        p_immediate=0.05, p_delayed=0.05, p_explosion_base=0.15,
        notes="Flammable liquid, vapour heavier than air",
    ),
    "diesel": IgnitionData(
        "Diesel", SubstanceCategory.C5_PLUS,
        p_immediate=0.01, p_delayed=0.01, p_explosion_base=0.05,
        notes="High flash point, low vapour pressure; low ignition risk at ambient",
    ),
    "kerosene": IgnitionData(
        "Kerosene / Jet A-1", SubstanceCategory.C5_PLUS,
        p_immediate=0.02, p_delayed=0.03, p_explosion_base=0.1,
        notes="Medium flash point",
    ),
    "naphtha": IgnitionData(
        "Naphtha", SubstanceCategory.C5_PLUS,
        p_immediate=0.05, p_delayed=0.05, p_explosion_base=0.2,
        notes="Volatile, wide boiling range",
    ),
    "crude_oil": IgnitionData(
        "Crude Oil", SubstanceCategory.C5_PLUS,
        p_immediate=0.03, p_delayed=0.03, p_explosion_base=0.1,
        notes="Variable composition, lighter ends increase ignition risk",
    ),
    "h2s": IgnitionData(
        "Hydrogen Sulphide (H₂S)", SubstanceCategory.H2S,
        p_immediate=0.05, p_delayed=0.05, p_explosion_base=0.05,
        notes="Toxic, flammable (4-44%), low ignition energy",
    ),
    "benzene": IgnitionData(
        "Benzene (C₆H₆)", SubstanceCategory.AROMATICS,
        p_immediate=0.05, p_delayed=0.05, p_explosion_base=0.15,
        notes="Carcinogenic aromatic hydrocarbon",
    ),
    "toluene": IgnitionData(
        "Toluene (C₇H₈)", SubstanceCategory.AROMATICS,
        p_immediate=0.04, p_delayed=0.05, p_explosion_base=0.15,
        notes="Aromatic solvent",
    ),
    "methanol": IgnitionData(
        "Methanol (CH₃OH)", SubstanceCategory.ALCOHOLS,
        p_immediate=0.08, p_delayed=0.08, p_explosion_base=0.15,
        notes="Low flame luminosity, wide flammability range",
    ),
    "ethanol": IgnitionData(
        "Ethanol (C₂H₅OH)", SubstanceCategory.ALCOHOLS,
        p_immediate=0.06, p_delayed=0.06, p_explosion_base=0.15,
        notes="Similar to methanol with higher flash point",
    ),
    "ethylene": IgnitionData(
        "Ethylene (C₂H₄)", SubstanceCategory.C1_C4,
        p_immediate=0.15, p_delayed=0.15, p_explosion_base=0.3,
        notes="Very wide flammability range (2.7-36%), high reactivity",
    ),
    "propylene": IgnitionData(
        "Propylene (C₃H₆)", SubstanceCategory.C1_C4,
        p_immediate=0.12, p_delayed=0.12, p_explosion_base=0.25,
        notes="Similar to ethylene but higher molecular weight",
    ),
    "ammonia": IgnitionData(
        "Ammonia (NH₃)", SubstanceCategory.OTHER,
        p_immediate=0.01, p_delayed=0.01, p_explosion_base=0.02,
        notes="Narrow flammability range (15-28%), high ignition energy",
    ),
}

# Generic fallback by category
CATEGORY_DEFAULTS: dict[SubstanceCategory, IgnitionData] = {
    SubstanceCategory.HYDROGEN: IgnitionData(
        "Generic H₂-like", SubstanceCategory.HYDROGEN,
        0.4, 0.2, 0.3,
    ),
    SubstanceCategory.C1_C4: IgnitionData(
        "Generic C1-C4", SubstanceCategory.C1_C4,
        0.1, 0.1, 0.2,
    ),
    SubstanceCategory.C5_PLUS: IgnitionData(
        "Generic C5+", SubstanceCategory.C5_PLUS,
        0.05, 0.05, 0.15,
    ),
    SubstanceCategory.H2S: IgnitionData(
        "Generic H₂S-like", SubstanceCategory.H2S,
        0.05, 0.05, 0.05,
    ),
    SubstanceCategory.AROMATICS: IgnitionData(
        "Generic Aromatic", SubstanceCategory.AROMATICS,
        0.05, 0.05, 0.15,
    ),
    SubstanceCategory.ALCOHOLS: IgnitionData(
        "Generic Alcohol", SubstanceCategory.ALCOHOLS,
        0.06, 0.06, 0.15,
    ),
    SubstanceCategory.OTHER: IgnitionData(
        "Generic Other", SubstanceCategory.OTHER,
        0.03, 0.03, 0.1,
    ),
}

# ──────────────────────────────────────────────────────────────────────
# TNO Purple Book — Ignition probability by installation type
# ──────────────────────────────────────────────────────────────────────

# TNO CPR 18E Table 4.1: Probability of direct ignition for
# stationary installations in the open air.
# Values depend on installation type and substance phase.

TNO_DIRECT_IGNITION: dict[str, dict[str, float]] = {
    # installation_type → {phase: probability}
    "gas_processing": {
        "gas": 0.5,
        "two_phase": 0.4,
        "liquid": 0.1,
    },
    "refinery": {
        "gas": 0.4,
        "two_phase": 0.3,
        "liquid": 0.1,
    },
    "chemical_plant": {
        "gas": 0.4,
        "two_phase": 0.3,
        "liquid": 0.1,
    },
    "storage": {
        "gas": 0.3,
        "two_phase": 0.2,
        "liquid": 0.05,
    },
    "loading_unloading": {
        "gas": 0.4,
        "two_phase": 0.3,
        "liquid": 0.1,
    },
    "pipeline": {
        "gas": 0.1,
        "two_phase": 0.1,
        "liquid": 0.02,
    },
    "compressor_station": {
        "gas": 0.5,
        "two_phase": 0.4,
        "liquid": 0.1,
    },
}

# TNO delayed ignition probability modifiers
TNO_DELAYED_IGNITION_BASE: float = 0.3  # Base probability for gas releases
TNO_DELAYED_IGNITION_MULTIPLIER: dict[str, float] = {
    "open_field": 0.5,
    "process_area": 1.0,
    "congested_area": 1.5,
    "built_up_area": 2.0,
}


# ──────────────────────────────────────────────────────────────────────
# HSE UK — Release rate categories for ignition probability
# ──────────────────────────────────────────────────────────────────────

# HSE ignition probability based on release rate (kg/s)
# From: HSE UK Ignition Probability Review
HSE_RELEASE_RATE_THRESHOLDS: list[tuple[float, float, float, float]] = [
    # (rate_min_kg_s, rate_max_kg_s, p_immediate, p_delayed)
    # For flammable gas releases
    (0.0, 0.1, 0.01, 0.01),
    (0.1, 1.0, 0.03, 0.05),
    (1.0, 10.0, 0.08, 0.1),
    (10.0, 50.0, 0.2, 0.15),
    (50.0, 100.0, 0.3, 0.2),
    (100.0, float("inf"), 0.5, 0.3),
]


# ──────────────────────────────────────────────────────────────────────
# Location adjustment factors
# ──────────────────────────────────────────────────────────────────────

LOCATION_IGNITION_FACTORS: dict[str, float] = {
    "onshore_process": 1.0,
    "offshore_process": 0.8,
    "storage_tank_farm": 0.7,
    "loading_unloading": 1.2,
    "pipeline": 0.5,
    "remote": 0.3,
    "residential": 0.1,
    "industrial": 1.1,
}


# ──────────────────────────────────────────────────────────────────────
# Congestion/confinement explosion probability factors
# ──────────────────────────────────────────────────────────────────────

CONGESTION_FACTORS: dict[str, float] = {
    "low": 0.3,
    "medium": 1.0,
    "high": 2.0,
}

CONFINEMENT_FACTORS: dict[str, float] = {
    "none": 0.2,
    "partial_2d": 0.5,
    "partial_3d": 1.0,
    "full": 2.0,
}


# ──────────────────────────────────────────────────────────────────────
# Module functions
# ──────────────────────────────────────────────────────────────────────

def default_ignition_data(substance: str) -> IgnitionData:
    """Get default ignition probability data for a substance.

    Parameters
    ----------
    substance : str
        Substance name (case-insensitive, e.g., "hydrogen", "methane", "propane").

    Returns
    -------
    IgnitionData
        Default ignition probabilities for the substance.
        Falls back to category defaults if specific substance not found.
    """
    key = substance.lower().replace(" ", "_").replace("-", "_")

    # Exact match
    if key in DEFAULT_SUBSTANCE_DATA:
        return DEFAULT_SUBSTANCE_DATA[key]

    # Try partial match
    for name, data in DEFAULT_SUBSTANCE_DATA.items():
        if key in name or name in key:
            return data

    # Try category match
    for cat in SubstanceCategory:
        if cat.value in key:
            return CATEGORY_DEFAULTS.get(cat, CATEGORY_DEFAULTS[SubstanceCategory.OTHER])

    return CATEGORY_DEFAULTS[SubstanceCategory.OTHER]


def immediate_ignition_probability(
    substance: str,
    release_rate: float = 1.0,
    location_type: Union[str, LocationType] = "onshore_process",
    model: Union[str, IgnitionModel] = IgnitionModel.COX_LEES_AMEY,
    phase: str = "gas",
    installation_type: str = "chemical_plant",
) -> float:
    """Calculate immediate (direct) ignition probability.

    Immediate ignition occurs at the point of release due to:
    - Static electricity discharge
    - Hot surfaces on the leaking equipment
    - Friction sparks from the release itself
    - Electrical equipment near the release point

    Parameters
    ----------
    substance : str
        Substance name (e.g., "hydrogen", "methane", "propane").
    release_rate : float
        Mass release rate in kg/s. Used by HSE model.
    location_type : str or LocationType
        Facility location type.
    model : str or IgnitionModel
        Ignition probability model to use.
    phase : str
        Release phase: "gas", "liquid", or "two_phase".
    installation_type : str
        Type of installation (for TNO model).

    Returns
    -------
    float
        Immediate ignition probability (0 to 1).

    Examples
    --------
    >>> immediate_ignition_probability("hydrogen", release_rate=10.0)
    0.4
    >>> immediate_ignition_probability("methane", release_rate=10.0)
    0.1
    >>> # H₂ should have higher P_imm than methane
    >>> p_h2 = immediate_ignition_probability("hydrogen")
    >>> p_ch4 = immediate_ignition_probability("methane")
    >>> p_h2 > p_ch4
    True
    """
    ign_data = default_ignition_data(substance)
    loc_str = location_type.value if isinstance(location_type, LocationType) else str(location_type)
    model_enum = IgnitionModel(model) if isinstance(model, str) else model

    # Base probability
    prob = ign_data.p_immediate

    if model_enum == IgnitionModel.TNO_PURPLE_BOOK:
        # Use TNO direct ignition tables
        inst_data = TNO_DIRECT_IGNITION.get(installation_type, TNO_DIRECT_IGNITION["chemical_plant"])
        prob = inst_data.get(phase, inst_data.get("gas", 0.3))

    elif model_enum == IgnitionModel.HSE_UK:
        # HSE release rate model
        prob = _hse_release_rate_probability(release_rate, immediate=True)

    elif model_enum == IgnitionModel.API_581:
        # API 581 simplified model
        prob = _api_581_ignition_probability(substance, release_rate, immediate=True)

    # Apply location factor
    loc_factor = LOCATION_IGNITION_FACTORS.get(loc_str, 1.0)
    prob *= loc_factor

    return min(1.0, max(0.0, prob))


def delayed_ignition_probability(
    substance: str,
    release_duration: float = 300.0,
    congestion: Union[str, CongestionLevel] = "medium",
    model: Union[str, IgnitionModel] = IgnitionModel.COX_LEES_AMEY,
    location_type: Union[str, LocationType] = "onshore_process",
    installation_type: str = "chemical_plant",
) -> float:
    """Calculate delayed ignition probability.

    Delayed ignition occurs after a vapour cloud has formed and
    drifted to an ignition source:
    - Electrical equipment in the surrounding area
    - Vehicles, furnaces, flares
    - Hot work in the vicinity
    - Lightning (very low probability)

    Parameters
    ----------
    substance : str
        Substance name.
    release_duration : float
        Duration of release in seconds. Longer releases have higher
        probability of finding an ignition source.
    congestion : str or CongestionLevel
        Level of congestion in the vapour cloud area.
    model : str or IgnitionModel
        Probability model.
    location_type : str or LocationType
        Facility location type.
    installation_type : str
        For TNO model.

    Returns
    -------
    float
        Delayed ignition probability (0 to 1).

    Examples
    --------
    >>> p = delayed_ignition_probability("propane", release_duration=600)
    >>> 0.0 < p < 1.0
    True
    """
    ign_data = default_ignition_data(substance)
    loc_str = location_type.value if isinstance(location_type, LocationType) else str(location_type)
    cong_str = congestion.value if isinstance(congestion, CongestionLevel) else str(congestion)
    model_enum = IgnitionModel(model) if isinstance(model, str) else model

    prob = ign_data.p_delayed

    if model_enum == IgnitionModel.TNO_PURPLE_BOOK:
        cong_factor = CONGESTION_FACTORS.get(cong_str, 1.0)
        inst_type_key = installation_type.replace(" ", "_").lower()
        multiplier = TNO_DELAYED_IGNITION_MULTIPLIER.get(
            inst_type_key,
            TNO_DELAYED_IGNITION_MULTIPLIER.get("process_area", 1.0),
        )
        prob = TNO_DELAYED_IGNITION_BASE * cong_factor * multiplier

    elif model_enum == IgnitionModel.HSE_UK:
        prob = _hse_release_rate_probability(
            release_duration / 10.0, immediate=False
        )  # Use duration as proxy

    # Duration effect: logarithmic relationship
    # P_delayed increases with exposure time
    if release_duration > 60:
        duration_factor = min(3.0, 1.0 + 0.3 * math.log10(release_duration / 60))
        prob *= duration_factor

    # Congestion effect
    cong_factor = CONGESTION_FACTORS.get(cong_str, 1.0)
    prob *= cong_factor

    # Location factor (reduced effect compared to immediate)
    loc_factor = LOCATION_IGNITION_FACTORS.get(loc_str, 1.0)
    prob *= (0.5 + 0.5 * loc_factor)

    return min(1.0, max(0.0, prob))


def explosion_probability(
    substance: str,
    congestion: Union[str, CongestionLevel] = "medium",
    confinement: Union[str, ConfinementLevel] = "partial_3d",
) -> float:
    """Calculate probability of explosion given delayed ignition.

    Not all delayed ignitions result in explosions. Whether a vapour
    cloud explodes depends on:
    - Flame speed acceleration to deflagration or detonation
    - Degree of congestion (obstacle density, repeated pipe arrays)
    - Degree of confinement (parallel walls, ceiling)
    - Reactivity of the substance (laminar burning velocity)
    - Cloud size at ignition time

    Parameters
    ----------
    substance : str
        Substance name.
    congestion : str or CongestionLevel
        Level of congestion.
    confinement : str or ConfinementLevel
        Level of confinement.

    Returns
    -------
    float
        Explosion probability (0 to 1), conditioned on ignition.

    Examples
    --------
    >>> p_exp = explosion_probability("hydrogen", "high", "full")
    >>> p_exp > 0.3  # H₂ + high congestion + full confinement = high explosion prob
    True
    >>> p_exp_low = explosion_probability("ammonia", "low", "none")
    >>> p_exp_low < 0.1
    True
    """
    ign_data = default_ignition_data(substance)
    cong_str = congestion.value if isinstance(congestion, CongestionLevel) else str(congestion)
    conf_str = confinement.value if isinstance(confinement, ConfinementLevel) else str(confinement)

    base_prob = ign_data.p_explosion_base

    # Congestion factor
    cong_factor = CONGESTION_FACTORS.get(cong_str, 1.0)

    # Confinement factor
    conf_factor = CONFINEMENT_FACTORS.get(conf_str, 1.0)

    # Combined probability using multiplicative model capped at 1.0
    prob = base_prob * cong_factor * conf_factor

    # Additional reactivity adjustment
    if ign_data.category == SubstanceCategory.HYDROGEN:
        prob *= 1.3  # H₂ is very reactive
    elif ign_data.category == SubstanceCategory.C1_C4:
        if substance.lower() in ("ethylene", "propylene"):
            prob *= 1.2  # More reactive than methane

    return min(1.0, max(0.0, prob))


def combined_ignition_probability(
    substance: str,
    release_rate: float = 1.0,
    release_duration: float = 300.0,
    location_type: Union[str, LocationType] = "onshore_process",
    congestion: Union[str, CongestionLevel] = "medium",
    confinement: Union[str, ConfinementLevel] = "partial_3d",
    model: Union[str, IgnitionModel] = IgnitionModel.COX_LEES_AMEY,
    phase: str = "gas",
    installation_type: str = "chemical_plant",
) -> dict[str, float]:
    """Calculate combined ignition and explosion probabilities.

    Returns a dictionary with all intermediate and combined
    probabilities for use in event tree analysis.

    Parameters
    ----------
    substance : str
        Substance name.
    release_rate : float
        Release rate in kg/s.
    release_duration : float
        Release duration in seconds.
    location_type : str or LocationType
        Facility location.
    congestion : str or CongestionLevel
        Congestion level.
    confinement : str or ConfinementLevel
        Confinement level.
    model : str or IgnitionModel
        Probability model.
    phase : str
        Release phase.
    installation_type : str
        Installation type.

    Returns
    -------
    dict
        Keys: p_immediate, p_delayed, p_explosion_given_delayed,
              p_explosion, p_no_ignition, p_any_ignition
    """
    p_imm = immediate_ignition_probability(
        substance, release_rate, location_type, model, phase, installation_type,
    )
    p_del = delayed_ignition_probability(
        substance, release_duration, congestion, model, location_type, installation_type,
    )

    # Explosion probability given delayed ignition occurred
    p_exp_given_del = explosion_probability(substance, congestion, confinement)

    # Combined probabilities
    p_no_ignition = (1.0 - p_imm) * (1.0 - p_del)
    p_any_ignition = 1.0 - p_no_ignition
    p_explosion = (1.0 - p_imm) * p_del * p_exp_given_del

    return {
        "p_immediate": p_imm,
        "p_delayed": p_del,
        "p_explosion_given_delayed": p_exp_given_del,
        "p_explosion": p_explosion,
        "p_no_ignition": p_no_ignition,
        "p_any_ignition": p_any_ignition,
    }


# ──────────────────────────────────────────────────────────────────────
# Internal helper functions
# ──────────────────────────────────────────────────────────────────────

def _hse_release_rate_probability(
    release_rate: float,
    immediate: bool = True,
) -> float:
    """HSE UK ignition probability based on release rate categories.

    Parameters
    ----------
    release_rate : float
        Release rate in kg/s.
    immediate : bool
        If True, return immediate ignition probability;
        if False, return delayed ignition probability.

    Returns
    -------
    float
        Ignition probability (0 to 1).
    """
    idx = 2 if immediate else 3  # Column index in thresholds table
    for rate_min, rate_max, p_imm, p_del in HSE_RELEASE_RATE_THRESHOLDS:
        if rate_min <= release_rate < rate_max or rate_max == float("inf"):
            return p_imm if immediate else p_del
    return 0.01


def _api_581_ignition_probability(
    substance: str,
    release_rate: float = 1.0,
    immediate: bool = True,
) -> float:
    """API RP 581 simplified ignition probability.

    Based on substance auto-ignition temperature (AIT) relative
    to operating temperature and release rate.

    Parameters
    ----------
    substance : str
        Substance name.
    release_rate : float
        Release rate in kg/s.
    immediate : bool
        If True, immediate; if False, delayed.

    Returns
    -------
    float
        Ignition probability (0 to 1).
    """
    # Simplified API approach
    if release_rate < 0.1:
        base = 0.01
    elif release_rate < 1.0:
        base = 0.05
    elif release_rate < 10.0:
        base = 0.1
    elif release_rate < 50.0:
        base = 0.2
    else:
        base = 0.3

    # Hydrogen and light gases have higher probability
    if substance.lower() in ("hydrogen", "h2"):
        base *= 2.0
    elif substance.lower() in ("ethylene", "propylene"):
        base *= 1.5

    if not immediate:
        base *= 0.7  # Delayed is typically lower than immediate

    return min(1.0, max(0.0, base))
