"""
Rekarisk Meteorology — Pasquill-Gifford Stability Classification.

Provides Pasquill-Gifford stability class determination from meteorological
conditions and dispersion coefficient (sigma-y, sigma-z) calculations.

Reference:
    Briggs, G.A. (1973). Diffusion estimation for small emissions.
    ATDL Contribution File No. 79. NOAA, Oak Ridge, TN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StabilityClass = Literal["A", "B", "C", "D", "E", "F"]
TerrainType = Literal["rural", "urban"]
RadiationType = Literal["strong", "moderate", "slight"]
CloudCoverType = Literal["overcast", "clear"]

# Combined stability classes from the P-G table
CombinedClass = Literal["A", "A-B", "B", "B-C", "C", "C-D", "D", "E", "F"]

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _SigmaCoeff(TypedDict):
    a: float
    b: float


class _SigmaZCoeff(TypedDict):
    x_min: float
    x_max: float
    a: float
    b: float
    c: float


# ---------------------------------------------------------------------------
# Load coefficient data
# ---------------------------------------------------------------------------

_COEFF_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "pasquill_coefficients.json"


def _load_coefficients() -> dict:
    """Load Pasquill-Gifford coefficients from JSON data file."""
    # Resolve relative to this file:
    #   src/rekarisk/meteorology/stability.py → 4 levels up → project root / data/
    module_dir = Path(__file__).resolve().parent  # src/rekarisk/meteorology
    candidates = [
        module_dir.parent.parent.parent / "data" / "pasquill_coefficients.json",  # 4 levels up = project root
        module_dir.parent.parent / "data" / "pasquill_coefficients.json",        # src/rekarisk/data/
        module_dir.parent / "data" / "pasquill_coefficients.json",                # src/rekarisk/meteorology/data/
        Path.cwd() / "data" / "pasquill_coefficients.json",                       # cwd fallback
    ]
    for coeff_path in candidates:
        if coeff_path.exists():
            with open(coeff_path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Cannot find pasquill_coefficients.json. Searched: "
        + ", ".join(str(p) for p in candidates)
    )


# Load at module level
_COEFF = _load_coefficients()

# ---------------------------------------------------------------------------
# PG stability classification table (built from JSON)
# ---------------------------------------------------------------------------

# Daytime classification: wind speed band → {radiation_level → stability}
# Handles combined classes like A-B, B-C, C-D
_DAYTIME_TABLE: Dict[str, Dict[str, str]] = {
    "wind_lt_2": {
        "strong": "A",
        "moderate": "A-B",
        "slight": "B",
    },
    "wind_2_to_3": {
        "strong": "A-B",
        "moderate": "B",
        "slight": "C",
    },
    "wind_3_to_5": {
        "strong": "B",
        "moderate": "B-C",
        "slight": "C",
    },
    "wind_5_to_6": {
        "strong": "C",
        "moderate": "C-D",
        "slight": "D",
    },
    "wind_gt_6": {
        "strong": "C",
        "moderate": "D",
        "slight": "D",
    },
}

# Nighttime classification: wind speed band → {cloud_cover → stability}
_NIGHTTIME_TABLE: Dict[str, Dict[str, str]] = {
    "wind_lt_2": {
        "overcast": "F",
        "clear": "F",
    },
    "wind_2_to_3": {
        "overcast": "E",
        "clear": "F",
    },
    "wind_3_to_5": {
        "overcast": "D",
        "clear": "E",
    },
    "wind_5_to_6": {
        "overcast": "D",
        "clear": "D",
    },
    "wind_gt_6": {
        "overcast": "D",
        "clear": "D",
    },
}

# ---------------------------------------------------------------------------
# Helper: wind speed band determination
# ---------------------------------------------------------------------------


def _wind_band(wind_speed_ms: float) -> str:
    """Determine wind speed band for PG classification table.

    Args:
        wind_speed_ms: Wind speed at 10 m height in m/s.

    Returns:
        Wind band identifier string.
    """
    if wind_speed_ms < 2.0:
        return "wind_lt_2"
    elif wind_speed_ms < 3.0:
        return "wind_2_to_3"
    elif wind_speed_ms < 5.0:
        return "wind_3_to_5"
    elif wind_speed_ms < 6.0:
        return "wind_5_to_6"
    else:
        return "wind_gt_6"


def _resolve_combined(stability_str: str, wind_speed_ms: float) -> StabilityClass:
    """Resolve combined PG classes (A-B, B-C, C-D) to single class.

    Uses wind speed within the band to bias toward one class.
    Lower wind in band → more unstable (earlier letter).
    Higher wind in band → more neutral (later letter).

    Args:
        stability_str: Combined stability string like "A-B", "B-C", "C-D".
        wind_speed_ms: Wind speed in m/s for precise resolution.

    Returns:
        Single stability class A-F.
    """
    if "-" not in stability_str:
        return stability_str  # type: ignore[return-value]

    classes = stability_str.split("-")
    # Determine the band midpoints to bias
    band = _wind_band(wind_speed_ms)

    band_ranges = {
        "wind_lt_2": (0.0, 2.0),
        "wind_2_to_3": (2.0, 3.0),
        "wind_3_to_5": (3.0, 5.0),
        "wind_5_to_6": (5.0, 6.0),
        "wind_gt_6": (6.0, 25.0),
    }
    lo, hi = band_ranges[band]
    fraction = (wind_speed_ms - lo) / (hi - lo) if hi > lo else 0.5

    # If wind is in lower part of band, pick more unstable (earlier letter)
    if fraction < 0.5:
        return classes[0]  # type: ignore[return-value]
    else:
        return classes[1]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API: Stability Classification
# ---------------------------------------------------------------------------


def classify_stability(
    wind_speed_ms: float,
    solar_radiation: Optional[float] = None,
    cloud_cover_oktas: Optional[float] = None,
    is_daytime: bool = True,
) -> StabilityClass:
    """Classify atmospheric stability using Pasquill-Gifford method.

    Args:
        wind_speed_ms: Wind speed at 10 m height [m/s].
        solar_radiation: Solar radiation in W/m². If provided, used to
            determine radiation level for daytime classification.
        cloud_cover_oktas: Cloud cover in oktas (0-8). Used for nighttime
            classification when solar_radiation not provided.
        is_daytime: Whether it is daytime. Used for radiation classification.

    Returns:
        Pasquill-Gifford stability class (A through F).

    Raises:
        ValueError: If wind_speed_ms is negative.

    Examples:
        >>> classify_stability(1.0, solar_radiation=800, is_daytime=True)
        'A'
        >>> classify_stability(7.0, cloud_cover_oktas=2, is_daytime=False)
        'D'
        >>> classify_stability(4.0, solar_radiation=300, is_daytime=True)
        'C'
    """
    if wind_speed_ms < 0:
        raise ValueError(f"Wind speed must be non-negative, got {wind_speed_ms}")

    # Apply minimum wind speed for classification
    wind = max(wind_speed_ms, 0.5)
    band = _wind_band(wind)

    if is_daytime:
        # Determine radiation level from solar_radiation if provided
        if solar_radiation is not None:
            if solar_radiation >= 600:
                radiation = "strong"
            elif solar_radiation >= 300:
                radiation = "moderate"
            else:
                radiation = "slight"
        else:
            # Default: assume moderate if no data
            radiation = "moderate"

        stability_str = _DAYTIME_TABLE[band][radiation]
    else:
        # Nighttime: determine cloud cover
        if cloud_cover_oktas is not None:
            cloud = "overcast" if cloud_cover_oktas > 4 else "clear"
        else:
            # Default: assume clear sky at night (worst case for stability)
            cloud = "clear"

        stability_str = _NIGHTTIME_TABLE[band][cloud]

    return _resolve_combined(stability_str, wind)


def classify_stability_from_radiation(
    wind_speed_ms: float,
    radiation_level: str = "moderate",
) -> StabilityClass:
    """Classify stability using descriptive radiation level (daytime).

    Convenience wrapper when radiation is specified as qualitative category.

    Args:
        wind_speed_ms: Wind speed at 10 m height [m/s].
        radiation_level: 'strong', 'moderate', or 'slight'.

    Returns:
        Pasquill-Gifford stability class.
    """
    if radiation_level not in ("strong", "moderate", "slight"):
        raise ValueError(
            f"radiation_level must be 'strong', 'moderate', or 'slight', "
            f"got '{radiation_level}'"
        )
    wind = max(wind_speed_ms, 0.5)
    band = _wind_band(wind)
    stability_str = _DAYTIME_TABLE[band][radiation_level]
    return _resolve_combined(stability_str, wind)


def classify_stability_from_cloud(
    wind_speed_ms: float,
    cloud_cover_oktas: float = 0,
) -> StabilityClass:
    """Classify stability using cloud cover (nighttime).

    Convenience wrapper for nighttime classification.

    Args:
        wind_speed_ms: Wind speed at 10 m height [m/s].
        cloud_cover_oktas: Cloud cover in oktas (0-8). 0 = clear, 8 = fully overcast.

    Returns:
        Pasquill-Gifford stability class.
    """
    wind = max(wind_speed_ms, 0.5)
    band = _wind_band(wind)
    cloud = "overcast" if cloud_cover_oktas > 4 else "clear"
    stability_str = _NIGHTTIME_TABLE[band][cloud]
    return _resolve_combined(stability_str, wind)


# ---------------------------------------------------------------------------
# Public API: Dispersion Coefficients (Sigma-y, Sigma-z)
# ---------------------------------------------------------------------------


def sigma_y(
    downwind_distance_m: float,
    stability: StabilityClass,
    terrain: TerrainType = "rural",
) -> float:
    """Calculate lateral dispersion coefficient sigma-y [m].

    Uses Briggs power-law formulation: σ_y = a * x^b
    where x is downwind distance in km and σ_y is in km, then converted to m.

    Args:
        downwind_distance_m: Downwind distance from source [m].
        stability: Pasquill-Gifford stability class (A-F).
        terrain: 'rural' or 'urban'.

    Returns:
        Lateral dispersion coefficient sigma_y [m].

    Examples:
        >>> sigma_y(1000.0, 'D', 'rural')
        80.0
        >>> sigma_y(500.0, 'A', 'urban')
        172.18...
    """
    if downwind_distance_m < 0:
        raise ValueError(f"Downwind distance must be non-negative, got {downwind_distance_m}")

    coeffs = _COEFF["sigma_y"][terrain][stability]
    x_km = downwind_distance_m / 1000.0
    sigma_y_km = coeffs["a"] * (x_km ** coeffs["b"])
    return sigma_y_km * 1000.0


def sigma_z(
    downwind_distance_m: float,
    stability: StabilityClass,
    terrain: TerrainType = "rural",
) -> float:
    """Calculate vertical dispersion coefficient sigma-z [m].

    Uses piecewise formulations from Pasquill-Gifford/Briggs model.
    For stability A: σ_z = a*x + c*x² (x in km, σ_z in km).
    For other stabilities: σ_z = a * x^b (x in km, σ_z in km).

    Args:
        downwind_distance_m: Downwind distance from source [m].
        stability: Pasquill-Gifford stability class (A-F).
        terrain: 'rural' or 'urban'.

    Returns:
        Vertical dispersion coefficient sigma_z [m].

    Examples:
        >>> sigma_z(1000.0, 'D', 'rural')
        60.0
        >>> sigma_z(100.0, 'A', 'rural')
        2.0
    """
    if downwind_distance_m < 0:
        raise ValueError(f"Downwind distance must be non-negative, got {downwind_distance_m}")

    pieces = _COEFF["sigma_z"][terrain][stability]
    x_km = downwind_distance_m / 1000.0

    # Find the appropriate piece for this distance
    for piece in pieces:
        if piece["x_min"] <= x_km <= piece["x_max"]:
            a, b, c = piece["a"], piece["b"], piece["c"]
            sigma_z_km = a * (x_km ** b) + c
            return sigma_z_km * 1000.0

    # Fallback: use the last piece (shouldn't normally reach here)
    last = pieces[-1]
    sigma_z_km = last["a"] * (x_km ** last["b"]) + last["c"]
    return sigma_z_km * 1000.0


def sigma_y_corrected(
    downwind_distance_m: float,
    stability: StabilityClass,
    terrain: TerrainType = "rural",
    sampling_time_s: float = 600.0,
    reference_time_s: float = 600.0,
) -> float:
    """Calculate sigma-y with sampling time correction.

    The PG coefficients are based on ~10-minute sampling times.
    For longer averaging periods, sigma-y increases due to plume meander.

    σ_y(t2) / σ_y(t1) = (t2 / t1)^0.2

    Args:
        downwind_distance_m: Downwind distance [m].
        stability: PG stability class.
        terrain: 'rural' or 'urban'.
        sampling_time_s: Desired sampling/averaging time [s].
        reference_time_s: Reference sampling time [s] (default 600 = 10 min).

    Returns:
        Time-corrected sigma_y [m].

    References:
        Turner, D.B. (1994). Workbook of Atmospheric Dispersion Estimates.
    """
    sigma_y_base = sigma_y(downwind_distance_m, stability, terrain)
    if sampling_time_s <= 0 or reference_time_s <= 0:
        return sigma_y_base
    ratio = (sampling_time_s / reference_time_s) ** 0.2
    return sigma_y_base * ratio


# ---------------------------------------------------------------------------
# Public API: Wind Profile Exponent
# ---------------------------------------------------------------------------


def power_law_exponent(stability: StabilityClass) -> float:
    """Get wind profile power-law exponent p for a given stability class.

    Used in: u(z) = u_ref * (z / z_ref)^p

    Args:
        stability: PG stability class (A-F).

    Returns:
        Power-law exponent p.

    Examples:
        >>> power_law_exponent('D')
        0.15
        >>> power_law_exponent('A')
        0.07
    """
    exponents = _COEFF["power_law_exponents"]
    return exponents[stability]


# ---------------------------------------------------------------------------
# Public API: Mixing Height
# ---------------------------------------------------------------------------


def mixing_height(
    stability: StabilityClass,
    is_daytime: bool = True,
) -> float:
    """Get default mixing height for a given stability class and time of day.

    Args:
        stability: PG stability class (A-F).
        is_daytime: True for daytime, False for nighttime.

    Returns:
        Default mixing height [m].

    Examples:
        >>> mixing_height('D', True)
        800
        >>> mixing_height('F', False)
        100
    """
    time_key = "daytime" if is_daytime else "nighttime"
    return _COEFF["mixing_height_defaults"][time_key][stability]


# ---------------------------------------------------------------------------
# Public API: Surface Roughness
# ---------------------------------------------------------------------------


def surface_roughness(terrain_type: str) -> float:
    """Get surface roughness length z0 for a named terrain type.

    Args:
        terrain_type: Terrain name, e.g. 'short_grass', 'urban', 'calm_sea'.

    Returns:
        Surface roughness length z0 [m].

    Raises:
        KeyError: If terrain_type is not recognized.
    """
    descriptions = _COEFF["wind_profile_descriptions"]
    if terrain_type not in descriptions:
        available = list(descriptions.keys())
        raise KeyError(
            f"Unknown terrain type '{terrain_type}'. Available: {available}"
        )
    return descriptions[terrain_type]["z0"]


def list_terrain_types() -> List[str]:
    """Return list of available terrain type names."""
    return list(_COEFF["wind_profile_descriptions"].keys())


def get_stability_description(stability: StabilityClass) -> str:
    """Return human-readable description of a stability class."""
    descriptions = _COEFF["stability_classification_table"]["classes"]
    return descriptions.get(stability, f"Unknown stability class: {stability}")
