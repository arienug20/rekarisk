"""
Rekarisk Meteorology — Core Meteorological Models.

Provides wind profile, atmospheric density, lapse rate, and integrated
weather calculations for dispersion modeling.

Key models:
    - Wind profile: power law and log law
    - Atmospheric density
    - Temperature lapse rate
    - Mixing height estimation
    - Integrated meteorological state
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .stability import (
    StabilityClass,
    TerrainType,
    classify_stability,
    mixing_height as get_mixing_height,
    power_law_exponent,
    sigma_y,
    sigma_z,
    sigma_y_corrected,
    surface_roughness,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# von Kármán constant
VON_KARMAN = 0.41

# Dry adiabatic lapse rate [K/m]
DRY_ADIABATIC_LAPSE_RATE = 0.0098  # 9.8 K/km

# Actual lapse rate by stability class [K/m]
LAPSE_RATE: Dict[StabilityClass, float] = {
    "A": -0.019,   # superadiabatic (strong heating from below)
    "B": -0.015,   # moderately superadiabatic
    "C": -0.012,   # slightly superadiabatic
    "D": -0.0098,  # neutral = dry adiabatic
    "E": -0.005,   # subadiabatic (stable, weak cooling)
    "F": -0.002,   # strongly stable (inversion)
}

# Gas constant for dry air [J/(kg·K)]
R_DRY_AIR = 287.058

# Gas constant for water vapor [J/(kg·K)]
R_WATER_VAPOR = 461.495

# Molar mass of dry air [kg/mol]
M_DRY_AIR = 0.028964

# Molar mass of water vapor [kg/mol]
M_WATER_VAPOR = 0.018015

# Minimum allowable wind speed for dispersion [m/s]
MIN_WIND_SPEED = 0.5

# Reference height for wind measurements [m]
Z_REF_DEFAULT = 10.0


# ---------------------------------------------------------------------------
# Dataclass: MeteorologicalState
# ---------------------------------------------------------------------------


@dataclass
class MeteorologicalState:
    """Complete meteorological state for dispersion calculations.

    Attributes:
        wind_speed_ms: Wind speed at reference height [m/s].
        wind_direction_deg: Wind direction from north [degrees].
        reference_height_m: Reference height for wind measurement [m].
        ambient_temperature_k: Ambient temperature [K].
        ambient_pressure_pa: Ambient pressure [Pa].
        relative_humidity_pct: Relative humidity [%].
        cloud_cover_oktas: Cloud cover [oktas, 0-8].
        solar_radiation_wm2: Solar radiation [W/m²].
        is_daytime: Whether it is daytime.
        surface_roughness_m: Surface roughness length z0 [m].
        stability_class: Pasquill-Gifford stability class (auto or manual).
        mixing_height_m: Mixing height [m] (0 means auto-calculate).
        lapse_rate_kpm: Temperature lapse rate [K/m].
    """

    wind_speed_ms: float = 3.0
    wind_direction_deg: float = 0.0
    reference_height_m: float = Z_REF_DEFAULT
    ambient_temperature_k: float = 298.15
    ambient_pressure_pa: float = 101325.0
    relative_humidity_pct: float = 50.0
    cloud_cover_oktas: float = 4.0
    solar_radiation_wm2: float = 500.0
    is_daytime: bool = True
    surface_roughness_m: float = 0.1  # agricultural
    stability_class: Optional[StabilityClass] = None
    mixing_height_m: float = 0.0  # 0 = auto
    lapse_rate_kpm: Optional[float] = None  # None = auto from stability

    def __post_init__(self) -> None:
        """Auto-classify stability and set defaults if not provided."""
        if self.stability_class is None:
            self.stability_class = classify_stability(
                wind_speed_ms=self.wind_speed_ms,
                solar_radiation=self.solar_radiation_wm2 if self.is_daytime else None,
                cloud_cover_oktas=self.cloud_cover_oktas if not self.is_daytime else None,
                is_daytime=self.is_daytime,
            )

        if self.lapse_rate_kpm is None:
            self.lapse_rate_kpm = get_lapse_rate(self.stability_class)

    @property
    def air_density_kgm3(self) -> float:
        """Calculate moist air density [kg/m³]."""
        return atmospheric_density(
            temperature_k=self.ambient_temperature_k,
            pressure_pa=self.ambient_pressure_pa,
            relative_humidity_pct=self.relative_humidity_pct,
        )

    @property
    def mixing_height(self) -> float:
        """Get mixing height [m], auto-calculated if set to 0."""
        if self.mixing_height_m > 0:
            return self.mixing_height_m
        return get_mixing_height(self.stability_class, self.is_daytime)

    def wind_speed_at_height(self, z_m: float, method: str = "power_law") -> float:
        """Calculate wind speed at height z [m] above ground."""
        if method == "power_law":
            return wind_power_law(
                z_m=z_m,
                u_ref=self.wind_speed_ms,
                z_ref=self.reference_height_m,
                stability=self.stability_class,
            )
        elif method == "log_law":
            return wind_log_law(
                z_m=z_m,
                u_ref=self.wind_speed_ms,
                z_ref=self.reference_height_m,
                z0=self.surface_roughness_m,
            )
        else:
            raise ValueError(f"Unknown wind profile method: {method}")

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_deg": self.wind_direction_deg,
            "reference_height_m": self.reference_height_m,
            "ambient_temperature_k": self.ambient_temperature_k,
            "ambient_pressure_pa": self.ambient_pressure_pa,
            "relative_humidity_pct": self.relative_humidity_pct,
            "cloud_cover_oktas": self.cloud_cover_oktas,
            "solar_radiation_wm2": self.solar_radiation_wm2,
            "is_daytime": self.is_daytime,
            "surface_roughness_m": self.surface_roughness_m,
            "stability_class": self.stability_class,
            "mixing_height_m": self.mixing_height_m,
            "lapse_rate_kpm": self.lapse_rate_kpm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MeteorologicalState":
        """Deserialize from dictionary."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Wind Profile Models
# ---------------------------------------------------------------------------


def wind_power_law(
    z_m: float,
    u_ref: float,
    z_ref: float = Z_REF_DEFAULT,
    stability: StabilityClass = "D",
) -> float:
    """Calculate wind speed at height z using power law profile.

    u(z) = u_ref * (z / z_ref)^p

    Args:
        z_m: Target height above ground [m].
        u_ref: Reference wind speed at z_ref [m/s].
        z_ref: Reference height [m].
        stability: PG stability class (determines exponent p).

    Returns:
        Wind speed at height z [m/s].

    Examples:
        >>> wind_power_law(50.0, 5.0, 10.0, 'D')
        6.365...
        >>> wind_power_law(100.0, 3.0, 10.0, 'A')
        3.525...
    """
    if z_m <= 0:
        return 0.0
    if u_ref < MIN_WIND_SPEED:
        u_ref = MIN_WIND_SPEED

    p = power_law_exponent(stability)
    # Use max to avoid division by zero
    z_ref_safe = max(z_ref, 0.1)
    return u_ref * (z_m / z_ref_safe) ** p


def wind_log_law(
    z_m: float,
    u_ref: float,
    z_ref: float = Z_REF_DEFAULT,
    z0: float = 0.1,
    u_star: Optional[float] = None,
) -> float:
    """Calculate wind speed at height z using log-law profile.

    If u_star (friction velocity) is not provided, it is derived from
    the reference wind speed:

        u* = u_ref * κ / ln(z_ref / z0)

    Then:
        u(z) = (u* / κ) * ln(z / z0)

    Args:
        z_m: Target height above ground [m].
        u_ref: Reference wind speed at z_ref [m/s].
        z_ref: Reference height [m].
        z0: Surface roughness length [m].
        u_star: Friction velocity [m/s] (optional, derived if not given).

    Returns:
        Wind speed at height z [m/s].

    Examples:
        >>> wind_log_law(50.0, 5.0, 10.0, 0.1)
        6.173...
        >>> wind_log_law(100.0, 3.0, 10.0, 0.01)
        3.606...
    """
    if z_m <= 0:
        return 0.0
    if u_ref < MIN_WIND_SPEED:
        u_ref = MIN_WIND_SPEED

    z0_safe = max(z0, 1e-5)
    z_ref_safe = max(z_ref, z0_safe * 1.1)
    z_m_safe = max(z_m, z0_safe * 1.01)

    if u_star is None:
        # Derive friction velocity from reference wind
        u_star = u_ref * VON_KARMAN / math.log(z_ref_safe / z0_safe)

    return (u_star / VON_KARMAN) * math.log(z_m_safe / z0_safe)


def friction_velocity(
    u_ref: float,
    z_ref: float = Z_REF_DEFAULT,
    z0: float = 0.1,
) -> float:
    """Calculate friction velocity u* from reference wind speed.

    u* = u_ref * κ / ln(z_ref / z0)

    Args:
        u_ref: Reference wind speed [m/s].
        z_ref: Reference height [m].
        z0: Surface roughness length [m].

    Returns:
        Friction velocity u* [m/s].
    """
    if u_ref < MIN_WIND_SPEED:
        u_ref = MIN_WIND_SPEED
    z0_safe = max(z0, 1e-5)
    z_ref_safe = max(z_ref, z0_safe * 1.1)
    return u_ref * VON_KARMAN / math.log(z_ref_safe / z0_safe)


# ---------------------------------------------------------------------------
# Atmospheric Density
# ---------------------------------------------------------------------------


def atmospheric_density(
    temperature_k: float = 298.15,
    pressure_pa: float = 101325.0,
    relative_humidity_pct: float = 50.0,
) -> float:
    """Calculate moist air density using ideal gas law with humidity correction.

    ρ = (p_a * M_a + p_v * M_v) / (R * T)

    where:
        p_v = relative_humidity * p_sat(T) / 100
        p_a = pressure - p_v
        M_a = molar mass dry air
        M_v = molar mass water vapor

    Args:
        temperature_k: Air temperature [K].
        pressure_pa: Atmospheric pressure [Pa].
        relative_humidity_pct: Relative humidity [%].

    Returns:
        Air density [kg/m³].

    Examples:
        >>> atmospheric_density(298.15, 101325.0, 50.0)
        1.177...
        >>> atmospheric_density(273.15, 101325.0, 0.0)
        1.292...
    """
    # Saturation vapor pressure using August-Roche-Magnus approximation
    es_pa = saturation_vapor_pressure(temperature_k)

    # Actual vapor pressure
    p_v = (relative_humidity_pct / 100.0) * es_pa

    # Dry air partial pressure
    p_a = pressure_pa - p_v

    # Ideal gas law for mixture
    R = 8.314462618  # universal gas constant J/(mol·K)
    density = (p_a * M_DRY_AIR + p_v * M_WATER_VAPOR) / (R * temperature_k)

    return density


def saturation_vapor_pressure(temperature_k: float) -> float:
    """Calculate saturation vapor pressure over liquid water.

    Uses the August-Roche-Magnus formula:
        e_s = 610.94 * exp(17.625 * (T - 273.15) / (T - 30.11))

    Valid for T between -40°C and +50°C.

    Args:
        temperature_k: Temperature [K].

    Returns:
        Saturation vapor pressure [Pa].
    """
    t_c = temperature_k - 273.15
    # Clamp to valid range for the approximation
    t_c = max(-40.0, min(50.0, t_c))
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


# ---------------------------------------------------------------------------
# Temperature Lapse Rate
# ---------------------------------------------------------------------------


def get_lapse_rate(stability: StabilityClass) -> float:
    """Get atmospheric temperature lapse rate for a stability class.

    Args:
        stability: PG stability class.

    Returns:
        Lapse rate [K/m]. Negative means temperature decreases with height.

    Examples:
        >>> get_lapse_rate('D')
        -0.0098
        >>> get_lapse_rate('A')
        -0.019
    """
    return LAPSE_RATE.get(stability, DRY_ADIABATIC_LAPSE_RATE)


def temperature_at_height(
    t_surface_k: float,
    height_m: float,
    lapse_rate_kpm: float,
) -> float:
    """Calculate temperature at a given height using lapse rate.

    T(z) = T_surface + lapse_rate * z

    Note: lapse_rate is typically negative (temperature decreases with height).

    Args:
        t_surface_k: Surface temperature [K].
        height_m: Height above surface [m].
        lapse_rate_kpm: Lapse rate [K/m] (negative for normal conditions).

    Returns:
        Temperature at height z [K].
    """
    return t_surface_k + lapse_rate_kpm * height_m


def pressure_at_height(
    p_surface_pa: float,
    t_surface_k: float,
    height_m: float,
    lapse_rate_kpm: float = -0.0065,
) -> float:
    """Estimate atmospheric pressure at height using barometric formula.

    Uses the isothermal or constant-lapse-rate barometric formula.

    Args:
        p_surface_pa: Surface pressure [Pa].
        t_surface_k: Surface temperature [K].
        height_m: Height above surface [m].
        lapse_rate_kpm: Lapse rate [K/m].

    Returns:
        Pressure at height [Pa].
    """
    g = 9.80665
    R = 287.058

    if abs(lapse_rate_kpm) < 1e-9:
        # Isothermal atmosphere
        return p_surface_pa * math.exp(-g * height_m / (R * t_surface_k))
    else:
        # Polytropic atmosphere
        exponent = g / (R * lapse_rate_kpm)
        t_z = t_surface_k + lapse_rate_kpm * height_m
        if t_z <= 0:
            t_z = 0.1  # avoid division by zero
        return p_surface_pa * (t_z / t_surface_k) ** (-exponent)


# ---------------------------------------------------------------------------
# Misc Utility
# ---------------------------------------------------------------------------


def wind_direction_to_components(
    speed_ms: float,
    direction_deg: float,
) -> Tuple[float, float]:
    """Convert wind speed and direction to u, v components.

    Uses meteorological convention:
        - Direction: angle FROM which wind blows (0 = North)
        - u = -speed * sin(direction)  (East-West, positive eastward)
        - v = -speed * cos(direction)  (North-South, positive northward)

    Args:
        speed_ms: Wind speed [m/s].
        direction_deg: Wind direction [degrees from north].

    Returns:
        (u, v) components [m/s].
    """
    theta_rad = math.radians(direction_deg)
    u = -speed_ms * math.sin(theta_rad)
    v = -speed_ms * math.cos(theta_rad)
    return u, v


def components_to_wind_direction(
    u_ms: float,
    v_ms: float,
) -> Tuple[float, float]:
    """Convert u, v components back to wind speed and direction.

    Args:
        u_ms: East-west component [m/s] (positive = eastward).
        v_ms: North-south component [m/s] (positive = northward).

    Returns:
        (speed_ms, direction_deg) tuple.
    """
    speed = math.sqrt(u_ms * u_ms + v_ms * v_ms)
    if speed < 1e-9:
        return 0.0, 0.0
    # Meteorological direction: angle FROM which wind blows
    direction_rad = math.atan2(-u_ms, -v_ms)
    direction_deg = math.degrees(direction_rad)
    if direction_deg < 0:
        direction_deg += 360.0
    return speed, direction_deg
