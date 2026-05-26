"""
Rekarisk — Pool Fire Thermal Radiation Models.

Calculates thermal radiation from circular pool fires using Mudan/Thomas
correlations for flame geometry and Shokri-Beyler solid flame model
for radiation.

References:
  - CCPS Guidelines for Consequence Analysis of Chemical Releases (1999)
  - TNO Yellow Book (CPR 14E), Chapter 5 — Pool Fires
  - Mudan, K.S. (1984) — Thermal radiation hazards from hydrocarbon pool fires,
    Progress in Energy and Combustion Science, 10(1), 59-80
  - Thomas, P.H. (1963) — The size of flames from natural fires,
    Proceedings of the Combustion Institute, 9, 844-859
  - Shokri, M. & Beyler, C.L. (1989) — Radiation from large pool fires,
    Journal of Fire Protection Engineering, 1(4), 141-150
  - AGA (1974) — LNG Safety Research Program, Report IS 3-1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.constants import (
    G, T_0C, P_ATM, AIR_DENSITY_NTP, AIR_SPECIFIC_HEAT,
    RADIANT_FRACTION_POOL_FIRE, RADIATION_ENDPOINTS, EPSILON,
)


# ══════════════════════════════════════════════════════════════════════════════
# Default Burning Rate Parameters
# ══════════════════════════════════════════════════════════════════════════════

# (m_dot_inf [kg/(m²·s)], k_beta [m⁻¹])
# Based on Babrauskas (1983), Drysdale (1985), and TNO Yellow Book
BURNING_RATE_PARAMS: Dict[str, Tuple[float, float]] = {
    "methane":       (0.025, 1.1),
    "lng":           (0.025, 1.1),
    "propane":       (0.055, 1.4),
    "lpg":           (0.055, 1.4),
    "butane":        (0.060, 1.4),
    "gasoline":      (0.055, 1.4),
    "petrol":        (0.055, 1.4),
    "kerosene":      (0.039, 1.9),
    "diesel":        (0.045, 2.1),
    "jp-4":          (0.052, 3.6),
    "jp-5":          (0.039, 1.9),
    "hexane":        (0.058, 1.5),
    "heptane":       (0.054, 1.5),
    "octane":        (0.044, 1.5),
    "benzene":       (0.060, 1.5),
    "toluene":       (0.057, 1.5),
    "xylene":        (0.055, 1.5),
    "methanol":      (0.015, 0.8),
    "ethanol":       (0.025, 0.9),
    "crude_oil":     (0.045, 1.5),
    "crude":         (0.045, 1.5),
    "fuel_oil":      (0.035, 2.0),
    "default":       (0.045, 1.5),
}

# Heats of combustion (lower heating value) [J/kg]
# Sources: CCPS, DIPPR, NIST Chemistry WebBook
HEATS_OF_COMBUSTION: Dict[str, float] = {
    "methane":      50.0e6,
    "lng":          50.0e6,
    "propane":      46.3e6,
    "lpg":          46.0e6,
    "butane":       45.7e6,
    "gasoline":     44.0e6,
    "petrol":       44.0e6,
    "kerosene":     43.5e6,
    "diesel":       43.0e6,
    "jp-4":         44.0e6,
    "jp-5":         43.5e6,
    "hexane":       45.1e6,
    "heptane":      44.9e6,
    "octane":       44.4e6,
    "benzene":      40.1e6,
    "toluene":      40.6e6,
    "xylene":       40.9e6,
    "methanol":     19.9e6,
    "ethanol":      26.8e6,
    "crude_oil":    42.5e6,
    "crude":        42.5e6,
    "fuel_oil":     41.0e6,
    "hydrogen":     120.0e6,
    "default":      43.0e6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PoolFireInput:
    """Input parameters for pool fire thermal radiation calculation.

    Attributes:
        pool_diameter: Pool diameter [m].
        substance: Substance name (for auto-selecting properties) or "custom".
        burning_rate: Mass burning rate per unit area [kg/(m²·s)].
            If None, auto-selected from BURNING_RATE_PARAMS.
        heat_of_combustion: Lower heating value [J/kg].
            If None, auto-selected from HEATS_OF_COMBUSTION.
        radiative_fraction: Fraction of total heat radiated [-].
            Default 0.35. Typical range 0.15-0.40.
        wind_speed: Wind speed at 10 m height [m/s]. Default 0.0.
        ambient_temperature: Ambient air temperature [K]. Default 298.15.
        relative_humidity: Relative humidity [%] for transmissivity. Default 50.0.
    """
    pool_diameter: float
    substance: str = "default"

    # Optional overrides
    burning_rate: Optional[float] = None
    heat_of_combustion: Optional[float] = None
    radiative_fraction: float = 0.35

    # Environmental
    wind_speed: float = 0.0
    ambient_temperature: float = 298.15
    relative_humidity: float = 50.0

    def __post_init__(self):
        if self.burning_rate is None:
            self.burning_rate = burning_rate_default(self.substance.lower(), self.pool_diameter)
        if self.heat_of_combustion is None:
            self.heat_of_combustion = HEATS_OF_COMBUSTION.get(
                self.substance.lower(), HEATS_OF_COMBUSTION["default"]
            )


@dataclass
class PoolFireResult:
    """Results from pool fire thermal radiation calculation.

    Attributes:
        pool_diameter: Pool diameter [m].
        flame_length: Visible flame length [m].
        flame_tilt: Flame tilt angle from vertical [deg].
        flame_drag: Downwind flame elongation ratio [-].
        burning_rate: Mass burning rate per unit area [kg/(m²·s)].
        total_burning_rate: Total mass burning rate [kg/s].
        sep: Surface emissive power [kW/m²].
        radiative_fraction: Radiative fraction used [-].
        thermal_radiation_vs_distance: (N,2) array — column 0: distance [m],
            column 1: heat flux [kW/m²].
        distance_to_thresholds: Dict mapping threshold [kW/m²] → distance [m].
        model_used: "point_source" or "solid_flame".
        status_messages: List of info/warning strings.
    """
    pool_diameter: float
    flame_length: float
    flame_tilt: float
    flame_drag: float
    burning_rate: float
    total_burning_rate: float
    sep: float
    radiative_fraction: float
    thermal_radiation_vs_distance: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    distance_to_thresholds: Dict[float, float] = field(default_factory=dict)
    model_used: str = "solid_flame"
    status_messages: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Burning Rate
# ══════════════════════════════════════════════════════════════════════════════

def burning_rate_default(substance: str, pool_diameter: float) -> float:
    """Calculate default mass burning rate for a given substance and pool size.

    Uses Babrauskas correlation:
        m" = m"∞ · (1 - e^(-k·β·D))

    where:
        m"∞ = maximum burning rate for infinite pool [kg/(m²·s)]
        k·β = extinction-absorption coefficient product [m⁻¹]
        D   = pool diameter [m]

    Args:
        substance: Substance name (key into BURNING_RATE_PARAMS).
        pool_diameter: Pool diameter [m].

    Returns:
        Mass burning rate [kg/(m²·s)].
    """
    sub = substance.lower().strip()
    m_dot_inf, k_beta = BURNING_RATE_PARAMS.get(
        sub, BURNING_RATE_PARAMS["default"]
    )
    diam = max(pool_diameter, EPSILON)
    return m_dot_inf * (1.0 - math.exp(-k_beta * diam))


# ══════════════════════════════════════════════════════════════════════════════
# Flame Geometry — Thomas Correlation
# ══════════════════════════════════════════════════════════════════════════════

def flame_length_thomas(
    m_dot: float,
    pool_diameter: float,
    rho_air: float = AIR_DENSITY_NTP,
    wind_speed: float = 0.0,
) -> float:
    """Calculate visible flame length using Thomas correlation.

    No-wind correlation (Thomas 1963):
        L/D = 42 · (m" / (ρ_a · √(g·D)))^0.61

    With wind, a reduction factor is applied (CCPS):
        Lw/L = 0.55 · (uw/u*)^-0.21  for uw/u* ≥ 1

    where u* = (g · m" · D / ρ_v)^(1/3) is the characteristic velocity.

    Args:
        m_dot: Mass burning rate per area [kg/(m²·s)].
        pool_diameter: Pool diameter [m].
        rho_air: Air density at ambient conditions [kg/m³].
        wind_speed: Wind speed at 10 m [m/s].

    Returns:
        Flame length [m].
    """
    if m_dot <= EPSILON or pool_diameter <= EPSILON:
        return 0.0

    D = pool_diameter
    # Non-dimensional burning rate (modified Froude number)
    denom = rho_air * math.sqrt(G * D)
    if denom <= EPSILON:
        return 0.0

    m_star = m_dot / denom

    # No-wind flame length ratio
    L_over_D = 42.0 * (m_star ** 0.61)
    L0 = L_over_D * D

    if wind_speed > 0.1:
        # Characteristic velocity
        rho_v = 1.0  # approximate vapor density [kg/m³] at flame temperature
        u_star = (G * m_dot * D / max(rho_v, EPSILON)) ** (1.0 / 3.0)
        uw_ratio = wind_speed / max(u_star, EPSILON)

        # CCPS wind reduction factor
        if uw_ratio >= 1.0:
            L = L0 * 0.55 * (uw_ratio ** -0.21)
        else:
            L = L0
    else:
        L = L0

    return max(L, EPSILON)


def flame_tilt_aga(
    wind_speed: float,
    pool_diameter: float,
    rho_air: float = AIR_DENSITY_NTP,
    rho_vapor: float = 0.2,
    m_dot: float = 0.05,
) -> Tuple[float, float]:
    """Calculate flame tilt using AGA (American Gas Association) correlation.

    cos(θ) = 1.0                      for u* ≤ 1
    cos(θ) = (u*)^-0.5               for u* > 1

    where u* = u_w / (g · m" · D / ρ_v)^(1/3)

    Flame drag (downwind elongation ratio):
        D' = 1.5 · D · sqrt(u*)      for u* > 0.05

    Args:
        wind_speed: Wind speed [m/s].
        pool_diameter: Pool diameter [m].
        rho_air: Air density [kg/m³].
        rho_vapor: Fuel vapor density [kg/m³] (approximate).
        m_dot: Burning rate [kg/(m²·s)].

    Returns:
        (tilt_angle_from_vertical [deg], flame_drag_ratio [-])
    """
    if wind_speed < EPSILON:
        return 0.0, 1.0

    # Characteristic velocity
    u_c = (G * m_dot * max(pool_diameter, 0.01) / max(rho_vapor, EPSILON)) ** (1.0 / 3.0)

    # Non-dimensional wind speed
    u_star = wind_speed / max(u_c, EPSILON)

    # Tilt angle
    if u_star <= 1.0:
        cos_theta = 1.0
    else:
        cos_theta = u_star ** -0.5

    cos_theta = min(max(cos_theta, 0.001), 1.0)
    tilt_rad = math.acos(cos_theta)
    tilt_deg = math.degrees(tilt_rad)

    # Flame drag
    if u_star > 0.05:
        drag = 1.5 * math.sqrt(u_star)
    else:
        drag = 1.0

    drag = max(drag, 1.0)

    return tilt_deg, drag


# ══════════════════════════════════════════════════════════════════════════════
# Surface Emissive Power
# ══════════════════════════════════════════════════════════════════════════════

def surface_emissive_power(
    m_dot: float,
    dhc: float,
    chi_r: float,
    flame_length: float,
    pool_diameter: float,
) -> float:
    """Calculate surface emissive power using the CCPS/Shokri-Beyler method.

    SEP = χ_r · m" · ΔHc / (1 + 4·L/D)

    For large pools (D > 3 m), SEP approaches the soot-limited value
    (~20-30 kW/m²) due to increased smoke obscuration.

    The CCPS correlation accounts for this with a diameter-dependent
    reduction factor for pools larger than ~10 m.

    Args:
        m_dot: Mass burning rate [kg/(m²·s)].
        dhc: Heat of combustion [J/kg].
        chi_r: Radiative fraction [-].
        flame_length: Flame length [m].
        pool_diameter: Pool diameter [m].

    Returns:
        Surface emissive power [kW/m²].
    """
    if pool_diameter <= EPSILON:
        return 0.0

    L_over_D = flame_length / max(pool_diameter, EPSILON)

    # Theoretical SEP for an optically-thin flame
    sep_theoretical = chi_r * m_dot * dhc / (1.0 + 4.0 * L_over_D)

    # Convert to kW/m²
    sep_kw = sep_theoretical / 1000.0

    # Smoke obscuration correction for large pool fires (CCPS, 1999)
    # For D > 10 m, SEP is limited to soot radiative limit
    D = pool_diameter
    if D > 3.0:
        # Shokri-Beyler correlation for large pool fires
        sep_max_smoky = 20.0 + 120.0 * math.exp(-0.12 * D)
        sep_kw = min(sep_kw, sep_max_smoky)

    return max(sep_kw, EPSILON)


# ══════════════════════════════════════════════════════════════════════════════
# Atmospheric Transmissivity
# ══════════════════════════════════════════════════════════════════════════════

def atmospheric_transmissivity(
    distance: float,
    ambient_temperature: float = 298.15,
    relative_humidity: float = 50.0,
    pool_diameter: float = 1.0,
) -> float:
    """Calculate atmospheric transmissivity for thermal radiation.

    Uses the Baker-Hottel correlation (CCPS, TNO):
        τ = c_w · (RH/100 · P_sat(Ta))^c_h · X^c_d

    Simplified form commonly used in consequence models:
        τ = 1.00 - 0.058 · ln(X)            for hydrocarbons (Simpson)
        τ = exp(-α · (X - D/2))             exponential decay

    We implement the widely-used TNO Yellow Book correlation:
        τ = exp(-κ · X)   where κ = f(RH, T)

    For hydrocarbon fires:
        κ = 0.5 · 10^-4  [m⁻¹]  for clear air
        κ adjusted by humidity factor

    Args:
        distance: Distance from fire center to receiver [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        pool_diameter: Pool diameter [m] (for flame size correction).

    Returns:
        Atmospheric transmissivity [-] (0 to 1).
    """
    if distance < EPSILON:
        return 1.0

    # Saturation vapor pressure of water [Pa] (Tetens formula)
    T_C = ambient_temperature - T_0C
    p_sat = 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))

    # Partial pressure of water vapor [Pa]
    p_w = (relative_humidity / 100.0) * p_sat

    # Absorption coefficient [m⁻¹] (TNO Yellow Book)
    # κ = 2.02 × 10^-5 · p_w^0.09  (p_w in Pa)
    # This gives κ ~ 4-8 × 10^-5 m⁻¹ for typical conditions
    kappa = 2.02e-5 * (max(p_w, 1.0) ** 0.09)

    # Flame-to-receiver distance correction
    effective_distance = max(distance - pool_diameter / 2.0, EPSILON)

    tau = math.exp(-kappa * effective_distance)

    return min(max(tau, 0.01), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# View Factors — Cylindrical Flame
# ══════════════════════════════════════════════════════════════════════════════

def view_factor_cylinder_vertical(
    L: float,
    D: float,
    x: float,
) -> float:
    """View factor from a vertical cylinder flame to a ground-level target.

    Uses the analytical formula from Mudan (1987) for a cylinder
    oriented vertically, with receiver at height zero at horizontal
    distance x from the flame center.

    F = sqrt(F_h² + F_v²)

    where F_h and F_v are horizontal and vertical components.

    Args:
        L: Flame length (cylinder height) [m].
        D: Flame diameter (cylinder diameter) [m].
        x: Distance from flame center to receiver [m].

    Returns:
        View factor [-] (0 to 1).
    """
    if x < EPSILON:
        return 1.0
    if L < EPSILON or D < EPSILON:
        return 0.0

    R = D / 2.0
    H = L
    S = x

    # Parameters
    A = (H ** 2 + S ** 2 - R ** 2) / (2.0 * S ** 2 + EPSILON)
    B = (S ** 2 - R ** 2) / (2.0 * S ** 2 + EPSILON)

    # Vertical component F_v
    # For a vertical cylinder to a perpendicular receiver at ground level
    # (Mudan/Crocker formula)
    term1 = math.atan(math.sqrt(max((A - B) / (1.0 - A + EPSILON), EPSILON)))
    term2 = math.atan(math.sqrt(max((A - B) / (1.0 + A + EPSILON), EPSILON)))

    if S > R:
        F_v = (1.0 / math.pi) * (term1 + term2)
    else:
        # Receiver within cylinder radius projection
        F_v = 0.5

    # Horizontal component F_h (Mudan, 1987 — cylindrical flame geometry)
    h = H / max(R, EPSILON)
    s = S / max(R, EPSILON)

    if s > 1.0 + EPSILON:
        # Formulas for horizontal component of cylindrical flame view factor
        A1 = h * h + s * s + 1.0
        B1 = s * s + 1.0
        F_h = (1.0 / (math.pi * s)) * math.atan(
            math.sqrt(max((s - 1.0) / (s + 1.0), EPSILON))
        )
        F_h += ((A1 - 2.0 * s * (s + 1.0)) / (math.pi * s * math.sqrt(max(A1 * A1 - 4.0 * s * s, EPSILON)))) * math.atan(
            math.sqrt(max((s - 1.0) / (s + 1.0) * A1 / B1, EPSILON))
        )
    else:
        F_h = 1.0

    F_h = abs(F_h)

    # Total view factor
    F = math.sqrt(F_v ** 2 + F_h ** 2)

    return min(F, 1.0)


def view_factor_cylinder_tilted(
    L: float,
    D: float,
    x: float,
    tilt_deg: float,
    wind_direction: str = "downwind",
) -> float:
    """View factor from a tilted cylinder flame to a ground-level target.

    For a cylindrical flame tilted by wind, the receiver at ground level
    sees a projected area. The downwind and upwind view factors differ.

    We decompose the tilted cylinder into a superposition of vertical
    and horizontal projected areas and apply the appropriate geometric
    factors.

    Args:
        L: Flame length [m].
        D: Flame diameter [m].
        x: Horizontal distance from flame base center to receiver [m].
        tilt_deg: Tilt angle from vertical [deg].
        wind_direction: "downwind" or "upwind" — receiver location
            relative to wind direction.

    Returns:
        View factor [-] (0 to 1).
    """
    if x < EPSILON:
        return 1.0
    if L < EPSILON or D < EPSILON:
        return 0.0

    tilt_rad = math.radians(tilt_deg)

    # Projected dimensions of tilted cylinder
    # Vertical projection height
    H_proj = L * math.cos(tilt_rad)
    # Horizontal projection length
    X_proj = L * math.sin(tilt_rad)

    # Effective center displacement
    if wind_direction == "downwind":
        # Flame tilts away from receiver (downwind)
        center_offset = X_proj / 2.0
        effective_x = x + center_offset
    else:
        # Upwind — flame tilts toward receiver
        center_offset = X_proj / 2.0
        effective_x = max(x - center_offset, EPSILON)

    # Effective flame height (mid-point of tilted flame above grade)
    H_eff = H_proj / 2.0

    # Compute view factor using vertical cylinder approximation
    # at effective distance with effective height
    R = D / 2.0
    S = effective_x
    H = H_proj

    # Vertical component
    A = (H ** 2 + S ** 2 - R ** 2) / (2.0 * S ** 2 + EPSILON)
    B = (S ** 2 - R ** 2) / (2.0 * S ** 2 + EPSILON)

    if S > R:
        term1 = math.atan(math.sqrt(max((A - B) / (1.0 - A + EPSILON), EPSILON)))
        term2 = math.atan(math.sqrt(max((A - B) / (1.0 + A + EPSILON), EPSILON)))
        F_v = (1.0 / math.pi) * (term1 + term2)
    else:
        F_v = 0.5

    # Horizontal component (simplified for tilted geometry)
    if S > R:
        # Increase in view factor due to horizontal projection (tilt)
        # The horizontal component adds ~ sin(tilt) contribution
        F_h_pure = view_factor_cylinder_vertical(L, D, effective_x)

        # Weight between pure vertical and tilted
        F = F_v * math.cos(tilt_rad) + F_h_pure * math.sin(tilt_rad)
    else:
        F = F_v

    return min(max(F, 0.0), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Thermal Radiation
# ══════════════════════════════════════════════════════════════════════════════

def thermal_radiation_point_source(
    total_burning_rate: float,
    heat_of_combustion: float,
    radiative_fraction: float,
    distance: float,
    tau: float = 1.0,
) -> float:
    """Calculate thermal radiation using point source model.

    Treats the entire fire as a single point source radiating equally
    in all directions.

        q = τ · χ_r · Q̇ / (4π · R²)

    where Q̇ = ṁ · ΔHc is the total heat release rate.

    Simple, conservative at far field, underestimates at near field.

    Args:
        total_burning_rate: Total mass burning rate [kg/s].
        heat_of_combustion: Lower heating value [J/kg].
        radiative_fraction: Radiative fraction [-].
        distance: Distance from fire center to receiver [m].
        tau: Atmospheric transmissivity [-].

    Returns:
        Thermal radiation flux [kW/m²].
    """
    if distance < EPSILON:
        return float('inf')

    Q_dot = total_burning_rate * heat_of_combustion  # [W]
    q_rad = tau * radiative_fraction * Q_dot / (4.0 * math.pi * distance ** 2)

    return q_rad / 1000.0  # Convert to kW/m²


def thermal_radiation_solid_flame(
    sep: float,
    view_factor: float,
    tau: float = 1.0,
) -> float:
    """Calculate thermal radiation using solid flame model.

    Treats the flame as a solid body emitting at the surface emissive
    power. This is more accurate at near field.

        q = τ · SEP · F_view

    Args:
        sep: Surface emissive power [kW/m²].
        view_factor: Geometric view factor [-].
        tau: Atmospheric transmissivity [-].

    Returns:
        Thermal radiation flux [kW/m²].
    """
    return tau * sep * view_factor


# ══════════════════════════════════════════════════════════════════════════════
# Distance Sweep
# ══════════════════════════════════════════════════════════════════════════════

def thermal_radiation_vs_distance(
    sep: float,
    flame_length: float,
    pool_diameter: float,
    tilt_deg: float,
    ambient_temperature: float,
    relative_humidity: float,
    min_distance: float = 0.5,
    max_distance: float = 200.0,
    n_points: int = 200,
    model: str = "solid_flame",
    total_burning_rate: Optional[float] = None,
    heat_of_combustion: Optional[float] = None,
    radiative_fraction: Optional[float] = None,
) -> np.ndarray:
    """Compute thermal radiation vs distance curve.

    Args:
        sep: Surface emissive power [kW/m²].
        flame_length: Flame length [m].
        pool_diameter: Pool diameter [m].
        tilt_deg: Flame tilt angle from vertical [deg].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        min_distance: Minimum receiver distance [m].
        max_distance: Maximum receiver distance [m].
        n_points: Number of evaluation points.
        model: "point_source" or "solid_flame".
        total_burning_rate: Total burning rate [kg/s] (for point source).
        heat_of_combustion: Heat of combustion [J/kg] (for point source).
        radiative_fraction: Radiative fraction [-] (for point source).

    Returns:
        (N, 2) numpy array — column 0: distance [m], column 1: flux [kW/m²].
    """
    distances = np.linspace(min_distance, max_distance, n_points)
    fluxes = np.zeros(n_points)

    for i, d in enumerate(distances):
        tau = atmospheric_transmissivity(
            d, ambient_temperature, relative_humidity, pool_diameter
        )

        if model == "point_source":
            if total_burning_rate is not None and heat_of_combustion is not None:
                fluxes[i] = thermal_radiation_point_source(
                    total_burning_rate,
                    heat_of_combustion,
                    radiative_fraction or 0.35,
                    d,
                    tau,
                )
            else:
                # Fallback to solid flame
                F = view_factor_cylinder_tilted(
                    flame_length, pool_diameter, d, tilt_deg
                )
                fluxes[i] = thermal_radiation_solid_flame(sep, F, tau)
        else:
            F = view_factor_cylinder_tilted(
                flame_length, pool_diameter, d, tilt_deg
            )
            fluxes[i] = thermal_radiation_solid_flame(sep, F, tau)

    result = np.column_stack((distances, fluxes))
    return result


def distance_to_thresholds(
    sep: float,
    flame_length: float,
    pool_diameter: float,
    tilt_deg: float,
    ambient_temperature: float,
    relative_humidity: float,
    thresholds: Optional[List[float]] = None,
    max_search_distance: float = 500.0,
) -> Dict[float, float]:
    """Find distance to each thermal radiation threshold.

    Uses binary search to find the distance where the heat flux
    drops to each specified threshold level.

    Args:
        sep: Surface emissive power [kW/m²].
        flame_length: Flame length [m].
        pool_diameter: Pool diameter [m].
        tilt_deg: Flame tilt angle from vertical [deg].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        thresholds: List of threshold flux values [kW/m²].
            Default: [37.5, 25.0, 12.5, 5.0, 4.0] kW/m².
        max_search_distance: Maximum distance to search [m].

    Returns:
        Dict mapping threshold [kW/m²] → distance [m].
        Returns max_search_distance if threshold not reached.
    """
    if thresholds is None:
        thresholds = [37.5, 25.0, 12.5, 5.0, 4.0]

    result = {}

    for threshold in thresholds:
        # Binary search for distance where q ≈ threshold
        lo = max(pool_diameter / 2.0, 0.1)
        hi = max_search_distance

        # Check if threshold is even reachable at minimum distance
        tau_min = atmospheric_transmissivity(
            lo, ambient_temperature, relative_humidity, pool_diameter
        )
        F_min = view_factor_cylinder_tilted(flame_length, pool_diameter, lo, tilt_deg)
        q_min = thermal_radiation_solid_flame(sep, F_min, tau_min)

        if q_min <= threshold:
            # Even at minimum distance, flux is below threshold
            # Need to search closer → threshold never reached
            result[threshold] = lo if q_min >= threshold else 0.0
            continue

        # Check if threshold is reached by maximum distance
        tau_max = atmospheric_transmissivity(
            hi, ambient_temperature, relative_humidity, pool_diameter
        )
        F_max = view_factor_cylinder_tilted(flame_length, pool_diameter, hi, tilt_deg)
        q_max = thermal_radiation_solid_flame(sep, F_max, tau_max)

        if q_max > threshold:
            # Not reached even at max search distance
            result[threshold] = max_search_distance
            continue

        # Binary search
        for _ in range(50):  # 50 iterations gives ~10^-15 precision
            mid = (lo + hi) / 2.0
            tau_mid = atmospheric_transmissivity(
                mid, ambient_temperature, relative_humidity, pool_diameter
            )
            F_mid = view_factor_cylinder_tilted(
                flame_length, pool_diameter, mid, tilt_deg
            )
            q_mid = thermal_radiation_solid_flame(sep, F_mid, tau_mid)

            if q_mid > threshold:
                lo = mid
            else:
                hi = mid

            if hi - lo < 0.001:
                break

        result[threshold] = round((lo + hi) / 2.0, 2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_pool_fire(
    input_data: PoolFireInput,
    model: str = "solid_flame",
    min_distance: float = 0.5,
    max_distance: float = 200.0,
    n_points: int = 200,
) -> PoolFireResult:
    """Calculate pool fire thermal radiation.

    Full calculation pipeline:
        1. Burning rate (auto or user-specified)
        2. Flame geometry (Thomas length, AGA tilt)
        3. Surface emissive power (SEP)
        4. Thermal radiation vs distance curve
        5. Distance to threshold mapping

    Args:
        input_data: PoolFireInput with pool diameter, substance, and environment.
        model: "solid_flame" (default) or "point_source".
        min_distance: Min distance for radiation curve [m].
        max_distance: Max distance for radiation curve [m].
        n_points: Number of points on radiation curve.

    Returns:
        PoolFireResult with all outputs.
    """
    messages = []
    D = input_data.pool_diameter
    m_dot = input_data.burning_rate
    dhc = input_data.heat_of_combustion
    chi_r = input_data.radiative_fraction

    if D <= 0.0:
        return PoolFireResult(
            pool_diameter=D,
            flame_length=0.0,
            flame_tilt=0.0,
            flame_drag=1.0,
            burning_rate=m_dot,
            total_burning_rate=0.0,
            sep=0.0,
            radiative_fraction=chi_r,
            status_messages=["Pool diameter must be positive"],
        )

    # 1. Flame geometry
    L = flame_length_thomas(m_dot, D, AIR_DENSITY_NTP, input_data.wind_speed)
    tilt_deg, drag = flame_tilt_aga(
        input_data.wind_speed, D, AIR_DENSITY_NTP, m_dot=m_dot
    )

    if L <= EPSILON:
        messages.append("Warning: Calculated flame length is near zero")

    # 2. Surface emissive power
    sep = surface_emissive_power(m_dot, dhc, chi_r, L, D)

    # 3. Total burning rate
    pool_area = math.pi * (D / 2.0) ** 2
    total_mdot = m_dot * pool_area

    # 4. Radiation vs distance
    rad_vs_dist = thermal_radiation_vs_distance(
        sep=sep,
        flame_length=L,
        pool_diameter=D,
        tilt_deg=tilt_deg,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        min_distance=min_distance,
        max_distance=max_distance,
        n_points=n_points,
        model=model,
        total_burning_rate=total_mdot,
        heat_of_combustion=dhc,
        radiative_fraction=chi_r,
    )

    # 5. Distance to thresholds
    thresholds = distance_to_thresholds(
        sep=sep,
        flame_length=L,
        pool_diameter=D,
        tilt_deg=tilt_deg,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        thresholds=[37.5, 25.0, 12.5, 5.0, 4.0],
    )

    # Validation checks
    if L > D * 20:
        messages.append("Info: Flame length is > 20× pool diameter — check inputs")
    if tilt_deg > 80:
        messages.append("Info: Flame is nearly horizontal due to high wind")
    if sep < 5.0:
        messages.append("Warning: Surface emissive power is very low — check radiative fraction")

    return PoolFireResult(
        pool_diameter=D,
        flame_length=round(L, 3),
        flame_tilt=round(tilt_deg, 1),
        flame_drag=round(drag, 3),
        burning_rate=m_dot,
        total_burning_rate=round(total_mdot, 4),
        sep=round(sep, 2),
        radiative_fraction=chi_r,
        thermal_radiation_vs_distance=rad_vs_dist,
        distance_to_thresholds=thresholds,
        model_used=model,
        status_messages=messages,
    )
