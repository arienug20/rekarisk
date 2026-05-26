"""
Rekarisk — Jet Fire Thermal Radiation Models.

Calculates thermal radiation from turbulent jet fires (gas/liquid spray)
following API RP 521, Kalghatgi, and Considine & Grint correlations.

References:
  - API RP 521 (2014) — Pressure-Relieving and Depressuring Systems
  - Kalghatgi, G.T. (1983) — The visible shape and size of a turbulent
    hydrocarbon jet diffusion flame, Combustion and Flame, 52, 91-106
  - Considine, M. & Grint, G.C. (1985) — Rapid assessment of the
    consequences of jet releases, IChemE Symposium Series No. 93
  - TNO Yellow Book (CPR 14E), Chapter 4 — Jet Fires
  - Chamberlain, G.A. (1987) — Developments in design methods for
    predicting thermal radiation from flares, Chem. Eng. Res. Des., 65
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...core.constants import (
    G, P_ATM, AIR_DENSITY_NTP, EPSILON,
)

# Substance molecular weights [kg/mol]
MOLECULAR_WEIGHTS: Dict[str, float] = {
    "methane": 0.01604,
    "ethane": 0.03007,
    "propane": 0.04410,
    "butane": 0.05812,
    "pentane": 0.07215,
    "hexane": 0.08618,
    "heptane": 0.10021,
    "octane": 0.11423,
    "benzene": 0.07811,
    "toluene": 0.09214,
    "xylene": 0.10617,
    "methanol": 0.03204,
    "ethanol": 0.04607,
    "hydrogen": 0.002016,
    "ethylene": 0.02805,
    "propylene": 0.04208,
    "ammonia": 0.01703,
    "lpg": 0.051,
    "lng": 0.016,
    "gasoline": 0.100,
    "kerosene": 0.150,
    "diesel": 0.170,
    "default": 0.029,
}

# Heats of combustion [J/kg]
JET_HEATS_OF_COMBUSTION: Dict[str, float] = {
    "methane": 50.0e6,
    "ethane": 47.5e6,
    "propane": 46.3e6,
    "butane": 45.7e6,
    "pentane": 45.4e6,
    "hexane": 45.1e6,
    "heptane": 44.9e6,
    "octane": 44.4e6,
    "benzene": 40.1e6,
    "toluene": 40.6e6,
    "xylene": 40.9e6,
    "methanol": 19.9e6,
    "ethanol": 26.8e6,
    "hydrogen": 120.0e6,
    "ethylene": 47.2e6,
    "propylene": 45.8e6,
    "ammonia": 18.6e6,
    "lpg": 46.0e6,
    "lng": 50.0e6,
    "gasoline": 44.0e6,
    "kerosene": 43.5e6,
    "diesel": 43.0e6,
    "default": 45.0e6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JetFireInput:
    """Input parameters for jet fire thermal radiation calculation.

    Attributes:
        orifice_diameter: Orifice/hole diameter [m].
        discharge_velocity: Discharge velocity at orifice [m/s].
        mass_flow_rate: Mass flow rate [kg/s]. If None, estimated from
            orifice and velocity: mdot = ρ · A · v.
        substance: Substance name for auto property lookup.
        heat_of_combustion: Lower heating value [J/kg]. Auto if None.
        radiative_fraction: Radiative fraction [-]. Default 0.30.
        wind_speed: Wind speed at 10 m [m/s]. Default 0.0.
        release_direction: 'horizontal' or 'vertical'.
        ambient_temperature: Ambient temperature [K]. Default 298.15.
        relative_humidity: Relative humidity [%]. Default 50.0.
        discharge_density: Density at orifice exit [kg/m³].
            Needed if mass_flow_rate is None.
    """
    orifice_diameter: float
    discharge_velocity: float
    mass_flow_rate: Optional[float] = None
    substance: str = "default"

    # Optional overrides
    heat_of_combustion: Optional[float] = None
    radiative_fraction: float = 0.30

    # Environmental and configuration
    wind_speed: float = 0.0
    release_direction: str = "horizontal"
    ambient_temperature: float = 298.15
    relative_humidity: float = 50.0

    # Physical properties
    discharge_density: Optional[float] = None

    def __post_init__(self):
        if self.heat_of_combustion is None:
            self.heat_of_combustion = JET_HEATS_OF_COMBUSTION.get(
                self.substance.lower(), JET_HEATS_OF_COMBUSTION["default"]
            )
        if self.mass_flow_rate is None:
            if self.discharge_density is not None and self.orifice_diameter > 0:
                area = math.pi * (self.orifice_diameter / 2.0) ** 2
                self.mass_flow_rate = self.discharge_density * area * self.discharge_velocity
            else:
                self.mass_flow_rate = 0.0


@dataclass
class JetFireResult:
    """Results from jet fire thermal radiation calculation.

    Attributes:
        flame_length: Visible flame length [m].
        flame_width: Flame width at widest point [m].
        flame_center_height: Center of flame above release point [m].
        flame_tilt_deg: Flame tilt from vertical due to wind [deg].
        sep: Surface emissive power [kW/m²].
        total_heat_release: Total heat release rate [W].
        thermal_radiation_vs_distance: (N,2) array — [distance_m, flux_kW_per_m2].
        distance_to_thresholds: Dict threshold [kW/m²] → distance [m].
        model_used: "point_source" or "solid_flame".
        status_messages: List of info/warning strings.
    """
    flame_length: float
    flame_width: float
    flame_center_height: float
    flame_tilt_deg: float
    sep: float
    total_heat_release: float
    thermal_radiation_vs_distance: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    distance_to_thresholds: Dict[float, float] = field(default_factory=dict)
    model_used: str = "point_source"
    status_messages: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Flame Length — Kalghatgi / Vertical Jet
# ══════════════════════════════════════════════════════════════════════════════

def flame_length_vertical_jet(
    orifice_diameter: float,
    jet_density: float,
    air_density: float = AIR_DENSITY_NTP,
) -> float:
    """Calculate flame length for a vertical momentum-dominated jet.

    Uses the simple momentum-ratio correlation:
        L ≈ 18 · D · √(ρ_j / ρ_a)

    where:
        D = orifice diameter [m]
        ρ_j = jet density at orifice [kg/m³]
        ρ_a = air density [kg/m³]

    This formula applies for subsonic, momentum-dominated turbulent
    jet diffusion flames (TNO Yellow Book, CCPS).

    Args:
        orifice_diameter: Orifice diameter [m].
        jet_density: Density of fluid at orifice exit [kg/m³].
        air_density: Ambient air density [kg/m³].

    Returns:
        Flame length [m].
    """
    if orifice_diameter <= EPSILON or jet_density <= EPSILON:
        return 0.0

    density_ratio = jet_density / max(air_density, EPSILON)
    L = 18.0 * orifice_diameter * math.sqrt(density_ratio)

    return max(L, EPSILON)


def flame_length_kalghatgi(
    orifice_diameter: float,
    discharge_velocity: float,
    jet_density: float,
    air_density: float = AIR_DENSITY_NTP,
    wind_speed: float = 0.0,
    release_direction: str = "horizontal",
) -> float:
    """Calculate visible flame length using Kalghatgi correlation.

    For turbulent jet diffusion flames (API RP 521 / Kalghatgi):

    Vertical turbulent jet:
        L = 0.00326 · (ṁ / √D)^(1/3) · (ΔHc / 10^6)^(2/3)  [m]

    Or, using the momentum jet length equation:
        L/D = 6.0 · Re^(0.25) · √(ρ_j/ρ_a)

    For subsonic jets, a simpler CCPS form is used:
        L ≈ k1 · (ṁ · ΔHc)^(k2)

    We implement the API 521 / Considine & Grint approach:
        L = 5.3 · D · √(ρ_j / ρ_a)

    With wind correction:
        Flame shortens due to enhanced air entrainment.

    Args:
        orifice_diameter: Orifice diameter [m].
        discharge_velocity: Discharge velocity [m/s].
        jet_density: Jet density at orifice exit [kg/m³].
        air_density: Ambient air density [kg/m³].
        wind_speed: Wind speed [m/s].
        release_direction: 'horizontal' or 'vertical'.

    Returns:
        Flame length [m].
    """
    if orifice_diameter <= EPSILON or jet_density <= EPSILON:
        return 0.0

    D = orifice_diameter
    density_ratio = jet_density / max(air_density, EPSILON)

    # Base flame length for momentum-dominated jet
    L0 = 5.3 * D * math.sqrt(density_ratio)

    # Reynolds number for transition check
    # Re = ρ · v · D / μ  — using approximate air viscosity
    Re = jet_density * discharge_velocity * D / 1.8e-5

    # For laminar/low-Re jets, length is shorter
    if Re < 2000:
        L0 *= (Re / 2000.0) ** 0.5

    # Wind effect on flame length
    if wind_speed > 0.5:
        # Enhanced entrainment → shorter flame
        # API 521: L_wind / L_no_wind = f(U_wind / U_jet)
        velocity_ratio = wind_speed / max(discharge_velocity, EPSILON)

        if release_direction == "horizontal":
            # Horizontal jet in crosswind — strong reduction
            reduction = max(0.3, math.exp(-2.0 * velocity_ratio))
        else:
            # Vertical jet — moderate reduction at low wind
            reduction = max(0.5, math.exp(-1.0 * velocity_ratio))

        L = L0 * reduction
    else:
        L = L0

    return max(L, EPSILON)


# ══════════════════════════════════════════════════════════════════════════════
# Flame Geometry Extensions
# ══════════════════════════════════════════════════════════════════════════════

def flame_width_cone(flame_length: float) -> float:
    """Estimate flame width at widest point.

    For a turbulent jet flame approximating a cone:
        W ≈ L / 5  (typical aspect ratio for momentum-dominated flames)

    Or, from Chamberlain:
        W = 0.12 · L  (for subsonic releases)

    Args:
        flame_length: Flame length [m].

    Returns:
        Flame width at widest point [m].
    """
    return max(0.12 * flame_length, EPSILON)


def flame_center_height(
    flame_length: float,
    release_direction: str = "horizontal",
    wind_speed: float = 0.0,
    discharge_velocity: float = 0.0,
) -> float:
    """Estimate flame center height above release point.

    For horizontal release: center at ~0 (ground-level flame), rises
    due to buoyancy.
    For vertical release: center at L/2 above release.

    With wind, horizontal jet bends and rises:
        H_center ≈ L · sin(θ_wind)

    Args:
        flame_length: Flame length [m].
        release_direction: 'horizontal' or 'vertical'.
        wind_speed: Wind speed [m/s].
        discharge_velocity: Discharge velocity [m/s].

    Returns:
        Flame center height above release [m].
    """
    if release_direction == "vertical":
        return flame_length / 2.0

    # Horizontal: buoyancy causes the flame to rise
    # Flame rise depends on the competition between momentum and buoyancy
    # For simple estimate: center at 0.2 × flame_length above release
    base_height = 0.2 * flame_length

    if wind_speed > 0.1 and discharge_velocity > 0.1:
        velocity_ratio = wind_speed / max(discharge_velocity, EPSILON)
        base_height *= (1.0 + velocity_ratio)

    return max(base_height, EPSILON)


# ══════════════════════════════════════════════════════════════════════════════
# Flame Tilt due to Wind
# ══════════════════════════════════════════════════════════════════════════════

def flame_tilt_jet(
    wind_speed: float,
    discharge_velocity: float,
    release_direction: str = "horizontal",
) -> float:
    """Calculate flame tilt angle for jet fires.

    For a jet in crosswind:
        tan(θ) ≈ (π · C_D · ρ_a · U_w² · D · L) / (2 · ρ_j · A · U_j²)

    Simplified (API 521):
        θ = atan(U_w / U_j)  (for horizontal jet)
        θ = 0               (for vertical jet, slight tilt ignored here)

    Args:
        wind_speed: Wind speed [m/s].
        discharge_velocity: Jet discharge velocity [m/s].
        release_direction: 'horizontal' or 'vertical'.

    Returns:
        Flame tilt angle from vertical [deg].
    """
    if wind_speed < 0.1 or discharge_velocity < EPSILON:
        return 0.0

    if release_direction == "vertical":
        # Vertical jet — wind bends flame slightly
        ratio = wind_speed / max(discharge_velocity, EPSILON)
        tilt_rad = math.atan(min(ratio, 10.0))
    else:
        # Horizontal jet — wind direction dominates at low jet velocity
        ratio = wind_speed / max(discharge_velocity, EPSILON)
        # Base tilt of a horizontal flame is ~90° from vertical
        # Wind adds to this
        base_tilt = math.radians(60.0)  # horizontal jet is tilted ~60° from vertical
        wind_effect = math.atan(min(ratio, 10.0))
        tilt_rad = base_tilt + wind_effect * math.cos(base_tilt)

    tilt_deg = math.degrees(tilt_rad)
    return min(tilt_deg, 89.0)


# ══════════════════════════════════════════════════════════════════════════════
# Surface Emissive Power — Jet Fire
# ══════════════════════════════════════════════════════════════════════════════

def sep_jet_fire(
    total_heat_release_rate: float,
    flame_length: float,
    flame_width: float,
    radiative_fraction: float,
) -> float:
    """Calculate surface emissive power for jet fire.

    SEP = χ_r · Q̇ / A_flame

    where:
        Q̇ = ṁ · ΔHc  [W]
        A_flame = π · D · L (cylinder approximation) [m²]

    Args:
        total_heat_release_rate: Total heat release rate [W].
        flame_length: Flame length [m].
        flame_width: Flame width [m].
        radiative_fraction: Radiative fraction [-].

    Returns:
        Surface emissive power [kW/m²].
    """
    if flame_length <= EPSILON or flame_width <= EPSILON:
        return 0.0

    # Flame area (cylinder surface)
    R = flame_width / 2.0
    A_flame = 2.0 * math.pi * R * flame_length + math.pi * R ** 2  # side + top

    if A_flame <= EPSILON:
        return 0.0

    sep_w_per_m2 = radiative_fraction * total_heat_release_rate / A_flame
    return sep_w_per_m2 / 1000.0  # kW/m²


# ══════════════════════════════════════════════════════════════════════════════
# Thermal Radiation — Point Source
# ══════════════════════════════════════════════════════════════════════════════

def thermal_radiation_point_source(
    total_heat_release: float,
    radiative_fraction: float,
    distance: float,
    tau: float = 1.0,
) -> float:
    """Point source model — radiation at a given distance.

    API 521: point source at 1/3 flame length from the release point.

        q = τ · χ_r · Q̇ / (4π · R²)

    Args:
        total_heat_release: Total heat release rate [W].
        radiative_fraction: Radiative fraction [-].
        distance: Distance from point source to receiver [m].
        tau: Atmospheric transmissivity [-].

    Returns:
        Heat flux [kW/m²].
    """
    if distance < EPSILON:
        return float('inf')

    q_rad = tau * radiative_fraction * total_heat_release / (4.0 * math.pi * distance ** 2)
    return q_rad / 1000.0


def thermal_radiation_solid_flame(
    sep: float,
    view_factor: float,
    tau: float = 1.0,
) -> float:
    """Solid flame model — radiation at a given distance.

        q = τ · SEP · F

    Args:
        sep: Surface emissive power [kW/m²].
        view_factor: View factor [-].
        tau: Atmospheric transmissivity [-].

    Returns:
        Heat flux [kW/m²].
    """
    return tau * sep * view_factor


# ══════════════════════════════════════════════════════════════════════════════
# Atmospheric Transmissivity (shared utility)
# ══════════════════════════════════════════════════════════════════════════════

def atmospheric_transmissivity(
    distance: float,
    ambient_temperature: float = 298.15,
    relative_humidity: float = 50.0,
) -> float:
    """Atmospheric transmissivity for thermal radiation.

    Simplified TNO correlation:
        τ = exp(-κ · X)

    where κ ≈ 5 × 10⁻⁵ m⁻¹ adjusted for humidity.

    Args:
        distance: Path length [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].

    Returns:
        Transmissivity [-] (0 to 1).
    """
    if distance < EPSILON:
        return 1.0

    # Saturation vapor pressure (Tetens)
    T_C = ambient_temperature - 273.15
    p_sat = 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))
    p_w = (relative_humidity / 100.0) * p_sat

    kappa = 2.02e-5 * (max(p_w, 1.0) ** 0.09)
    tau = math.exp(-kappa * distance)

    return min(max(tau, 0.01), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# View Factor — Jet Flame as Tilted Cylinder
# ══════════════════════════════════════════════════════════════════════════════

def view_factor_jet_flame(
    flame_length: float,
    flame_width: float,
    distance: float,
    tilt_deg: float = 0.0,
    center_height: float = 0.0,
) -> float:
    """View factor from a jet flame (approximated as cylinder) to target.

    Uses simplified geometric approximation for a cylinder at height
    h_center above ground, tilted by wind.

    Args:
        flame_length: Flame length [m].
        flame_width: Flame width [m].
        distance: Horizontal distance from flame center to receiver [m].
        tilt_deg: Flame tilt from vertical [deg].
        center_height: Height of flame center above grade [m].

    Returns:
        View factor [-] (0 to 1).
    """
    if distance < EPSILON:
        return 1.0
    if flame_length < EPSILON or flame_width < EPSILON:
        return 0.0

    L = flame_length
    D = flame_width
    x = distance

    # Adjust for tilted flame — effective dimensions
    tilt_rad = math.radians(tilt_deg)

    # Vertical projection height
    H_eff = L * math.cos(tilt_rad) + center_height

    # Horizontal extent
    X_eff = L * math.sin(tilt_rad)

    # Effective distance from flame mid-point
    effective_x = math.sqrt((x + X_eff / 2.0) ** 2 + H_eff ** 2)

    if effective_x < EPSILON:
        return 1.0

    # Point source approximation for view factor
    # F = A_projected / (π · R_effective²)
    # Project flame area toward receiver
    R = D / 2.0

    # Cylinder projected area toward receiver
    # Side area: 2·R·L (rectangle projection)
    # Top area: π·R² (circle projection)
    # For tilted cylinder, projected area is proportionally reduced
    area_projected = 2.0 * R * L * math.cos(tilt_rad) + math.pi * R ** 2 * math.sin(tilt_rad)
    area_projected = max(area_projected, EPSILON)

    F = area_projected / (4.0 * math.pi * effective_x ** 2)

    return min(F, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Distance Sweep
# ══════════════════════════════════════════════════════════════════════════════

def thermal_radiation_vs_distance_jet(
    total_heat_release: float,
    radiative_fraction: float,
    sep: float,
    flame_length: float,
    flame_width: float,
    tilt_deg: float,
    center_height: float,
    ambient_temperature: float,
    relative_humidity: float,
    min_distance: float = 1.0,
    max_distance: float = 200.0,
    n_points: int = 200,
    model: str = "point_source",
) -> np.ndarray:
    """Compute thermal radiation vs distance for jet fire.

    Args:
        total_heat_release: Total heat release rate [W].
        radiative_fraction: Radiative fraction [-].
        sep: Surface emissive power [kW/m²].
        flame_length: Flame length [m].
        flame_width: Flame width [m].
        tilt_deg: Flame tilt [deg].
        center_height: Flame center height [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        min_distance: Minimum distance [m].
        max_distance: Maximum distance [m].
        n_points: Number of points.
        model: "point_source" or "solid_flame".

    Returns:
        (N, 2) array — [distance_m, flux_kW_per_m2].
    """
    distances = np.linspace(min_distance, max_distance, n_points)
    fluxes = np.zeros(n_points)

    for i, d in enumerate(distances):
        tau = atmospheric_transmissivity(d, ambient_temperature, relative_humidity)

        if model == "point_source":
            # Effective distance from flame center to receiver
            # Flame center is at center_height above release
            # For horizontal release, center at (L/2, h_center) from release point
            # For simplicity: point source at (0, L/3) from release
            flame_center_dist_x = max(d, EPSILON)
            flame_center_height_eff = center_height + flame_length / 3.0

            eff_dist = math.sqrt(
                flame_center_dist_x ** 2 + flame_center_height_eff ** 2
            )
            fluxes[i] = thermal_radiation_point_source(
                total_heat_release, radiative_fraction, eff_dist, tau
            )
        else:
            F = view_factor_jet_flame(
                flame_length, flame_width, d, tilt_deg, center_height
            )
            fluxes[i] = thermal_radiation_solid_flame(sep, F, tau)

    return np.column_stack((distances, fluxes))


# ══════════════════════════════════════════════════════════════════════════════
# Distance to Thresholds
# ══════════════════════════════════════════════════════════════════════════════

def distance_to_thresholds_jet(
    total_heat_release: float,
    radiative_fraction: float,
    sep: float,
    flame_length: float,
    flame_width: float,
    tilt_deg: float,
    center_height: float,
    ambient_temperature: float,
    relative_humidity: float,
    thresholds: Optional[List[float]] = None,
    max_search_distance: float = 500.0,
    model: str = "point_source",
) -> Dict[float, float]:
    """Find distances to thermal radiation thresholds for jet fire.

    Args:
        (same as thermal_radiation_vs_distance_jet)
        thresholds: List of thresholds [kW/m²].
        max_search_distance: Max search distance [m].
        model: "point_source" or "solid_flame".

    Returns:
        Dict threshold → distance [m].
    """
    if thresholds is None:
        thresholds = [37.5, 25.0, 12.5, 5.0, 4.0]

    result = {}

    for threshold in thresholds:
        lo = max(flame_width / 2.0, 0.1)
        hi = max_search_distance

        # Evaluate flux at lo
        tau_lo = atmospheric_transmissivity(lo, ambient_temperature, relative_humidity)

        if model == "point_source":
            eff_dist = math.sqrt(lo ** 2 + (center_height + flame_length / 3.0) ** 2)
            q_lo = thermal_radiation_point_source(
                total_heat_release, radiative_fraction, eff_dist, tau_lo
            )
        else:
            F_lo = view_factor_jet_flame(flame_length, flame_width, lo, tilt_deg, center_height)
            q_lo = thermal_radiation_solid_flame(sep, F_lo, tau_lo)

        if q_lo <= threshold:
            result[threshold] = 0.0
            continue

        # Evaluate at hi
        tau_hi = atmospheric_transmissivity(hi, ambient_temperature, relative_humidity)
        if model == "point_source":
            eff_dist_hi = math.sqrt(hi ** 2 + (center_height + flame_length / 3.0) ** 2)
            q_hi = thermal_radiation_point_source(
                total_heat_release, radiative_fraction, eff_dist_hi, tau_hi
            )
        else:
            F_hi = view_factor_jet_flame(flame_length, flame_width, hi, tilt_deg, center_height)
            q_hi = thermal_radiation_solid_flame(sep, F_hi, tau_hi)

        if q_hi > threshold:
            result[threshold] = max_search_distance
            continue

        # Binary search
        for _ in range(50):
            mid = (lo + hi) / 2.0
            tau_mid = atmospheric_transmissivity(mid, ambient_temperature, relative_humidity)

            if model == "point_source":
                eff_mid = math.sqrt(mid ** 2 + (center_height + flame_length / 3.0) ** 2)
                q_mid = thermal_radiation_point_source(
                    total_heat_release, radiative_fraction, eff_mid, tau_mid
                )
            else:
                F_mid = view_factor_jet_flame(
                    flame_length, flame_width, mid, tilt_deg, center_height
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

def calculate_jet_fire(
    input_data: JetFireInput,
    model: str = "point_source",
    min_distance: float = 1.0,
    max_distance: float = 200.0,
    n_points: int = 200,
) -> JetFireResult:
    """Calculate jet fire thermal radiation.

    Full pipeline:
        1. Determine mass flow rate
        2. Flame length (Kalghatgi / vertical jet)
        3. Flame width, center height, tilt
        4. Surface emissive power
        5. Radiation vs distance
        6. Distance to thresholds

    Args:
        input_data: JetFireInput with all parameters.
        model: "point_source" (default) or "solid_flame".
        min_distance: Min distance for curve [m].
        max_distance: Max distance for curve [m].
        n_points: Number of evaluation points.

    Returns:
        JetFireResult with all outputs.
    """
    messages = []
    D = input_data.orifice_diameter
    v = input_data.discharge_velocity
    mdot = input_data.mass_flow_rate or 0.0
    dhc = input_data.heat_of_combustion or 0.0
    chi_r = input_data.radiative_fraction

    if D <= 0.0 or mdot <= 0.0:
        return JetFireResult(
            flame_length=0.0,
            flame_width=0.0,
            flame_center_height=0.0,
            flame_tilt_deg=0.0,
            sep=0.0,
            total_heat_release=0.0,
            status_messages=["Orifice diameter and mass flow rate must be positive"],
        )

    # Estimate jet density if not provided
    rho_j = input_data.discharge_density
    if rho_j is None:
        # Approximate from gas law at ambient
        MW = MOLECULAR_WEIGHTS.get(input_data.substance.lower(), 0.029)
        rho_j = (P_ATM * MW) / (8.314 * input_data.ambient_temperature)

    # 1. Flame length
    L = flame_length_kalghatgi(
        D, v, rho_j, AIR_DENSITY_NTP,
        input_data.wind_speed, input_data.release_direction,
    )

    if L <= EPSILON:
        messages.append("Warning: Flame length near zero — check input parameters")

    # 2. Flame width
    W = flame_width_cone(L)

    # 3. Flame center height
    h_center = flame_center_height(
        L, input_data.release_direction, input_data.wind_speed, v
    )

    # 4. Flame tilt
    tilt = flame_tilt_jet(
        input_data.wind_speed, v, input_data.release_direction
    )

    # 5. Total heat release rate
    Q_dot = mdot * dhc  # [W]

    # 6. Surface emissive power
    sep = sep_jet_fire(Q_dot, L, W, chi_r)

    # 7. Radiation vs distance
    rad_vs_dist = thermal_radiation_vs_distance_jet(
        total_heat_release=Q_dot,
        radiative_fraction=chi_r,
        sep=sep,
        flame_length=L,
        flame_width=W,
        tilt_deg=tilt,
        center_height=h_center,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        min_distance=min_distance,
        max_distance=max_distance,
        n_points=n_points,
        model=model,
    )

    # 8. Distance to thresholds
    thresholds = distance_to_thresholds_jet(
        total_heat_release=Q_dot,
        radiative_fraction=chi_r,
        sep=sep,
        flame_length=L,
        flame_width=W,
        tilt_deg=tilt,
        center_height=h_center,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        thresholds=[37.5, 25.0, 12.5, 5.0, 4.0],
        model=model,
    )

    if sep > 400:
        messages.append("Info: Very high SEP — possible sonic/supersonic release")

    return JetFireResult(
        flame_length=round(L, 3),
        flame_width=round(W, 3),
        flame_center_height=round(h_center, 2),
        flame_tilt_deg=round(tilt, 1),
        sep=round(sep, 2),
        total_heat_release=round(Q_dot / 1e6, 3),  # MW
        thermal_radiation_vs_distance=rad_vs_dist,
        distance_to_thresholds=thresholds,
        model_used=model,
        status_messages=messages,
    )
