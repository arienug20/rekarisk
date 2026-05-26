"""
Rekarisk — BLEVE / Fireball Thermal Radiation Models.

Calculates thermal radiation from Boiling Liquid Expanding Vapor Explosions
(BLEVE) and fireballs. Implemented per HSE, CCPS, TNO Yellow Book, and
Roberts correlation for fireball dimensions.

References:
  - CCPS Guidelines for Consequence Analysis of Chemical Releases (1999)
  - TNO Yellow Book (CPR 14E), Chapter 5 — BLEVE
  - Roberts, A.F. (1981) — Thermal Radiation Hazards from Releases of LPG
    from Pressurised Storage, Fire Safety Journal, 4(3), 197-212
  - HSE (UK) — Failure Rate and Event Data for use within Risk Assessments
  - AIChE/CCPS (1994) — Guidelines for Evaluating the Characteristics of
    Vapor Cloud Explosions, Flash Fires, and BLEVEs
  - Prugh, R.W. (1991) — Quantitative Evaluation of BLEVE Hazards,
    J. Fire Protection Engineering, 3(1), 9-24
  - API RP 521 — Pressure-Relieving and Depressuring Systems
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...core.constants import G, T_0C, EPSILON, P_ATM


# ══════════════════════════════════════════════════════════════════════════════
# Substance Properties for BLEVE
# ══════════════════════════════════════════════════════════════════════════════

# Heats of combustion [J/kg]
BLEVE_HEATS_OF_COMBUSTION: Dict[str, float] = {
    "propane": 46.3e6,
    "butane": 45.7e6,
    "lpg": 46.0e6,
    "methane": 50.0e6,
    "lng": 50.0e6,
    "ethane": 47.5e6,
    "ethylene": 47.2e6,
    "propylene": 45.8e6,
    "ammonia": 18.6e6,
    "hydrogen": 120.0e6,
    "gasoline": 44.0e6,
    "hexane": 45.1e6,
    "default": 46.0e6,
}

# Stored energy for BLEVE energy release
# Energy stored in pressurized liquid: E = m · (h_liquid - h_vapor_at_atm)
# Approximate: E_stored ≈ m · cp · (T_storage - T_boil)
STORE_ENERGY_HEAT_CAPACITY: Dict[str, float] = {
    "propane": 2650.0,
    "butane": 2480.0,
    "lpg": 2550.0,
    "lng": 3600.0,
    "ethane": 3400.0,
    "default": 2500.0,
}

# Sound speed in vessel contents [m/s] — for fragment velocity
SOUND_SPEED_VESSEL: Dict[str, float] = {
    "propane": 220.0,
    "butane": 180.0,
    "lpg": 210.0,
    "default": 200.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Input/Output Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BLEVEInput:
    """Input parameters for BLEVE / fireball calculation.

    Attributes:
        vessel_mass: Mass of contents released [kg].
            This is the mass that participates in the fireball.
        substance: Substance name for property auto-lookup.
        heat_of_combustion: Lower heating value [J/kg]. Auto if None.
        radiative_fraction: Radiative fraction [-]. Default 0.30.
            Typical 0.25-0.40 for BLEVE/fireball (CCPS).
        sep_override: Override surface emissive power [kW/m²].
            If None, auto-calculated. Typical 200-350 kW/m².
        ambient_temperature: Ambient temperature [K]. Default 298.15.
        relative_humidity: Relative humidity [%]. Default 50.0.
        vessel_pressure: Vessel pressure at failure [Pa abs].
            For fragment throw estimation.
        vessel_temperature: Contents temperature at failure [K].
        vessel_volume: Vessel volume [m³]. For fragment analysis.
    """
    vessel_mass: float
    substance: str = "default"

    heat_of_combustion: Optional[float] = None
    radiative_fraction: float = 0.30
    sep_override: Optional[float] = None

    # Environmental
    ambient_temperature: float = 298.15
    relative_humidity: float = 50.0

    # For fragment analysis (optional)
    vessel_pressure: Optional[float] = None
    vessel_temperature: Optional[float] = None
    vessel_volume: Optional[float] = None

    def __post_init__(self):
        if self.heat_of_combustion is None:
            self.heat_of_combustion = BLEVE_HEATS_OF_COMBUSTION.get(
                self.substance.lower(), BLEVE_HEATS_OF_COMBUSTION["default"]
            )


@dataclass
class BLEVEResult:
    """Results from BLEVE / fireball calculation.

    Attributes:
        fireball_diameter: Maximum fireball diameter [m].
        fireball_duration: Fireball duration [s].
        center_height: Height of fireball center at max size [m].
        sep: Surface emissive power [kW/m²].
        total_radiative_energy: Total radiative energy released [J].
        thermal_radiation_vs_distance: (N,2) array — [distance_m, flux_kW_per_m2].
        distance_to_thresholds: Dict threshold [kW/m²] → distance [m].
        fragment_max_distance: Estimated maximum fragment throw [m] (if calc'd).
        status_messages: List of info/warning strings.
    """
    fireball_diameter: float
    fireball_duration: float
    center_height: float
    sep: float
    total_radiative_energy: float
    thermal_radiation_vs_distance: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    distance_to_thresholds: Dict[float, float] = field(default_factory=dict)
    fragment_max_distance: float = 0.0
    status_messages: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Fireball Diameter — Roberts Correlation
# ══════════════════════════════════════════════════════════════════════════════

def fireball_diameter_roberts(mass: float) -> float:
    """Calculate maximum fireball diameter using Roberts correlation.

    Roberts (1981):
        D_max = 5.8 · M^0.333

    where:
        D_max = maximum fireball diameter [m]
        M     = mass of fuel in fireball [kg]

    Alternative (CCPS, TNO):
        D_max = 6.48 · M^0.325   (for LPG/propane)
        D_max = 5.25 · M^0.314   (for other hydrocarbons)

    We use the widely-accepted Roberts value 5.8 per HSE/CCPS
    recommendations.

    Args:
        mass: Mass of fuel participating [kg].

    Returns:
        Maximum fireball diameter [m].
    """
    if mass <= 0:
        return 0.0

    return 5.8 * (mass ** (1.0 / 3.0))


# ══════════════════════════════════════════════════════════════════════════════
# Fireball Duration — Roberts Correlation
# ══════════════════════════════════════════════════════════════════════════════

def fireball_duration_roberts(mass: float) -> float:
    """Calculate fireball duration using Roberts correlation.

    Roberts (1981) — simplified form:
        t_d = 0.45 · M^0.333

    where:
        t_d = fireball duration [s]
        M   = mass [kg]

    More detailed form (TNO):
        t_d = 7.4 · 10^-3 · M^0.333 · (ΔHc / (cp · Tb))^0.333

    The simplified coefficient 0.45 is the product of the factors for
    typical LPG/propane. For other substances, adjust proportionally.

    Alternative (CCPS):
        t_d = 0.825 · M^0.26

    We provide the most common simplified Roberts form.

    Args:
        mass: Mass of fuel [kg].

    Returns:
        Fireball duration [s].
    """
    if mass <= 0:
        return 0.0

    return 0.45 * (mass ** (1.0 / 3.0))


# ══════════════════════════════════════════════════════════════════════════════
# Fireball Surface Emissive Power
# ══════════════════════════════════════════════════════════════════════════════

def fireball_sep(
    mass: float,
    heat_of_combustion: float,
    radiative_fraction: float,
    diameter: Optional[float] = None,
    duration: Optional[float] = None,
) -> float:
    """Calculate surface emissive power for BLEVE fireball.

    Method 1 (from energy balance):
        SEP = χ_r · ΔHc · M / (π · D² · t_d)

    Method 2 (CCPS fixed values):
        SEP = 200-350 kW/m²  (typical for BLEVE fireballs)

    Method 3 (HSE):
        SEP = χ_r · ΔHc / (π · D² · t_d) · M_fireball

    Where:
        χ_r  = radiative fraction
        ΔHc  = heat of combustion [J/kg]
        M    = mass [kg]
        D    = fireball diameter [m]
        t_d  = fireball duration [s]

    Args:
        mass: Mass of fuel [kg].
        heat_of_combustion: Lower heating value [J/kg].
        radiative_fraction: Radiative fraction [-].
        diameter: Fireball diameter [m]. Auto-calculated if None.
        duration: Fireball duration [s]. Auto-calculated if None.

    Returns:
        Surface emissive power [kW/m²].
    """
    if mass <= 0:
        return 0.0

    if diameter is None:
        diameter = fireball_diameter_roberts(mass)
    if duration is None:
        duration = fireball_duration_roberts(mass)

    if diameter <= EPSILON or duration <= EPSILON:
        return 0.0

    # Method 1: energy balance
    fireball_area = math.pi * diameter ** 2
    total_released = radiative_fraction * heat_of_combustion * mass
    sep_theoretical = total_released / (fireball_area * duration)  # [W/m²]
    sep_kw = sep_theoretical / 1000.0

    # Constrain to physically reasonable range (CCPS)
    # BLEVE fireball SEP ranges from 150 to 400 kW/m²
    sep_kw = min(max(sep_kw, 100.0), 450.0)

    return sep_kw


# ══════════════════════════════════════════════════════════════════════════════
# Fireball Center Height
# ══════════════════════════════════════════════════════════════════════════════

def fireball_center_height(
    mass: float,
    diameter: Optional[float] = None,
) -> float:
    """Calculate center height of fireball at maximum size.

    The fireball rises due to buoyancy. At maximum diameter, the center
    is approximately at height = D/2 to D above ground, depending on
    mass and geometry.

    HSE/TNO: H_center ≈ 0.75 · D_max for ground-level BLEVE.

    For elevated vessels: add vessel height.

    Args:
        mass: Mass of fuel [kg].
        diameter: Fireball diameter [m]. Auto-calculated if None.

    Returns:
        Fireball center height above ground [m].
    """
    if mass <= 0:
        return 0.0

    if diameter is None:
        diameter = fireball_diameter_roberts(mass)

    # At maximum diameter, fireball base may touch ground
    # Center at ~ D/2 for ground-level BLEVE
    # With buoyancy rise during growth, center goes to ~0.75·D
    return 0.75 * diameter


# ══════════════════════════════════════════════════════════════════════════════
# View Factor — Sphere
# ══════════════════════════════════════════════════════════════════════════════

def view_factor_sphere(
    radius: float,
    distance: float,
    height: float = 0.0,
) -> float:
    """View factor from a sphere to a small receiver at ground level.

    For a sphere of radius R at height H above ground, with receiver
    at ground level at horizontal distance x from the center projection:

        F = R² / (x² + H²)     for x² + H² ≥ R²
        F = 1                  for x² + H² < R²

    This is the point-receiver form from a Lambertian sphere.

    For a receiver at ground level looking up at the sphere:
        F = (R / d_center)² · cos(θ_receiver)

    where θ_receiver is the angle between the ground normal and the
    line connecting receiver to sphere center.

    Simplified (CCPS):
        F = R² / (R² + d_center²)

    Args:
        radius: Sphere radius [m].
        distance: Horizontal distance from center projection [m].
        height: Height of sphere center above ground [m].

    Returns:
        View factor [-] (0 to 1).
    """
    if radius <= EPSILON:
        return 0.0

    d_center = math.sqrt(distance ** 2 + height ** 2)

    if d_center <= radius:
        return 1.0

    # Lambertian sphere to planar receiver
    # F = (R/d)² · (H/d)   — for receiver facing upward
    # The cos(theta) factor accounts for the receiver's orientation
    cos_theta = height / max(d_center, EPSILON)

    # View factor from sphere surface to receiver element
    # (CCPS simplified form for distant receiver)
    F = (radius / d_center) ** 2 * cos_theta

    # Alternative: uniform spherical emission
    # F = R² / d_center²  (for point receiver far away)

    return min(max(F, 0.0), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Atmospheric Transmissivity
# ══════════════════════════════════════════════════════════════════════════════

def atmospheric_transmissivity(
    distance: float,
    ambient_temperature: float = 298.15,
    relative_humidity: float = 50.0,
) -> float:
    """Atmospheric transmissivity for thermal radiation.

    Args:
        distance: Path length [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].

    Returns:
        Transmissivity [-] (0 to 1).
    """
    if distance < EPSILON:
        return 1.0

    T_C = ambient_temperature - T_0C
    p_sat = 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))
    p_w = (relative_humidity / 100.0) * p_sat
    kappa = 2.02e-5 * (max(p_w, 1.0) ** 0.09)
    tau = math.exp(-kappa * distance)

    return min(max(tau, 0.01), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Fragment Throw Distance (Simplified)
# ══════════════════════════════════════════════════════════════════════════════

def fragment_throw_distance(
    vessel_pressure: float,
    vessel_volume: float,
    fragment_mass: float = 1.0,
    fragment_drag_coefficient: float = 0.8,
) -> float:
    """Estimate maximum fragment throw distance from BLEVE.

    Simplified ballistic model (CCPS, TNO):

    Energy released by vessel rupture:
        E = (P1 - P_atm) · V / (γ - 1)    for gas
        E ≈ (P1 · V) · ln(P1/P_atm)       for liquid

    Fragment initial velocity:
        v0 = sqrt(2 · E_fragment / m_fragment)

    Where E_fragment is the portion of total energy imparted to
    the fragment (~20% for small fragments, ~40% for large).

    Maximum throw at 45° launch:
        R_max = v0² / g

    Args:
        vessel_pressure: Vessel pressure at failure [Pa abs].
        vessel_volume: Vessel volume [m³].
        fragment_mass: Mass of a typical fragment [kg].
        fragment_drag_coefficient: Drag coefficient [-].

    Returns:
        Estimated maximum fragment throw distance [m].
    """
    if vessel_pressure <= P_ATM or vessel_volume <= 0:
        return 0.0

    # Stored energy (isentropic expansion approximation for gas)
    # γ ≈ 1.3 for hydrocarbon gases
    gamma = 1.3

    # Energy from compressed gas
    P_ratio = vessel_pressure / P_ATM
    E_total = (vessel_pressure * vessel_volume / (gamma - 1.0)) * (
        1.0 - P_ratio ** ((1.0 - gamma) / gamma)
    )

    # More conservative: use simplified stored energy
    E_total_alt = (vessel_pressure - P_ATM) * vessel_volume
    E_total = max(E_total, E_total_alt)

    # Fraction of energy imparted to one fragment
    # Assume single typical fragment gets 5-20% of total energy
    energy_fraction = 0.10

    if fragment_mass <= EPSILON:
        return 0.0

    v0_squared = 2.0 * energy_fraction * E_total / fragment_mass

    # Practical limit: fragment velocity < speed of sound in material
    max_v = min(math.sqrt(v0_squared), 300.0)  # [m/s] typical max

    # Simplified ballistic range at 45° with drag reduction
    range_no_drag = max_v ** 2 / G  # [m]

    # Apply drag reduction (simplified)
    # For fragments with drag, range is reduced
    range_with_drag = range_no_drag * max(0.3, 1.0 - 0.1 * fragment_drag_coefficient)

    return round(range_with_drag, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Thermal Radiation vs Distance
# ══════════════════════════════════════════════════════════════════════════════

def thermal_radiation_vs_distance_bleve(
    sep: float,
    fireball_radius: float,
    fireball_height: float,
    ambient_temperature: float,
    relative_humidity: float,
    min_distance: float = 1.0,
    max_distance: float = 500.0,
    n_points: int = 200,
) -> np.ndarray:
    """Compute thermal radiation vs distance for BLEVE fireball.

    Args:
        sep: Surface emissive power [kW/m²].
        fireball_radius: Fireball radius [m].
        fireball_height: Fireball center height [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        min_distance: Minimum receiver distance [m].
        max_distance: Maximum receiver distance [m].
        n_points: Number of evaluation points.

    Returns:
        (N, 2) array — [distance_m, flux_kW_per_m2].
    """
    distances = np.linspace(min_distance, max_distance, n_points)
    fluxes = np.zeros(n_points)

    for i, d in enumerate(distances):
        tau = atmospheric_transmissivity(d, ambient_temperature, relative_humidity)

        # View factor from fireball sphere to receiver at ground level
        F = view_factor_sphere(fireball_radius, d, fireball_height)

        fluxes[i] = tau * sep * F

    return np.column_stack((distances, fluxes))


# ══════════════════════════════════════════════════════════════════════════════
# Distance to Thresholds
# ══════════════════════════════════════════════════════════════════════════════

def distance_to_thresholds_bleve(
    sep: float,
    fireball_radius: float,
    fireball_height: float,
    ambient_temperature: float,
    relative_humidity: float,
    thresholds: Optional[List[float]] = None,
    max_search_distance: float = 1000.0,
) -> Dict[float, float]:
    """Find distances to thermal radiation thresholds for BLEVE.

    Args:
        sep: Surface emissive power [kW/m²].
        fireball_radius: Fireball radius [m].
        fireball_height: Fireball center height [m].
        ambient_temperature: Ambient temperature [K].
        relative_humidity: Relative humidity [%].
        thresholds: List of thresholds [kW/m²].
        max_search_distance: Max search distance [m].

    Returns:
        Dict threshold → distance [m].
    """
    if thresholds is None:
        thresholds = [37.5, 25.0, 12.5, 5.0, 4.0]

    result = {}

    for threshold in thresholds:
        lo = max(fireball_radius * 0.5, 0.1)
        hi = max_search_distance

        # Check lo
        tau_lo = atmospheric_transmissivity(lo, ambient_temperature, relative_humidity)
        F_lo = view_factor_sphere(fireball_radius, lo, fireball_height)
        q_lo = tau_lo * sep * F_lo

        if q_lo <= threshold:
            result[threshold] = 0.0
            continue

        # Check hi
        tau_hi = atmospheric_transmissivity(hi, ambient_temperature, relative_humidity)
        F_hi = view_factor_sphere(fireball_radius, hi, fireball_height)
        q_hi = tau_hi * sep * F_hi

        if q_hi > threshold:
            result[threshold] = max_search_distance
            continue

        # Binary search
        for _ in range(50):
            mid = (lo + hi) / 2.0
            tau_mid = atmospheric_transmissivity(mid, ambient_temperature, relative_humidity)
            F_mid = view_factor_sphere(fireball_radius, mid, fireball_height)
            q_mid = tau_mid * sep * F_mid

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

def calculate_bleve(
    input_data: BLEVEInput,
    min_distance: float = 1.0,
    max_distance: float = 500.0,
    n_points: int = 200,
) -> BLEVEResult:
    """Calculate BLEVE / fireball thermal radiation.

    Full pipeline:
        1. Fireball diameter (Roberts)
        2. Fireball duration (Roberts)
        3. Center height
        4. Surface emissive power (SEP)
        5. Thermal radiation vs distance
        6. Distance to thresholds
        7. Fragment throw distance (if vessel data provided)

    Args:
        input_data: BLEVEInput with vessel mass, substance, environment.
        min_distance: Min distance for curve [m].
        max_distance: Max distance for curve [m].
        n_points: Number of evaluation points.

    Returns:
        BLEVEResult with all outputs.
    """
    messages = []
    M = input_data.vessel_mass
    dhc = input_data.heat_of_combustion
    chi_r = input_data.radiative_fraction

    if M <= 0:
        return BLEVEResult(
            fireball_diameter=0.0,
            fireball_duration=0.0,
            center_height=0.0,
            sep=0.0,
            total_radiative_energy=0.0,
            status_messages=["Vessel mass must be positive"],
        )

    # 1. Fireball diameter
    D_fb = fireball_diameter_roberts(M)
    R_fb = D_fb / 2.0

    # 2. Fireball duration
    t_d = fireball_duration_roberts(M)

    # 3. Center height
    h_center = fireball_center_height(M, D_fb)

    # 4. Surface emissive power
    if input_data.sep_override is not None:
        sep = input_data.sep_override
    else:
        sep = fireball_sep(M, dhc, chi_r, D_fb, t_d)

    # 5. Total radiative energy
    E_rad_total = chi_r * dhc * M  # [J]

    # 6. Radiation vs distance
    rad_vs_dist = thermal_radiation_vs_distance_bleve(
        sep=sep,
        fireball_radius=R_fb,
        fireball_height=h_center,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        min_distance=min_distance,
        max_distance=max_distance,
        n_points=n_points,
    )

    # 7. Distance to thresholds
    thresholds = distance_to_thresholds_bleve(
        sep=sep,
        fireball_radius=R_fb,
        fireball_height=h_center,
        ambient_temperature=input_data.ambient_temperature,
        relative_humidity=input_data.relative_humidity,
        thresholds=[37.5, 25.0, 12.5, 5.0, 4.0],
        max_search_distance=max_distance * 2.0,
    )

    # 8. Fragment throw (if vessel data available)
    frag_dist = 0.0
    if (input_data.vessel_pressure is not None
            and input_data.vessel_volume is not None
            and input_data.vessel_pressure > P_ATM
            and input_data.vessel_volume > 0):
        frag_dist = fragment_throw_distance(
            vessel_pressure=input_data.vessel_pressure,
            vessel_volume=input_data.vessel_volume,
        )
        messages.append("Fragment throw distance estimated (simplified model)")

    # Validation
    if sep < 100:
        messages.append("Warning: SEP is very low for a BLEVE fireball (< 100 kW/m²)")
    if sep > 500:
        messages.append("Warning: SEP is unusually high (> 500 kW/m²)")
    if t_d > 60:
        messages.append(f"Warning: Fireball duration ({t_d:.1f}s) is unusually long")

    return BLEVEResult(
        fireball_diameter=round(D_fb, 2),
        fireball_duration=round(t_d, 2),
        center_height=round(h_center, 2),
        sep=round(sep, 1),
        total_radiative_energy=round(E_rad_total / 1e6, 3),  # MJ
        thermal_radiation_vs_distance=rad_vs_dist,
        distance_to_thresholds=thresholds,
        fragment_max_distance=frag_dist,
        status_messages=messages,
    )
