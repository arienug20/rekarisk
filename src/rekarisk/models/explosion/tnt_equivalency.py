"""
Rekarisk Explosion — TNT Equivalency Method.

Implements the TNT equivalency method for vapor cloud explosions using
Kingery-Bulmash blast parameters for hemispherical surface bursts.

Theory
------
The TNT equivalency method expresses the explosive energy of a vapor
cloud in terms of an equivalent mass of TNT:

    W_TNT = η · M_flammable · ΔHc / ΔHc_TNT

where:
    η          = explosion efficiency (0.01–0.10, typically 0.04)
    M_flammable = mass of flammable material released [kg]
    ΔHc        = heat of combustion of the material [J/kg]
    ΔHc_TNT    = heat of detonation of TNT = 4.68 MJ/kg

Blast parameters (overpressure, impulse, duration) are then obtained
from the scaled distance:

    Z = R / W_TNT^(1/3)    [m/kg^(1/3)]

using polynomial fits to the Kingery-Bulmash curves.

References
----------
- Kingery, C.N., Bulmash, G. (1984). "Airblast Parameters from TNT
  Spherical Air Burst and Hemispherical Surface Burst."
  ARBRL-TR-02555, US Army Ballistic Research Laboratory.
- UFC 3-340-02 (2008). Structures to Resist the Effects of
  Accidental Explosions, US DoD.
- CCPS (1994). Guidelines for Evaluating the Characteristics of
  Vapor Cloud Explosions, Flash Fires, and BLEVEs. AIChE/CCPS.
- TNO Yellow Book (2005). CPR 14E, Chapter 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rekarisk.core.constants import P_ATM, TNT_HEAT_OF_DETONATION, TNT_YIELD_FACTOR

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Standard overpressure damage thresholds in psi and kPa
DAMAGE_THRESHOLDS = {
    1.0: "Glass breakage (minor)",
    3.0: "Glass breakage (50%)",
    5.0: "Glass breakage (95%)",
    8.0: "Minor structural damage",
    10.0: "Steel frame distortion",
}

PSI_TO_KPA = 6.89475729


# ══════════════════════════════════════════════════════════════════════════════
# Kingery-Bulmash Polynomial Coefficients
# ══════════════════════════════════════════════════════════════════════════════
#
# Hemispherical surface burst (reflected into ground, doubling TNT weight).
# Coefficients are for the polynomial:
#
#     ln(Y) = a₀ + a₁·ln(Z) + a₂·ln(Z)² + a₃·ln(Z)³ + a₄·ln(Z)⁴ + ...
#
# where:
#   Y = P_s [kPa]       for overpressure
#   Y = i_s/W^(1/3)     for scaled impulse [kPa·ms/kg^(1/3)]
#   Y = t_d/W^(1/3)     for scaled duration [ms/kg^(1/3)]
#
# The published coefficients are typically for log10 basis. We convert
# to natural log (ln) for consistency — the KB fits here use ln basis.
#
# Valid range: Z ∈ [0.05, 40] m/kg^(1/3)
# ──────────────────────────────────────────────────────────────────────────────

# Overpressure polynomial coefficients — ln(P_s) vs ln(Z)
# P_s in kPa, Z in m/kg^(1/3)
# 6 regions covering Z = 0.05 to 40

KB_REGIONS = [
    # (Z_min, Z_max) — range of validity
    (0.05, 0.30),
    (0.30, 0.74),
    (0.74, 1.50),
    (1.50, 5.00),
    (5.00, 15.0),
    (15.0, 40.0),
]

# Coefficients: [ln(Z)⁰, ln(Z)¹, ln(Z)², ln(Z)³] — ln basis
# Derived from standard log10 KB polynomials converted to ln
# These produce P_s in kPa for hemispherical surface burst TNT
KB_OVERPRESSURE_COEFFS = [
    # Region 0: Z ∈ [0.05, 0.30)
    #   Very near-field — extremely high pressures
    {
        "a": [7.7675, -1.5084, -0.2096, 0.0129],
    },
    # Region 1: Z ∈ [0.30, 0.74)
    #   Near-field — high pressures
    {
        "a": [6.9041, -1.9670, 0.0079, 0.0000],
    },
    # Region 2: Z ∈ [0.74, 1.50)
    #   Transition region
    {
        "a": [5.3745, -1.1404, 0.1071, 0.0000],
    },
    # Region 3: Z ∈ [1.50, 5.00)
    #   Mid-field
    {
        "a": [4.3075, -1.0011, -0.0056, 0.0000],
    },
    # Region 4: Z ∈ [5.00, 15.0)
    #   Far-field
    {
        "a": [2.7805, -0.7791, -0.0982, 0.0000],
    },
    # Region 5: Z ∈ [15.0, 40.0)
    #   Very far-field — acoustic limit approached
    {
        "a": [0.1936, -0.0694, -0.1654, 0.0000],
    },
]

# Impulse polynomial coefficients — ln(i_s/W^(1/3)) vs ln(Z)
# i_s/W^(1/3) in kPa·ms/kg^(1/3), Z in m/kg^(1/3)
KB_IMPULSE_COEFFS = [
    # Region 0: Z ∈ [0.05, 0.30)
    {
        "a": [7.8743, 0.0938, -0.0368, 0.0046],
    },
    # Region 1: Z ∈ [0.30, 0.74)
    {
        "a": [7.3662, 0.5245, -0.0320, 0.0004],
    },
    # Region 2: Z ∈ [0.74, 1.50)
    {
        "a": [6.9534, 0.6409, -0.0307, 0.0000],
    },
    # Region 3: Z ∈ [1.50, 5.00)
    {
        "a": [6.6280, 0.7633, -0.0363, 0.0000],
    },
    # Region 4: Z ∈ [5.00, 15.0)
    {
        "a": [5.7680, 0.6266, -0.0155, 0.0000],
    },
    # Region 5: Z ∈ [15.0, 40.0)
    {
        "a": [4.8457, 0.5208, -0.0107, 0.0000],
    },
]

# Positive phase duration — ln(t_d/W^(1/3)) vs ln(Z)
# t_d/W^(1/3) in ms/kg^(1/3)
KB_DURATION_COEFFS = [
    # Region 0: Z ∈ [0.05, 0.30)
    {
        "a": [-0.2731, 0.2309, 0.0097, -0.0018],
    },
    # Region 1: Z ∈ [0.30, 0.74)
    {
        "a": [-0.1609, 0.2050, 0.0018, -0.0010],
    },
    # Region 2: Z ∈ [0.74, 1.50)
    {
        "a": [0.0884, 0.1689, -0.0025, -0.0008],
    },
    # Region 3: Z ∈ [1.50, 5.00)
    {
        "a": [0.4104, 0.2561, -0.0142, 0.0000],
    },
    # Region 4: Z ∈ [5.00, 15.0)
    {
        "a": [1.0225, 0.2180, -0.0135, 0.0000],
    },
    # Region 5: Z ∈ [15.0, 40.0)
    {
        "a": [1.4823, 0.1705, -0.0085, 0.0000],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Substance Heat of Combustion Database (ΔHc in J/kg)
# ══════════════════════════════════════════════════════════════════════════════

# Heat of combustion data for common flammable substances
# Used when substance DB lookup is unavailable
HEAT_OF_COMBUSTION = {
    "methane": 55.5e6,
    "ethane": 51.9e6,
    "propane": 50.35e6,
    "n-butane": 49.5e6,
    "isobutane": 49.3e6,
    "n-pentane": 49.0e6,
    "n-hexane": 48.4e6,
    "n-heptane": 48.1e6,
    "n-octane": 47.9e6,
    "ethylene": 50.3e6,
    "propylene": 48.9e6,
    "butylene": 48.2e6,
    "butadiene": 46.9e6,
    "acetylene": 49.9e6,
    "benzene": 42.0e6,
    "toluene": 42.5e6,
    "xylene": 43.1e6,
    "methanol": 22.7e6,
    "ethanol": 29.7e6,
    "acetone": 30.8e6,
    "hydrogen": 141.8e6,
    "ammonia": 22.5e6,
    "carbon_monoxide": 10.1e6,
    "hydrogen_sulfide": 16.5e6,
    "gasoline": 46.4e6,
    "kerosene": 46.2e6,
    "diesel": 45.5e6,
    "jet_fuel": 46.2e6,
    "lpg": 49.6e6,
    "lng": 55.0e6,
    "natural_gas": 50.0e6,
    "crude_oil": 44.0e6,
    "ethylene_oxide": 31.5e6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Input / Output Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TNTInput:
    """Input parameters for TNT equivalency calculation.

    Attributes:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion of material [J/kg].
            If None, will attempt lookup from substance database.
        explosion_efficiency: TNT equivalency efficiency factor η [-].
            Default 0.04 (typical for unconfined VCE). Range 0.01–0.10.
        distances: Array of distances R to evaluate [m].
            If None, auto-generates a logarithmic range.
        substance_name: Optional substance name for ΔHc lookup.
        ambient_pressure: Ambient pressure P_0 [Pa], default 101325.
    """
    mass_flammable: float
    heat_of_combustion: Optional[float] = None
    explosion_efficiency: float = 0.04
    distances: Optional[np.ndarray] = None
    substance_name: Optional[str] = None
    ambient_pressure: float = P_ATM

    def __post_init__(self):
        if self.explosion_efficiency <= 0 or self.explosion_efficiency > 1:
            raise ValueError(
                f"explosion_efficiency must be in (0, 1], got {self.explosion_efficiency}"
            )
        if self.mass_flammable <= 0:
            raise ValueError(
                f"mass_flammable must be > 0, got {self.mass_flammable}"
            )

        # Auto-resolve heat of combustion
        if self.heat_of_combustion is None:
            if self.substance_name:
                self.heat_of_combustion = _lookup_delta_hc(self.substance_name)
            if self.heat_of_combustion is None:
                raise ValueError(
                    "heat_of_combustion not provided and "
                    "substance_name lookup failed"
                )
        if self.heat_of_combustion <= 0:
            raise ValueError(
                f"heat_of_combustion must be > 0, got {self.heat_of_combustion}"
            )

        # Auto-generate distances if not provided
        if self.distances is None:
            tnt_eq = self.tnt_equivalent_mass
            r_min = max(5.0, 0.5 * tnt_eq ** (1 / 3))
            r_max = max(100.0, 50.0 * tnt_eq ** (1 / 3))
            self.distances = np.logspace(
                math.log10(r_min), math.log10(r_max), 100
            )

    @property
    def tnt_equivalent_mass(self) -> float:
        """Calculate TNT equivalent mass [kg]."""
        return (
            self.explosion_efficiency
            * self.mass_flammable
            * self.heat_of_combustion
            / TNT_HEAT_OF_DETONATION
        )

    @property
    def tnt_equivalent_energy(self) -> float:
        """Calculate TNT equivalent energy [J]."""
        return self.tnt_equivalent_mass * TNT_HEAT_OF_DETONATION

    def scaled_distance(self, r: float) -> float:
        """Compute scaled distance Z = R / W^(1/3)."""
        w_cube_root = self.tnt_equivalent_mass ** (1.0 / 3.0)
        return r / w_cube_root if w_cube_root > 0 else float("inf")


@dataclass
class ExplosionResult:
    """Unified explosion result containing blast parameters vs distance.

    Attributes:
        model_name: Name of the model used ('TNT', 'TNO', 'BST').
        tnt_equivalent_mass: Equivalent TNT mass [kg] (if applicable).
        energy: Available explosion energy [J].
        distances: Array of distances R evaluated [m].
        overpressure: Array of peak side-on overpressure P_s [kPa].
        impulse: Array of scaled impulse i_s [kPa·ms].
        positive_phase_duration: Array of positive phase duration t_d [ms].
        distance_to_thresholds: Dict mapping threshold (psi) → distance (m).
        model_params: Additional model-specific parameters.
    """
    model_name: str
    tnt_equivalent_mass: float
    energy: float
    distances: np.ndarray
    overpressure: np.ndarray
    impulse: np.ndarray
    positive_phase_duration: np.ndarray
    distance_to_thresholds: Dict[float, float] = field(default_factory=dict)
    model_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Ensure arrays are numpy
        self.distances = np.asarray(self.distances)
        self.overpressure = np.asarray(self.overpressure)
        self.impulse = np.asarray(self.impulse)
        self.positive_phase_duration = np.asarray(self.positive_phase_duration)

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "model_name": self.model_name,
            "tnt_equivalent_mass_kg": self.tnt_equivalent_mass,
            "energy_j": self.energy,
            "distances_m": self.distances.tolist(),
            "overpressure_kPa": self.overpressure.tolist(),
            "impulse_kPa_ms": self.impulse.tolist(),
            "positive_phase_duration_ms": self.positive_phase_duration.tolist(),
            "distance_to_thresholds": {
                f"{psi}psi": dist
                for psi, dist in self.distance_to_thresholds.items()
            },
            "model_params": self.model_params,
        }


# Alias for backward compatibility and unified interface
TNTResult = ExplosionResult


# ══════════════════════════════════════════════════════════════════════════════
# Kingery-Bulmash Blast Parameter Functions
# ══════════════════════════════════════════════════════════════════════════════

def _select_region(
    z: float, coeffs_list: List[Dict]
) -> Tuple[Dict, int]:
    """Select the appropriate polynomial region for a given scaled distance.

    Args:
        z: Scaled distance Z [m/kg^(1/3)].
        coeffs_list: List of coefficient dictionaries.

    Returns:
        Tuple of (coefficients dict, region index).
    """
    for i, (z_min, z_max) in enumerate(KB_REGIONS):
        if z_min <= z < z_max:
            return coeffs_list[i], i

    # Extrapolation handling
    if z < KB_REGIONS[0][0]:
        # Below minimum — use lowest region with clamping
        return coeffs_list[0], 0
    else:
        # Above maximum — use highest region
        return coeffs_list[-1], len(coeffs_list) - 1


def _evaluate_polynomial(ln_z: float, coeffs: Dict) -> float:
    """Evaluate a polynomial in ln(Z) space.

    Args:
        ln_z: Natural logarithm of scaled distance.
        coeffs: Dict with 'a' key containing coefficient list [a0, a1, a2, ...].

    Returns:
        ln(Y) where Y is the blast parameter.
    """
    a = coeffs["a"]
    ln_y = 0.0
    for i, a_i in enumerate(a):
        ln_y += a_i * (ln_z ** i)
    return ln_y


def kingery_bulmash_overpressure(z: float) -> float:
    """Calculate peak side-on overpressure from Kingery-Bulmash curves.

    Args:
        z: Scaled distance Z = R / W^(1/3) [m/kg^(1/3)].

    Returns:
        Peak side-on overpressure P_s [kPa].
    """
    if z < 0.01:
        return 1e6  # Essentially infinite pressure at point of detonation
    if z > 100:
        # Acoustic limit: P_s ≈ P_0 * (0.06 / Z) at very large Z
        # but returns to ambient — use approximate sound wave formula
        return P_ATM / 1000.0 * 0.05 / z

    coeffs, _ = _select_region(z, KB_OVERPRESSURE_COEFFS)
    ln_z = math.log(z)
    ln_ps = _evaluate_polynomial(ln_z, coeffs)
    ps = math.exp(ln_ps)

    # Clamp — overpressure can't exceed physical limits
    # and must be above ambient
    return max(0.001, min(1e6, ps))


def kingery_bulmash_impulse(z: float) -> float:
    """Calculate scaled impulse from Kingery-Bulmash curves.

    Args:
        z: Scaled distance Z = R / W^(1/3) [m/kg^(1/3)].

    Returns:
        Scaled impulse i_s/W^(1/3) [kPa·ms/kg^(1/3)].
    """
    if z < 0.01:
        return 1e6
    if z > 100:
        return 0.1

    coeffs, _ = _select_region(z, KB_IMPULSE_COEFFS)
    ln_z = math.log(z)
    ln_is = _evaluate_polynomial(ln_z, coeffs)
    return max(0.01, math.exp(ln_is))


def kingery_bulmash_duration(z: float) -> float:
    """Calculate scaled positive phase duration from Kingery-Bulmash curves.

    Args:
        z: Scaled distance Z = R / W^(1/3) [m/kg^(1/3)].

    Returns:
        Scaled positive phase duration t_d/W^(1/3) [ms/kg^(1/3)].
    """
    if z < 0.01:
        return 0.01
    if z > 100:
        return 20.0

    coeffs, _ = _select_region(z, KB_DURATION_COEFFS)
    ln_z = math.log(z)
    ln_td = _evaluate_polynomial(ln_z, coeffs)
    return max(0.001, math.exp(ln_td))


# ══════════════════════════════════════════════════════════════════════════════
# Core Calculation Functions
# ══════════════════════════════════════════════════════════════════════════════

def overpressure_at_distance(
    mass_flammable: float,
    heat_of_combustion: float,
    distance: float,
    explosion_efficiency: float = 0.04,
) -> float:
    """Calculate peak overpressure at a specific distance.

    Convenience function for single-point evaluation.

    Args:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion [J/kg].
        distance: Distance from explosion center [m].
        explosion_efficiency: TNT efficiency factor [-].

    Returns:
        Peak side-on overpressure P_s [kPa].
    """
    w_tnt = explosion_efficiency * mass_flammable * heat_of_combustion / TNT_HEAT_OF_DETONATION
    z = distance / (w_tnt ** (1.0 / 3.0)) if w_tnt > 0 else float("inf")
    return kingery_bulmash_overpressure(z)


def distance_to_overpressure(
    mass_flammable: float,
    heat_of_combustion: float,
    target_overpressure_kpa: float,
    explosion_efficiency: float = 0.04,
    r_min: float = 1.0,
    r_max: float = 10000.0,
) -> float:
    """Find the distance at which a target overpressure is achieved.

    Uses bisection search on the KB curve.

    Args:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion [J/kg].
        target_overpressure_kpa: Target overpressure [kPa].
        explosion_efficiency: TNT efficiency factor [-].
        r_min: Minimum search distance [m].
        r_max: Maximum search distance [m].

    Returns:
        Distance R [m] where P_s ≈ target_overpressure_kpa.
    """
    w_tnt = explosion_efficiency * mass_flammable * heat_of_combustion / TNT_HEAT_OF_DETONATION
    w_cuberoot = w_tnt ** (1.0 / 3.0)

    # Bisection search on Z
    def _f(z_guess):
        return kingery_bulmash_overpressure(z_guess) - target_overpressure_kpa

    z_low = r_min / w_cuberoot
    z_high = r_max / w_cuberoot

    # Ensure bracket contains root
    f_low = _f(z_low)
    f_high = _f(z_high)

    if f_low * f_high > 0:
        # Target not found in range — return min or max
        if abs(f_low) < abs(f_high):
            return r_min
        return r_max

    # Bisection
    for _ in range(100):
        z_mid = (z_low + z_high) / 2.0
        f_mid = _f(z_mid)
        if abs(f_mid) < 0.01:  # 0.01 kPa tolerance
            break
        if f_low * f_mid < 0:
            z_high = z_mid
            f_high = f_mid
        else:
            z_low = z_mid
            f_low = f_mid

    return z_mid * w_cuberoot


def distance_to_thresholds(
    mass_flammable: float,
    heat_of_combustion: float,
    explosion_efficiency: float = 0.04,
    thresholds_psi: Optional[List[float]] = None,
) -> Dict[float, float]:
    """Calculate distances to common overpressure damage thresholds.

    Args:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion [J/kg].
        explosion_efficiency: TNT efficiency factor [-].
        thresholds_psi: List of thresholds in psi. Default: [1, 3, 5, 8, 10].

    Returns:
        Dict mapping threshold (psi) → distance (m).
    """
    if thresholds_psi is None:
        thresholds_psi = [1.0, 3.0, 5.0, 8.0, 10.0]

    results: Dict[float, float] = {}
    for psi in thresholds_psi:
        target_kpa = psi * PSI_TO_KPA
        d = distance_to_overpressure(
            mass_flammable, heat_of_combustion, target_kpa, explosion_efficiency
        )
        results[psi] = d

    return results


def calculate_tnt_equivalency(input_data: TNTInput) -> ExplosionResult:
    """Perform full TNT equivalency calculation for an array of distances.

    Computes overpressure, impulse, positive phase duration, and
    distance-to-threshold for each distance in the input.

    Args:
        input_data: TNTInput with mass, heat of combustion, efficiency,
                    and distances.

    Returns:
        ExplosionResult with blast parameters vs distance.
    """
    inp = input_data
    w_tnt = inp.tnt_equivalent_mass
    w_cuberoot = w_tnt ** (1.0 / 3.0)
    energy = inp.tnt_equivalent_energy
    distances = np.asarray(inp.distances, dtype=np.float64)

    n = len(distances)
    overpressure = np.zeros(n)
    impulse_scaled = np.zeros(n)
    duration_scaled = np.zeros(n)

    for i, r in enumerate(distances):
        z = r / w_cuberoot if w_cuberoot > 0 else float("inf")
        overpressure[i] = kingery_bulmash_overpressure(z)
        impulse_scaled[i] = kingery_bulmash_impulse(z)
        duration_scaled[i] = kingery_bulmash_duration(z)

    # Convert scaled parameters to absolute values
    impulse = impulse_scaled * w_cuberoot  # [kPa·ms]
    positive_duration = duration_scaled * w_cuberoot  # [ms]

    # Calculate distance to standard thresholds
    thresholds = distance_to_thresholds(
        inp.mass_flammable,
        inp.heat_of_combustion,
        inp.explosion_efficiency,
    )

    return ExplosionResult(
        model_name="TNT Equivalency (Kingery-Bulmash)",
        tnt_equivalent_mass=w_tnt,
        energy=energy,
        distances=distances,
        overpressure=overpressure,
        impulse=impulse,
        positive_phase_duration=positive_duration,
        distance_to_thresholds=thresholds,
        model_params={
            "mass_flammable_kg": inp.mass_flammable,
            "heat_of_combustion_J_per_kg": inp.heat_of_combustion,
            "explosion_efficiency": inp.explosion_efficiency,
            "heat_of_detonation_TNT_J_per_kg": TNT_HEAT_OF_DETONATION,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Substance Heat of Combustion Lookup
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_delta_hc(name: str) -> Optional[float]:
    """Look up heat of combustion for a named substance.

    Checks built-in table first, then tries rekarisk substance database.

    Args:
        name: Substance name (case-insensitive).

    Returns:
        Heat of combustion [J/kg], or None if not found.
    """
    name_lower = name.lower().strip().replace(" ", "_")

    # Check built-in table
    if name_lower in HEAT_OF_COMBUSTION:
        return HEAT_OF_COMBUSTION[name_lower]

    # Try substance database
    try:
        from rekarisk.core.substance import Substance
        from rekarisk.core.substance_db import SUBSTANCE_DB

        # Direct lookup
        for key, sub in SUBSTANCE_DB.items():
            if key.lower() == name_lower:
                if hasattr(sub, "heat_of_combustion") and sub.heat_of_combustion:
                    return sub.heat_of_combustion
                break

        # Fuzzy name match
        for key, sub in SUBSTANCE_DB.items():
            if hasattr(sub, "name") and sub.name and sub.name.lower() == name_lower:
                if hasattr(sub, "heat_of_combustion") and sub.heat_of_combustion:
                    return sub.heat_of_combustion
    except ImportError:
        pass

    return None
