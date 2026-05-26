"""
Rekarisk Explosion — TNO Multi-Energy Method.

Implements the TNO Multi-Energy method for vapor cloud explosion blast
prediction. Unlike TNT equivalency, the Multi-Energy method accounts
for the effects of confinement and congestion on blast severity.

Theory
------
The method recognizes that only the confined/congested portion of a
flammable cloud contributes significantly to blast generation. The
blast is characterized by a "strength" index (1–10) selected based on:

    - Degree of confinement (none, 1D, 2D, 3D)
    - Level of congestion (low, medium, high)
    - Obstacle density and blockage ratio

Blast parameters are derived from scaled distance:

    R_s = R / (E / P_0)^(1/3)

where E is the available combustion energy [J] and P_0 is ambient
pressure [Pa]. Overpressure and impulse are then read from
strength-specific blast curves.

Confinement Classes:
    - 'none': No confinement (open field)
    - '1D': 1D confinement (pipe rack, narrow alley)
    - '2D': 2D confinement (platforms, multi-story buildings)
    - '3D': 3D confinement (enclosed spaces, vessels)

Congestion Levels:
    - 'low': Sparse equipment, few obstacles
    - 'medium': Typical process plant with pipe racks
    - 'high': Densely packed equipment, high blockage ratio

Blast Strength Guide (Auto-selection):
    Strength 1–3:  Open, no confinement, low congestion
    Strength 4–6:  Partially confined, medium congestion
    Strength 7–8:  Confined, high congestion
    Strength 9–10: Very confined, high congestion, parallel planes

References
----------
- van den Berg, A.C. (1985). "The Multi-Energy Method — A Framework
  for Vapor Cloud Explosion Blast Prediction." J. Hazardous Materials,
  12, 1–10.
- TNO Yellow Book (2005). CPR 14E, Chapter 7.
- CCPS (1994). Guidelines for Evaluating the Characteristics of
  Vapor Cloud Explosions, Flash Fires, and BLEVEs.
- Mercx, W.P.M. et al. (2000). Developments in Vapour Cloud Explosion
  Blast Modeling. J. Hazardous Materials, 71, 301–319.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rekarisk.core.constants import P_ATM, TNT_HEAT_OF_DETONATION

# Reuse unified result
from .tnt_equivalency import (
    ExplosionResult,
    PSI_TO_KPA,
    DAMAGE_THRESHOLDS,
    HEAT_OF_COMBUSTION,
)

# ──────────────────────────────────────────────────────────────────────────────
# Confinement and Congestion Assessment
# ──────────────────────────────────────────────────────────────────────────────

# Blast strength auto-selection matrix
# Rows = confinement class, Cols = congestion level
# Value = recommended blast strength (1–10)
BLAST_STRENGTH_MATRIX = {
    "none": {"low": 1, "medium": 2, "high": 3},
    "1D": {"low": 3, "medium": 5, "high": 7},
    "2D": {"low": 5, "medium": 7, "high": 9},
    "3D": {"low": 7, "medium": 9, "high": 10},
}

# Human-readable descriptions
BLAST_STRENGTH_DESCRIPTIONS = {
    1: "Very weak — unconfined, negligible congestion",
    2: "Weak — unconfined, light congestion",
    3: "Light — unconfined or simple 1D confinement",
    4: "Moderate-low — some confinement, light congestion",
    5: "Moderate — 1D/2D confinement, medium congestion",
    6: "Moderate-high — confined with moderate congestion",
    7: "Strong — confined, high congestion",
    8: "Very strong — 2D confinement, high congestion",
    9: "Severe — 3D confinement, high congestion",
    10: "Detonation-level — maximum confinement and congestion",
}

# Confinement class descriptions
CONFINEMENT_DESCRIPTIONS = {
    "none": "No confinement — open field or large open space",
    "1D": "1D confinement — pipe rack, narrow alley, corridor",
    "2D": "2D confinement — platform, multi-story structure, deck",
    "3D": "3D confinement — enclosed space, vessel, compressor house",
}

# Congestion level descriptions
CONGESTION_DESCRIPTIONS = {
    "low": "Sparse equipment, few obstacles, open layout",
    "medium": "Typical process plant, pipe racks, some equipment",
    "high": "Densely packed equipment, high blockage ratio (>30%)",
}


# ══════════════════════════════════════════════════════════════════════════════
# TNO Blast Curves — Scaled Overpressure vs Scaled Distance
# ══════════════════════════════════════════════════════════════════════════════
#
# The TNO blast curves give P_s/P_0 as a function of scaled distance
# R_s = R / (E / P_0)^(1/3)  where:
#   - R is distance from explosion center [m]
#   - E is combustion energy [J]
#   - P_0 is ambient pressure [Pa]
#
# Each strength (1–10) has its own curve. Published data is digitized
# from the TNO Yellow Book, Chapter 7, Figures 7.4A–7.4C.
#
# We encode each curve as discrete (R_s, P_s/P_0) data pairs with
# linear interpolation in log-log space.
# ──────────────────────────────────────────────────────────────────────────────

# TNO blast curve data: (R_s, P_s/P_0) for each strength level
# R_s = scaled distance [-], P_s/P_0 = dimensionless overpressure
# Data digitized from TNO Yellow Book and literature

def _tno_raw_curve(strength: int) -> np.ndarray:
    """Return raw (R_s, P_s/P_0) data for a given blast strength.

    Data digitized from TNO Yellow Book (CPR 14E), Chapter 7.
    Each row: [R_s, P_s/P_0].

    Args:
        strength: Blast strength 1–10.

    Returns:
        N×2 numpy array of (scaled_distance, dimensionless_overpressure).
    """
    # Curve data points for each strength
    # Format: [(R_s, P_s/P_0), ...]
    _curves = {
        1: np.array([
            [0.1, 0.85], [0.15, 0.45], [0.2, 0.22],
            [0.3, 0.09], [0.5, 0.035], [0.7, 0.020],
            [1.0, 0.012], [1.5, 0.007], [2.0, 0.005],
            [3.0, 0.0032], [5.0, 0.0018], [10.0, 0.0010],
        ]),
        2: np.array([
            [0.1, 1.3], [0.15, 0.65], [0.2, 0.35],
            [0.3, 0.14], [0.5, 0.052], [0.7, 0.029],
            [1.0, 0.017], [1.5, 0.010], [2.0, 0.0065],
            [3.0, 0.0042], [5.0, 0.0023], [10.0, 0.0013],
        ]),
        3: np.array([
            [0.1, 2.0], [0.15, 1.0], [0.2, 0.55],
            [0.3, 0.22], [0.5, 0.078], [0.7, 0.042],
            [1.0, 0.024], [1.5, 0.013], [2.0, 0.0087],
            [3.0, 0.0053], [5.0, 0.0029], [10.0, 0.0016],
        ]),
        4: np.array([
            [0.1, 3.0], [0.15, 1.5], [0.2, 0.8],
            [0.3, 0.32], [0.5, 0.11], [0.7, 0.058],
            [1.0, 0.032], [1.5, 0.017], [2.0, 0.011],
            [3.0, 0.0068], [5.0, 0.0036], [10.0, 0.0020],
        ]),
        5: np.array([
            [0.1, 5.0], [0.15, 2.5], [0.2, 1.3],
            [0.3, 0.48], [0.5, 0.16], [0.7, 0.082],
            [1.0, 0.044], [1.5, 0.023], [2.0, 0.015],
            [3.0, 0.0088], [5.0, 0.0046], [10.0, 0.0025],
        ]),
        6: np.array([
            [0.1, 8.0], [0.15, 4.0], [0.2, 2.0],
            [0.3, 0.72], [0.5, 0.23], [0.7, 0.115],
            [1.0, 0.060], [1.5, 0.031], [2.0, 0.019],
            [3.0, 0.011], [5.0, 0.0059], [10.0, 0.0031],
        ]),
        7: np.array([
            [0.1, 12.0], [0.15, 5.5], [0.2, 2.8],
            [0.3, 1.0], [0.5, 0.31], [0.7, 0.155],
            [1.0, 0.080], [1.5, 0.041], [2.0, 0.025],
            [3.0, 0.015], [5.0, 0.0075], [10.0, 0.0039],
        ]),
        8: np.array([
            [0.1, 18.0], [0.15, 8.0], [0.2, 4.0],
            [0.3, 1.4], [0.5, 0.42], [0.7, 0.205],
            [1.0, 0.105], [1.5, 0.053], [2.0, 0.032],
            [3.0, 0.019], [5.0, 0.0096], [10.0, 0.0048],
        ]),
        9: np.array([
            [0.1, 25.0], [0.15, 11.0], [0.2, 5.5],
            [0.3, 1.9], [0.5, 0.55], [0.7, 0.27],
            [1.0, 0.135], [1.5, 0.068], [2.0, 0.040],
            [3.0, 0.023], [5.0, 0.012], [10.0, 0.0059],
        ]),
        10: np.array([
            [0.1, 35.0], [0.15, 15.0], [0.2, 7.5],
            [0.3, 2.5], [0.5, 0.72], [0.7, 0.34],
            [1.0, 0.170], [1.5, 0.084], [2.0, 0.050],
            [3.0, 0.028], [5.0, 0.014], [10.0, 0.0070],
        ]),
    }
    return _curves.get(strength, _curves[5])  # default to strength 5


# TNO scaled impulse curves: i_s / (E^(1/3) · P_0^(2/3) · c_0^(-1))
# where c_0 = speed of sound in air ≈ 340 m/s
# Data: (R_s, dimensionless_scaled_impulse)
def _tno_impulse_raw_curve(strength: int) -> np.ndarray:
    """Return raw impulse curve data for a given blast strength.

    Returns:
        N×2 array of (scaled_distance, dimensionless_impulse).
    """
    # Approximate impulse data based on TNO Yellow Book
    _impulse_curves = {
        1: np.array([
            [0.1, 0.18], [0.2, 0.080], [0.3, 0.048],
            [0.5, 0.022], [1.0, 0.0085], [2.0, 0.0038],
            [5.0, 0.0015], [10.0, 0.0008],
        ]),
        5: np.array([
            [0.1, 0.35], [0.2, 0.16], [0.3, 0.095],
            [0.5, 0.042], [1.0, 0.016], [2.0, 0.0070],
            [5.0, 0.0030], [10.0, 0.0015],
        ]),
        10: np.array([
            [0.1, 0.65], [0.2, 0.30], [0.3, 0.18],
            [0.5, 0.078], [1.0, 0.030], [2.0, 0.013],
            [5.0, 0.0055], [10.0, 0.0028],
        ]),
    }

    # Interpolate between available curves
    if strength in _impulse_curves:
        return _impulse_curves[strength]
    elif strength <= 1:
        return _impulse_curves[1]
    elif strength >= 10:
        return _impulse_curves[10]
    elif strength <= 5:
        # Linear interpolation between strengths 1 and 5 in log space
        frac = (strength - 1) / 4.0
        c1 = _impulse_curves[1]
        c5 = _impulse_curves[5]
        # Interpolate at matching Rs values
        rs_vals = c1[:, 0]
        i1 = np.interp(rs_vals, c1[:, 0], np.log(c1[:, 1]))
        i5 = np.interp(rs_vals, c5[:, 0], np.log(c5[:, 1]))
        i_interp = np.exp(i1 + frac * (i5 - i1))
        return np.column_stack([rs_vals, i_interp])
    else:
        frac = (strength - 5) / 5.0
        c5 = _impulse_curves[5]
        c10 = _impulse_curves[10]
        rs_vals = c5[:, 0]
        i5 = np.interp(rs_vals, c5[:, 0], np.log(c5[:, 1]))
        i10 = np.interp(rs_vals, c10[:, 0], np.log(c10[:, 1]))
        i_interp = np.exp(i5 + frac * (i10 - i5))
        return np.column_stack([rs_vals, i_interp])


def _interpolate_curve(
    rs: float, curve_data: np.ndarray
) -> float:
    """Interpolate a value from a TNO curve at a given scaled distance.

    Uses log-log linear interpolation.

    Args:
        rs: Scaled distance [-].
        curve_data: N×2 array of (R_s, value) data points.

    Returns:
        Interpolated value at rs.
    """
    if rs <= curve_data[0, 0]:
        return float(curve_data[0, 1])
    if rs >= curve_data[-1, 0]:
        return float(curve_data[-1, 1])

    log_rs = math.log(rs)
    log_r_vals = np.log(curve_data[:, 0])
    log_vals = np.log(curve_data[:, 1])
    log_val = float(np.interp(log_rs, log_r_vals, log_vals))
    return math.exp(log_val)


# ══════════════════════════════════════════════════════════════════════════════
# Blast Strength Selection
# ══════════════════════════════════════════════════════════════════════════════

def auto_blast_strength(
    confinement_class: str,
    congestion_level: str,
) -> int:
    """Automatically select TNO blast strength from confinement and congestion.

    Args:
        confinement_class: One of 'none', '1D', '2D', '3D'.
        congestion_level: One of 'low', 'medium', 'high'.

    Returns:
        Blast strength 1–10.
    """
    confinement = confinement_class.lower()
    congestion = congestion_level.lower()

    # Case-insensitive key lookup
    matrix_lower = {k.lower(): v for k, v in BLAST_STRENGTH_MATRIX.items()}
    if confinement not in matrix_lower:
        raise ValueError(
            f"Unknown confinement_class '{confinement_class}'. "
            f"Use one of: {list(BLAST_STRENGTH_MATRIX.keys())}"
        )
    if congestion not in {"low", "medium", "high"}:
        raise ValueError(
            f"Unknown congestion_level '{congestion_level}'. "
            f"Use one of: low, medium, high"
        )

    return matrix_lower[confinement][congestion]


def blast_strength_description(strength: int) -> str:
    """Get human-readable description for a blast strength level.

    Args:
        strength: Blast strength 1–10.

    Returns:
        Description string.
    """
    return BLAST_STRENGTH_DESCRIPTIONS.get(strength, f"Unknown strength {strength}")


# ══════════════════════════════════════════════════════════════════════════════
# TNO Calculation Functions
# ══════════════════════════════════════════════════════════════════════════════

def tno_overpressure(
    scaled_distance: float,
    blast_strength: int,
) -> float:
    """Calculate dimensionless overpressure P_s/P_0 from TNO curve.

    Args:
        scaled_distance: R_s = R / (E / P_0)^(1/3) [-].
        blast_strength: Blast strength 1–10.

    Returns:
        Dimensionless overpressure P_s/P_0 [-].
    """
    curve = _tno_raw_curve(blast_strength)
    return _interpolate_curve(scaled_distance, curve)


def tno_impulse(
    scaled_distance: float,
    blast_strength: int,
) -> float:
    """Calculate dimensionless impulse from TNO curve.

    Args:
        scaled_distance: R_s [-].
        blast_strength: Blast strength 1–10.

    Returns:
        Dimensionless impulse (i_s / (E^(1/3) · P_0^(2/3) · c_0^(-1))).
    """
    curve = _tno_impulse_raw_curve(blast_strength)
    return _interpolate_curve(scaled_distance, curve)


def tno_positive_duration(
    scaled_distance: float,
    blast_strength: int,
) -> float:
    """Estimate dimensionless positive phase duration.

    Approximated from impulse / (2 * overpressure).

    Args:
        scaled_distance: R_s [-].
        blast_strength: Blast strength 1–10.

    Returns:
        Dimensionless positive phase duration.
    """
    imp = tno_impulse(scaled_distance, blast_strength)
    op = tno_overpressure(scaled_distance, blast_strength)
    if op < 1e-10:
        return 0.0
    return imp / (2.0 * op) if op > 1e-6 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Input / Output Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TNOInput:
    """Input parameters for TNO Multi-Energy calculation.

    Attributes:
        mass_flammable: Mass of flammable material [kg].
        heat_of_combustion: Heat of combustion of material [J/kg].
            If None, will attempt lookup from substance_name.
        explosion_efficiency: Fraction of cloud energy contributing to blast.
            Default 1.0 (Multi-Energy already accounts for this via
            confinement/congestion assessment).
        confinement_class: 'none', '1D', '2D', or '3D'.
        congestion_level: 'low', 'medium', or 'high'.
        blast_strength: Manual override for blast strength (1–10).
            If None, auto-selects from confinement + congestion.
        distances: Array of distances R to evaluate [m].
            If None, auto-generates based on energy.
        substance_name: Optional substance name for ΔHc lookup.
        ambient_pressure: Ambient pressure P_0 [Pa], default 101325.
    """
    mass_flammable: float
    heat_of_combustion: Optional[float] = None
    explosion_efficiency: float = 1.0
    confinement_class: str = "1D"
    congestion_level: str = "medium"
    blast_strength: Optional[int] = None
    distances: Optional[np.ndarray] = None
    substance_name: Optional[str] = None
    ambient_pressure: float = P_ATM

    def __post_init__(self):
        if self.mass_flammable <= 0:
            raise ValueError(
                f"mass_flammable must be > 0, got {self.mass_flammable}"
            )

        # Resolve heat of combustion
        if self.heat_of_combustion is None:
            if self.substance_name:
                self.heat_of_combustion = _lookup_delta_hc_tno(
                    self.substance_name
                )
            if self.heat_of_combustion is None:
                raise ValueError(
                    "heat_of_combustion not provided and substance lookup failed"
                )

        # Auto-select blast strength
        if self.blast_strength is None:
            self.blast_strength = auto_blast_strength(
                self.confinement_class, self.congestion_level
            )
        if not (1 <= self.blast_strength <= 10):
            raise ValueError(
                f"blast_strength must be 1–10, got {self.blast_strength}"
            )

        # Normalize strings
        self.confinement_class = self.confinement_class.lower()
        self.congestion_level = self.congestion_level.lower()

        # Auto-generate distances
        if self.distances is None:
            energy = self.combustion_energy
            r_char = (energy / self.ambient_pressure) ** (1.0 / 3.0)
            r_min = max(1.0, 0.1 * r_char)
            r_max = max(50.0, 10.0 * r_char)
            self.distances = np.logspace(
                math.log10(r_min), math.log10(r_max), 100
            )

    @property
    def combustion_energy(self) -> float:
        """Total available combustion energy [J]."""
        return self.explosion_efficiency * self.mass_flammable * self.heat_of_combustion

    @property
    def characteristic_length(self) -> float:
        """Characteristic length scale (E / P_0)^(1/3) [m]."""
        return (self.combustion_energy / self.ambient_pressure) ** (1.0 / 3.0)

    def scaled_distance(self, r: float) -> float:
        """Compute scaled distance R_s = R / (E/P_0)^(1/3)."""
        char_len = self.characteristic_length
        return r / char_len if char_len > 0 else float("inf")


# Alias for unified result
TNOResult = ExplosionResult


# ══════════════════════════════════════════════════════════════════════════════
# Substance Lookup
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_delta_hc_tno(name: str) -> Optional[float]:
    """Look up heat of combustion for TNO method."""
    name_lower = name.lower().strip().replace(" ", "_")
    if name_lower in HEAT_OF_COMBUSTION:
        return HEAT_OF_COMBUSTION[name_lower]
    try:
        from rekarisk.core.substance_db import SUBSTANCE_DB
        for key, sub in SUBSTANCE_DB.items():
            if key.lower() == name_lower:
                if hasattr(sub, "heat_of_combustion") and sub.heat_of_combustion:
                    return sub.heat_of_combustion
    except ImportError:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_tno_multi_energy(input_data: TNOInput) -> ExplosionResult:
    """Perform full TNO Multi-Energy blast calculation.

    Computes overpressure, impulse, and distance-to-thresholds using
    the TNO Multi-Energy blast curves.

    Args:
        input_data: TNOInput with mass, confinement, congestion, distances.

    Returns:
        ExplosionResult with blast parameters vs distance.
    """
    inp = input_data
    energy = inp.combustion_energy
    char_len = inp.characteristic_length
    strength = inp.blast_strength
    distances = np.asarray(inp.distances, dtype=np.float64)
    p0 = inp.ambient_pressure
    c0 = 340.0  # speed of sound [m/s]

    n = len(distances)
    overpressure = np.zeros(n)
    impulse = np.zeros(n)
    duration = np.zeros(n)

    for i, r in enumerate(distances):
        rs = r / char_len if char_len > 0 else float("inf")

        # Dimensionless overpressure
        ps_over_p0 = tno_overpressure(rs, strength)
        overpressure[i] = ps_over_p0 * p0 / 1000.0  # convert to kPa

        # Dimensionless impulse → dimensional [kPa·ms]
        # i_dimensionless = i_s / (E^(1/3) * P_0^(2/3) / c_0)
        imp_dimless = tno_impulse(rs, strength)
        ref_impulse = (
            energy ** (1.0 / 3.0)
            * p0 ** (2.0 / 3.0)
            / c0
        )
        impulse[i] = imp_dimless * ref_impulse / 1000.0  # convert to kPa·ms

        # Duration [ms]
        td_dimless = tno_positive_duration(rs, strength)
        ref_time = char_len / c0 * 1000.0  # [ms]
        duration[i] = td_dimless * ref_time

    # Approximate TNT equivalent for comparison
    tnt_equiv = inp.mass_flammable * inp.heat_of_combustion / TNT_HEAT_OF_DETONATION

    # Calculate distance to thresholds using bisection
    thresholds: Dict[float, float] = {}
    for psi in [1.0, 3.0, 5.0, 8.0, 10.0]:
        target_kpa = psi * PSI_TO_KPA
        # Binary search for distance
        d = _find_distance_to_overpressure_tno(
            inp, target_kpa, r_min=1.0, r_max=50000.0
        )
        thresholds[psi] = d

    return ExplosionResult(
        model_name=f"TNO Multi-Energy (Strength {strength})",
        tnt_equivalent_mass=tnt_equiv,
        energy=energy,
        distances=distances,
        overpressure=overpressure,
        impulse=impulse,
        positive_phase_duration=duration,
        distance_to_thresholds=thresholds,
        model_params={
            "blast_strength": strength,
            "confinement_class": inp.confinement_class,
            "congestion_level": inp.congestion_level,
            "combustion_energy_J": energy,
            "characteristic_length_m": char_len,
            "mass_flammable_kg": inp.mass_flammable,
            "heat_of_combustion_J_per_kg": inp.heat_of_combustion,
            "ambient_pressure_Pa": p0,
            "blast_strength_description": blast_strength_description(strength),
        },
    )


def _find_distance_to_overpressure_tno(
    inp: TNOInput, target_kpa: float, r_min: float = 1.0, r_max: float = 50000.0
) -> float:
    """Bisection search for distance at target overpressure in TNO model.

    Args:
        inp: TNOInput with all parameters.
        target_kpa: Target overpressure [kPa].
        r_min, r_max: Search bounds [m].

    Returns:
        Distance [m] where P_s ≈ target_kpa.
    """
    char_len = inp.characteristic_length
    strength = inp.blast_strength
    p0 = inp.ambient_pressure

    def _overpressure_at_r(r_guess):
        rs = r_guess / char_len if char_len > 0 else float("inf")
        return tno_overpressure(rs, strength) * p0 / 1000.0

    # Check if target is achievable
    op_min = _overpressure_at_r(r_min)
    op_max = _overpressure_at_r(r_max)

    if target_kpa >= op_min:
        return r_min
    if target_kpa <= op_max:
        return r_max

    # Bisection
    r_low, r_high = r_min, r_max
    for _ in range(100):
        r_mid = (r_low + r_high) / 2.0
        op_mid = _overpressure_at_r(r_mid) - target_kpa
        if abs(op_mid) < 0.01:
            break
        if (_overpressure_at_r(r_low) - target_kpa) * op_mid < 0:
            r_high = r_mid
        else:
            r_low = r_mid

    return (r_low + r_high) / 2.0
