"""
Rekarisk Dispersion — Dense Gas Dispersion Model.

Implements a simplified dense gas dispersion model based on the SLAB
framework. Handles three distinct phases:

1. Gravity Spreading (slumping): The cloud spreads radially under gravity
   as a density current before significant dilution.
2. Air Entrainment: Gradual dilution via top surface and edge entrainment.
3. Transition to Passive: When the cloud density approaches ambient or
   the Richardson number falls below critical, transitions to passive
   Gaussian dispersion.

Triggered when: ρ_cloud / ρ_air > 1.1 (density ratio criterion)

References:
    - Ermak, D.L. (1990). User's Manual for SLAB. UCRL-MA-105607.
    - HEGADAS model (TNO Yellow Book, Chapter 4).
    - CCPS Guidelines for Consequence Analysis of Chemical Releases.
    - Britter, R.E. & McQuaid, J. (1988). Workbook on the Dispersion of
      Dense Gases. HSE Contract Research Report No. 17/1988.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from rekarisk.core.constants import G, P_ATM, T_0C
from rekarisk.meteorology.stability import (
    StabilityClass,
    TerrainType,
    sigma_y,
    sigma_z,
)

# If Gaussian plume is available, use it for transition calculations
try:
    from rekarisk.models.dispersion.gaussian_plume import (
        PlumeInput,
        concentration_at_point,
    )
    HAS_PLUME = True
except ImportError:
    HAS_PLUME = False


# ---------------------------------------------------------------------------
# Constants for Dense Gas Dispersion
# ---------------------------------------------------------------------------

# Top entrainment coefficient (Stretch & Britter)
ALPHA_TOP_DEFAULT = 0.1

# Edge entrainment coefficient multiplier
ALPHA_EDGE_BASE = 0.5

# Critical Richardson number for transition to passive
RI_CRITICAL = 0.5

# Critical density ratio for transition (ρ/ρ_air - 1)
DELTA_RHO_CRITICAL = 0.01  # 1% density excess

# Maximum time step multiplier for integration
DT_MULTIPLIER = 0.1  # dt ≤ 0.1 * spreading time scale

# Gravity spreading constant (similarity theory)
GRAVITY_SPREADING_C = 1.5

# Minimum Richardson number
RI_MIN = 1e-6

# Cloud aspect ratio at release (height / radius)
ASPECT_RATIO_INITIAL = 1.0

# ReleaseType
ReleaseType = Literal["instantaneous", "continuous"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DenseGasInput:
    """Input parameters for dense gas dispersion calculation.

    Attributes:
        source_mass: Total mass released [kg] (instantaneous).
        source_rate: Mass flow rate [kg/s] (continuous).
        release_type: 'instantaneous' or 'continuous'.
        release_duration: Duration of release [s] (continuous only).
        cloud_density: Initial cloud density ρ_c [kg/m³].
        air_density: Ambient air density ρ_a [kg/m³].
        wind_speed: Wind speed at 10 m [m/s].
        stability_class: PG stability class.
        release_height: Height of release center [m] (default 0).
        terrain_type: 'rural' or 'urban'.
        temperature_cloud: Cloud temperature [K].
        temperature_ambient: Ambient temperature [K].
        cloud_radius_initial: Initial cloud radius [m]. If 0, estimated
            from release area or default aspect ratio.
        cloud_height_initial: Initial cloud height [m]. If 0, estimated.
        time_end: End time for simulation [s].
        time_step_max: Maximum time step [s] (0 for auto).
        enable_transition: Whether to transition to Gaussian when passive.
        molecular_weight: MW [g/mol].
    """

    source_mass: float = 0.0  # [kg], instantaneous
    source_rate: float = 0.0  # [kg/s], continuous
    release_type: ReleaseType = "instantaneous"
    release_duration: float = 0.0  # [s]
    cloud_density: float = 2.0  # [kg/m³]
    air_density: float = 1.2  # [kg/m³]
    wind_speed: float = 3.0  # [m/s]
    stability_class: StabilityClass = "D"
    release_height: float = 0.0  # [m]
    terrain_type: TerrainType = "rural"
    temperature_cloud: float = 250.0  # [K]
    temperature_ambient: float = 298.15  # [K]
    cloud_radius_initial: float = 0.0  # [m] (0 = auto)
    cloud_height_initial: float = 0.0  # [m] (0 = auto)
    time_end: float = 3600.0  # [s]
    time_step_max: float = 0.0  # [s] (0 = auto)
    enable_transition: bool = True
    molecular_weight: float = 29.0

    def __post_init__(self):
        """Validate and set defaults."""
        if self.release_type == "instantaneous" and self.source_mass <= 0:
            raise ValueError(
                "source_mass must be > 0 for instantaneous release"
            )
        if self.release_type == "continuous":
            if self.source_rate <= 0:
                raise ValueError(
                    "source_rate must be > 0 for continuous release"
                )
            if self.release_duration <= 0:
                raise ValueError(
                    "release_duration must be > 0 for continuous release"
                )

    @property
    def density_ratio(self) -> float:
        """Ratio of cloud density to air density."""
        return self.cloud_density / max(self.air_density, 1e-6)

    @property
    def density_excess(self) -> float:
        """Density excess ratio: (ρ_c - ρ_a) / ρ_a."""
        return self.density_ratio - 1.0

    @property
    def total_mass(self) -> float:
        """Total mass released [kg]."""
        if self.release_type == "instantaneous":
            return self.source_mass
        else:
            return self.source_rate * self.release_duration

    @property
    def initial_volume(self) -> float:
        """Initial cloud volume [m³]."""
        return self.total_mass / max(self.cloud_density, 1e-6)


@dataclass
class DenseGasDensePhaseRecord:
    """Single time step record during dense gas phase.

    Attributes:
        time: Elapsed time [s].
        radius: Cloud radius at time t [m].
        height: Cloud height at time t [m].
        concentration_center: Centerline concentration [mg/m³].
        density_ratio: ρ_c / ρ_a at this time.
        richardson_number: Bulk Richardson number.
        volume: Cloud volume [m³].
        distance: Advection distance from source [m].
    """

    time: float
    radius: float
    height: float
    concentration_center: float  # [mg/m³]
    density_ratio: float
    richardson_number: float
    volume: float
    distance: float


@dataclass
class DenseGasResult:
    """Results of dense gas dispersion calculation.

    Attributes:
        time_series: Records during dense gas phase.
        transition_distance: Downwind distance where cloud becomes passive [m].
        transition_time: Time to transition [s].
        max_concentration: Peak centerline concentration [mg/m³].
        transition_radius: Cloud radius at transition [m].
        transition_height: Cloud height at transition [m].
        transition_density_ratio: Final density ratio before transition.
        initial_density_ratio: Initial density ratio.
        input: DenseGasInput used.
    """

    time_series: List[DenseGasDensePhaseRecord] = field(default_factory=list)
    transition_distance: float = 0.0
    transition_time: float = 0.0
    max_concentration: float = 0.0
    transition_radius: float = 0.0
    transition_height: float = 0.0
    transition_density_ratio: float = 0.0
    initial_density_ratio: float = 0.0
    input: Optional[DenseGasInput] = None

    @property
    def times(self) -> np.ndarray:
        return np.array([r.time for r in self.time_series])

    @property
    def radii(self) -> np.ndarray:
        return np.array([r.radius for r in self.time_series])

    @property
    def heights(self) -> np.ndarray:
        return np.array([r.height for r in self.time_series])

    @property
    def density_ratios(self) -> np.ndarray:
        return np.array([r.density_ratio for r in self.time_series])

    def to_dict(self) -> dict:
        """Serialize key results."""
        return {
            "transition_distance_m": self.transition_distance,
            "transition_time_s": self.transition_time,
            "max_concentration_mgm3": self.max_concentration,
            "initial_density_ratio": self.initial_density_ratio,
            "times": self.times.tolist(),
            "radii": self.radii.tolist(),
            "heights": self.heights.tolist(),
            "density_ratios": self.density_ratios.tolist(),
        }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def check_if_dense(
    substance_density: float,
    air_density: float,
    threshold: float = 1.1,
) -> bool:
    """Check whether a release qualifies as a dense gas.

    Args:
        substance_density: Cloud density [kg/m³].
        air_density: Ambient air density [kg/m³].
        threshold: Density ratio threshold (default 1.1).

    Returns:
        True if ρ_c/ρ_a > threshold.

    Examples:
        >>> check_if_dense(2.0, 1.2)
        True
        >>> check_if_dense(1.15, 1.2)
        False
    """
    if air_density <= 0:
        return False
    return (substance_density / air_density) > threshold


def transition_criteria(
    richardson_number: float,
    density_ratio: float,
    ri_critical: float = RI_CRITICAL,
    drho_critical: float = DELTA_RHO_CRITICAL,
) -> bool:
    """Determine whether dense cloud should transition to passive dispersion.

    Transition occurs when either:
    1. Richardson number < critical (turbulence overtakes gravity spreading)
    2. Density excess < critical (cloud diluted to near-ambient density)

    Args:
        richardson_number: Bulk Richardson number Ri.
        density_ratio: ρ_c / ρ_a.
        ri_critical: Critical Ri threshold.
        drho_critical: Critical density excess threshold.

    Returns:
        True if transition should occur.

    Examples:
        >>> transition_criteria(0.1, 1.02)  # low Ri
        True
        >>> transition_criteria(2.0, 1.005)  # near-ambient density
        True
        >>> transition_criteria(3.0, 2.0)  # still dense
        False
    """
    if richardson_number < ri_critical:
        return True
    if (density_ratio - 1.0) < drho_critical:
        return True
    return False


def _initial_cloud_geometry(input: DenseGasInput) -> Tuple[float, float]:
    """Estimate initial cloud radius and height.

    Args:
        input: DenseGasInput.

    Returns:
        Tuple of (radius_m, height_m).
    """
    if input.cloud_radius_initial > 0 and input.cloud_height_initial > 0:
        return input.cloud_radius_initial, input.cloud_height_initial

    volume = input.initial_volume

    if input.cloud_radius_initial > 0:
        r0 = input.cloud_radius_initial
        h0 = volume / (math.pi * r0 ** 2)
    elif input.cloud_height_initial > 0:
        h0 = input.cloud_height_initial
        r0 = math.sqrt(volume / (math.pi * h0))
    else:
        # Default: aspect ratio 1 (ΔH = R)
        r0 = math.pow(volume / math.pi, 1.0 / 3.0)
        h0 = r0

    return max(r0, 0.1), max(h0, 0.01)


def _richardson_number(
    density_excess: float,
    cloud_height: float,
    wind_speed: float,
) -> float:
    """Calculate bulk Richardson number for the dense cloud.

    Ri = g' · H / u*²

    where g' = g · (ρ_c - ρ_a) / ρ_a is reduced gravity,
    and u* is friction velocity (approximated as u* ≈ u/30).

    Args:
        density_excess: (ρ_c - ρ_a) / ρ_a.
        cloud_height: Cloud height [m].
        wind_speed: Wind speed [m/s].

    Returns:
        Richardson number.
    """
    if density_excess <= 0 or cloud_height <= 0:
        return 0.0

    g_prime = G * density_excess
    # Approximate friction velocity: u* ≈ u/30 (neutral stability)
    # More rigorously: u* = κ·u / ln(z/z0), but ~u/30 is common
    u_star = max(wind_speed / 30.0, 0.01)

    Ri = g_prime * cloud_height / (u_star ** 2)

    return max(Ri, 0.0)


def _entrainment_velocity_top(
    ri: float,
    u_star: float,
    alpha_top: float = ALPHA_TOP_DEFAULT,
) -> float:
    """Calculate top entrainment velocity.

    u_e_top = α_top · u* · f(Ri)

    where:
        f(Ri) = 1 / (1 + 0.5·Ri)  (for stable layers)

    Args:
        ri: Richardson number.
        u_star: Friction velocity [m/s].
        alpha_top: Top entrainment coefficient.

    Returns:
        Entrainment velocity [m/s].
    """
    f_ri = 1.0 / (1.0 + 0.5 * ri) if ri > 0 else 1.0
    return alpha_top * u_star * f_ri


def _entrainment_velocity_edge(
    ri: float,
    u_star: float,
    alpha_edge_base: float = ALPHA_EDGE_BASE,
) -> float:
    """Calculate edge entrainment velocity.

    Edge entrainment is driven by wind shear around the cloud.
    Typical form: u_e_edge ∝ u* · f(Ri).

    Args:
        ri: Richardson number.
        u_star: Friction velocity [m/s].
        alpha_edge_base: Base edge entrainment coefficient.

    Returns:
        Entrainment velocity [m/s].
    """
    # Edge entrainment decreases with increasing Ri
    f_ri = 1.0 / (1.0 + ri) if ri > 0 else 1.0
    return alpha_edge_base * u_star * f_ri


# ---------------------------------------------------------------------------
# Main Dense Gas Calculation
# ---------------------------------------------------------------------------


def calculate_dense_gas(input: DenseGasInput) -> DenseGasResult:
    """Calculate dense gas dispersion.

    Simulates the three phases:
    1. Gravity spreading (slumping)
    2. Air entrainment and dilution
    3. Transition to passive dispersion

    Args:
        input: Complete DenseGasInput.

    Returns:
        DenseGasResult with time series and transition data.

    Examples:
        >>> dg_input = DenseGasInput(
        ...     source_mass=1000.0,
        ...     release_type='instantaneous',
        ...     cloud_density=2.0,
        ...     air_density=1.2,
        ...     wind_speed=3.0,
        ... )
        >>> result = calculate_dense_gas(dg_input)
        >>> result.max_concentration > 0
        True
        >>> result.initial_density_ratio > 1.1
        True
    """
    g = G
    u = max(input.wind_speed, 0.5)
    u_star = max(u / 30.0, 0.01)

    # Initial geometry
    r0, h0 = _initial_cloud_geometry(input)

    # Initial conditions
    r = r0
    h = h0
    V = input.initial_volume
    mass_initial = input.total_mass
    rho_c = input.cloud_density
    rho_a = input.air_density
    drho_excess = (rho_c - rho_a) / rho_a
    ri = _richardson_number(drho_excess, h, u)

    # Concentration (centerline, uniform in cloud)
    C_center = (mass_initial / V) * 1e6  # [mg/m³]

    # Advection velocity (wind carries cloud)
    # For dense gas, wind speed at cloud height is somewhat reduced
    u_cloud = u * 0.7  # reduced by cloud stability

    # Time stepping
    t = 0.0
    x_pos = 0.0  # downwind position of cloud center

    # Determine time step
    if input.time_step_max > 0:
        dt_max = input.time_step_max
    else:
        # Auto time step based on spreading time scale
        g_prime = g * drho_excess if drho_excess > 0 else 0.01
        spreading_timescale = math.sqrt(r0 / g_prime) if g_prime > 0 else 10.0
        dt_max = max(0.1, DT_MULTIPLIER * spreading_timescale)

    # Time series records
    records: List[DenseGasDensePhaseRecord] = []

    # Record initial state
    records.append(DenseGasDensePhaseRecord(
        time=t,
        radius=r,
        height=h,
        concentration_center=C_center,
        density_ratio=rho_c / rho_a,
        richardson_number=ri,
        volume=V,
        distance=x_pos,
    ))

    max_C_seen = C_center

    # Simulation loop
    transition_occurred = False
    max_iterations = 50000
    iter_count = 0

    while t < input.time_end and iter_count < max_iterations:
        iter_count += 1

        # Check transition criteria
        if input.enable_transition and transition_criteria(ri, rho_c / rho_a):
            transition_occurred = True
            break

        # --- Phase 1: Gravity Spreading ---
        # Radial spreading rate: dR/dt = C · sqrt(g' · h)
        g_prime = g * drho_excess if drho_excess > 0 else 0.0

        if g_prime > 1e-6 and h > 0:
            dR_dt_spread = GRAVITY_SPREADING_C * math.sqrt(g_prime * h)
        else:
            dR_dt_spread = 0.0

        # --- Phase 2: Air Entrainment ---
        # Top entrainment: dV/dt = α_top · π · R² · u_e_top
        u_e_top = _entrainment_velocity_top(ri, u_star)
        dV_dt_top = ALPHA_TOP_DEFAULT * math.pi * r ** 2 * u_e_top

        # Edge entrainment: dV/dt = α_edge · 2π · R · H · u_e_edge
        u_e_edge = _entrainment_velocity_edge(ri, u_star)
        dV_dt_edge = ALPHA_EDGE_BASE * 2.0 * math.pi * r * h * u_e_edge

        dV_dt = dV_dt_top + dV_dt_edge

        # --- Time Step ---
        # Limit time step for numerical stability
        dt_spread = r / max(dR_dt_spread, 0.001) * 0.1
        dt_entrain = V / max(dV_dt, 0.001) * 0.1
        dt = min(dt_max, dt_spread, dt_entrain)
        dt = max(dt, 0.01)
        dt = min(dt, input.time_end - t)

        # --- Integration ---
        # Update radius
        r_new = r + dR_dt_spread * dt

        # Update volume via entrainment
        V_new = V + dV_dt * dt

        # Update cloud height: H = V / (π · R²) for cylindrical cloud
        # But spreading also thins the cloud
        if r_new > 0:
            h_new = V_new / (math.pi * r_new ** 2)
        else:
            h_new = h

        # Update density (dilution)
        # Mass of substance is conserved, volume increases
        # ρ_cloud = (mass_substance + mass_air_entrained) / V
        # Simple dilution: cloud density decreases toward ambient
        if V_new > 0:
            # Air mass entrained
            V_entrained = V_new - V
            if V_entrained > 0:
                # Mixing: cloud mass remains, but volume increases
                # New density = (ρ_c_old * V + ρ_a * V_entrained) / V_new
                rho_c_new = (rho_c * V + rho_a * V_entrained) / V_new
            else:
                rho_c_new = rho_c
        else:
            rho_c_new = rho_c

        # Update mass concentration
        if V_new > 0:
            C_center = (mass_initial / V_new) * 1e6
        else:
            C_center = C_center

        # Update position (advection)
        x_pos += u_cloud * dt

        # Update state
        r = r_new
        h = h_new
        V = V_new
        rho_c = rho_c_new
        drho_excess = max((rho_c - rho_a) / rho_a, 0.0)
        ri = _richardson_number(drho_excess, h, u)
        t += dt

        if C_center > max_C_seen:
            max_C_seen = C_center

        records.append(DenseGasDensePhaseRecord(
            time=t,
            radius=r,
            height=h,
            concentration_center=C_center,
            density_ratio=rho_c / rho_a,
            richardson_number=ri,
            volume=V,
            distance=x_pos,
        ))

    # Transition results
    if transition_occurred:
        trans_dist = x_pos
        trans_time = t
        trans_radius = r
        trans_height = h
        trans_dr = rho_c / rho_a
    else:
        # Ran to end of simulation without transition
        trans_dist = x_pos
        trans_time = t
        trans_radius = r
        trans_height = h
        trans_dr = rho_c / rho_a

    return DenseGasResult(
        time_series=records,
        transition_distance=trans_dist,
        transition_time=trans_time,
        max_concentration=max_C_seen,
        transition_radius=trans_radius,
        transition_height=trans_height,
        transition_density_ratio=trans_dr,
        initial_density_ratio=input.density_ratio,
        input=input,
    )


# ---------------------------------------------------------------------------
# Dense Gas Calculator Class
# ---------------------------------------------------------------------------


class DenseGasCalculator:
    """Main calculator for dense gas dispersion.

    Usage:
        calc = DenseGasCalculator()

        # Check if dense
        if calc.check_if_dense(2.0, 1.2):
            result = calc.calculate(dense_input)
    """

    def calculate(self, input: DenseGasInput) -> DenseGasResult:
        """Run dense gas dispersion simulation.

        Args:
            input: DenseGasInput.

        Returns:
            DenseGasResult.
        """
        return calculate_dense_gas(input)

    @staticmethod
    def check_if_dense(
        substance_density: float,
        air_density: float,
        threshold: float = 1.1,
    ) -> bool:
        """Check if release qualifies as dense gas."""
        return check_if_dense(substance_density, air_density, threshold)

    @staticmethod
    def transition_criteria(
        ri: float,
        density_ratio: float,
    ) -> bool:
        """Check if transition to passive dispersion should occur."""
        return transition_criteria(ri, density_ratio)
