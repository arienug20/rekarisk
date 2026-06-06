"""
Rekarisk Dispersion — Continuous Gaussian Plume Model.

Implements the steady-state Gaussian plume dispersion model for continuous
releases of buoyant or neutrally buoyant gases, based on the Pasquill-Gifford
dispersion coefficients.

Core Equation:
    C(x,y,z) = Q/(2π·u·σy·σz) · exp(-y²/2σy²)
                · [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)]

References:
    - Turner, D.B. (1994). Workbook of Atmospheric Dispersion Estimates.
    - Briggs, G.A. (1973). Diffusion estimation for small emissions.
    - CCPS Guidelines for Consequence Analysis of Chemical Releases.
    - TNO Yellow Book, Chapter 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from rekarisk.core.constants import G, T_0C, P_ATM
from rekarisk.meteorology.stability import (
    StabilityClass,
    TerrainType,
    sigma_y,
    sigma_z,
    sigma_y_corrected,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PlumeModelType = Literal["gaussian", "dense", "puff", "jet"]

# ---------------------------------------------------------------------------
# Building Wake Factor (Huber-Snyder Method)
# ---------------------------------------------------------------------------

@dataclass
class BuildingParams:
    """Building geometry for wake effect calculations.

    Attributes:
        height: Building height H_b [m].
        width: Building width W_b (crosswind) [m].
        length: Building length L_b (downwind) [m].
    """
    height: float = 0.0
    width: float = 0.0
    length: float = 0.0


def sigma_building_wake(
    x: float,
    sigma_atm: float,
    building: BuildingParams,
    sigma_type: str = "y",
) -> float:
    """Apply Huber-Snyder building wake enhancement to dispersion coefficient.

    The Huber-Snyder method (EPA, 1985) accounts for increased turbulence
    in the wake region of a building. Within ~3-5 building heights downwind,
    the plume is well-mixed in the wake cavity, enhancing both lateral and
    vertical dispersion.

    Enhancement formula:
        σ_eff = sqrt(σ_atm² + c²)

    where:
        c = 0.35 × W_b  for lateral (σ_y)
        c = 0.70 × H_b  for vertical (σ_z)

    The enhancement decays exponentially beyond 5 building heights:
        decay = exp(-0.5 × (x / (5 × H_b) - 1)²)   for x > 5 × H_b

    Args:
        x: Downwind distance [m].
        sigma_atm: Unperturbed atmospheric sigma (σ_y or σ_z) [m].
        building: Building geometry parameters.
        sigma_type: 'y' for lateral, 'z' for vertical.

    Returns:
        Enhanced sigma [m] (always >= sigma_atm).
    """
    H = building.height
    if H <= 0:
        return sigma_atm

    if sigma_type == "y":
        c = 0.35 * building.width
    else:
        c = 0.70 * H

    # Wake zone extends to ~5 building heights
    wake_distance = 5.0 * H
    decay = 1.0
    if x > wake_distance and wake_distance > 0:
        # Smooth exponential decay beyond wake zone
        ratio = (x - wake_distance) / wake_distance
        decay = max(0.05, math.exp(-0.5 * ratio ** 2))

    sigma_enhanced = math.sqrt(sigma_atm ** 2 + (c * decay) ** 2)
    return max(sigma_enhanced, sigma_atm)


def building_wake_correction(
    x: float,
    sigma_y_val: float,
    sigma_z_val: float,
    building: BuildingParams,
) -> Tuple[float, float]:
    """Apply full building wake correction to sigma_y and sigma_z.

    Args:
        x: Downwind distance [m].
        sigma_y_val: Uncorrected sigma_y [m].
        sigma_z_val: Uncorrected sigma_z [m].
        building: Building geometry.

    Returns:
        (sigma_y_corrected, sigma_z_corrected).
    """
    if building.height <= 0:
        return sigma_y_val, sigma_z_val

    sy = sigma_building_wake(x, sigma_y_val, building, "y")
    sz = sigma_building_wake(x, sigma_z_val, building, "z")
    return sy, sz


# ---------------------------------------------------------------------------
# Input Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlumeInput:
    """Input parameters for continuous Gaussian plume dispersion calculation.

    Attributes:
        source_rate: Emission rate Q [kg/s].
        wind_speed: Mean wind speed u at release height [m/s].
        stability_class: Pasquill-Gifford stability class ('A' through 'F').
        release_height: Effective stack height H = physical height + plume rise [m].
        terrain_type: 'rural' or 'urban'.
        temperature: Ambient temperature [K] (for buoyancy/density corrections).
        pressure: Ambient pressure [Pa].
        decay_rate: First-order chemical decay rate λ [1/s] (default 0 = no decay).
        deposition_velocity: Dry deposition velocity v_d [m/s] (default 0 = no deposition).
        sampling_time: Sampling/averaging time [s] for σ correction (default 600 = 10 min).
        source_diameter: Stack/vent inside diameter [m] (for plume rise).
        stack_exit_velocity: Exit velocity at stack tip [m/s] (for momentum rise).
        stack_temperature: Stack gas temperature [K] (for buoyancy rise).
        reference_time: Reference sampling time for σ correction [s] (default 600).
        grid_x_range: Tuple of (x_min, x_max, x_points) for concentration grid [m].
        grid_y_range: Tuple of (y_min, y_max, y_points) for concentration grid [m].
        grid_z_range: Tuple of (z_min, z_max, z_points) for concentration grid [m].
        molecular_weight: Molecular weight [g/mol] (for unit conversions).
    """

    source_rate: float = 1.0  # Q [kg/s]
    wind_speed: float = 5.0  # u [m/s]
    stability_class: StabilityClass = "D"
    release_height: float = 0.0  # H [m], effective (physical + plume rise)
    terrain_type: TerrainType = "rural"
    temperature: float = 298.15  # [K]
    pressure: float = P_ATM  # [Pa]
    decay_rate: float = 0.0  # λ [1/s]
    deposition_velocity: float = 0.0  # v_d [m/s]
    sampling_time: float = 600.0  # [s], default 10 min
    source_diameter: float = 0.0  # d [m]
    stack_exit_velocity: float = 0.0  # Vs [m/s]
    stack_temperature: float = 298.15  # T_s [K]
    molecular_weight: float = 29.0  # MW [g/mol]
    reference_time: float = 600.0  # t_ref [s]
    building: Optional[BuildingParams] = None  # building geometry for wake factor
    grid_x_range: Tuple[float, float, int] = field(
        default_factory=lambda: (100.0, 5000.0, 50)
    )
    grid_y_range: Tuple[float, float, int] = field(
        default_factory=lambda: (-500.0, 500.0, 51)
    )
    grid_z_range: Tuple[float, float, int] = field(
        default_factory=lambda: (0.0, 200.0, 21)
    )

    def __post_init__(self):
        """Validate and clamp inputs."""
        if self.source_rate < 0:
            raise ValueError(f"Source rate must be non-negative, got {self.source_rate}")
        if self.wind_speed < 0.1:
            raise ValueError(f"Wind speed must be ≥ 0.1 m/s for plume dispersion")

        # Ensure valid grid ranges
        for attr, name in [
            (self.grid_x_range, "grid_x_range"),
            (self.grid_y_range, "grid_y_range"),
            (self.grid_z_range, "grid_z_range"),
        ]:
            if attr[0] >= attr[1]:
                raise ValueError(f"{name}: x_min must be < x_max")
            if attr[2] < 2:
                raise ValueError(f"{name}: at least 2 points required")


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlumeResult:
    """Results of continuous Gaussian plume dispersion calculation.

    Attributes:
        concentration_grid: 3D concentration array [mg/m³] with shape
            (n_x, n_y, n_z) indexed as C[x_idx, y_idx, z_idx].
        x_coords: Downwind distance coordinates [m] (shape: n_x).
        y_coords: Cross-wind coordinates [m] (shape: n_y).
        z_coords: Vertical coordinates [m] (shape: n_z).
        max_concentration: Peak concentration in the grid [mg/m³].
        max_distance: Downwind distance to peak concentration [m].
        max_y_distance: Cross-wind offset of peak [m] (always 0 for plume).
        max_z_height: Height of peak concentration [m].
        centerline_concentration: Array of C(x, 0, 0) along centerline [mg/m³].
        ground_concentration: 2D array C(x, y, 0) at ground level [mg/m³].
        input: The PlumeInput used for this calculation.
        plume_rise_delta: Computed plume rise [m] (0 if not applicable).
    """

    concentration_grid: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1)))
    x_coords: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y_coords: np.ndarray = field(default_factory=lambda: np.zeros(1))
    z_coords: np.ndarray = field(default_factory=lambda: np.zeros(1))
    max_concentration: float = 0.0
    max_distance: float = 0.0
    max_y_distance: float = 0.0
    max_z_height: float = 0.0
    centerline_concentration: np.ndarray = field(
        default_factory=lambda: np.zeros(1)
    )
    ground_concentration: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1))
    )
    input: Optional[PlumeInput] = None
    plume_rise_delta: float = 0.0

    @property
    def concentration_units(self) -> str:
        return "mg/m³"

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.concentration_grid.shape

    def to_dict(self) -> dict:
        """Serialize key results to dictionary for JSON export."""
        return {
            "max_concentration_mgm3": float(self.max_concentration),
            "max_distance_m": float(self.max_distance),
            "plume_rise_m": float(self.plume_rise_delta),
            "x_coords": self.x_coords.tolist(),
            "y_coords": self.y_coords.tolist(),
            "z_coords": self.z_coords.tolist(),
            "ground_concentration": self.ground_concentration.tolist(),
            "centerline_concentration": self.centerline_concentration.tolist(),
        }


# ---------------------------------------------------------------------------
# Briggs Plume Rise
# ---------------------------------------------------------------------------


def plume_rise_briggs(
    Q_heat: float,
    wind_speed: float,
    stack_diameter: float,
    T_stack: float,
    T_ambient: float,
    stack_height: float = 0.0,
    max_distance: float = 1000.0,
) -> float:
    """Calculate plume rise using Briggs equations.

    Determines whether buoyancy or momentum dominates, then calculates
    the final plume rise ΔH. Uses the Briggs (1975) formulations for
    bent-over plumes in neutral/unstable conditions (classes A-D).

    Args:
        Q_heat: Heat emission rate from stack [W or J/s].
            Estimate as: Q_heat = m_dot * Cp * (T_stack - T_ambient).
        wind_speed: Wind speed at stack height [m/s].
        stack_diameter: Inside stack diameter [m].
        T_stack: Stack gas absolute temperature [K].
        T_ambient: Ambient air absolute temperature [K].
        stack_height: Physical stack height [m].
        max_distance: Downwind distance to calculate rise at [m].

    Returns:
        Plume rise ΔH [m]. Total effective height = stack_height + ΔH.

    References:
        Briggs, G.A. (1975). Plume rise predictions. Lectures on Air
        Pollution and Environmental Impact Analysis, AMS.

    Examples:
        >>> # 10 MW heat, 5 m/s wind, 2m stack, 400K, 300K
        >>> delta = plume_rise_briggs(1e7, 5.0, 2.0, 400.0, 300.0)
        >>> delta > 0
        True
    """
    if wind_speed < 0.5:
        wind_speed = 0.5  # minimum for calculation
    if stack_diameter <= 0:
        return 0.0

    g = G

    # Buoyancy flux parameter F_b [m⁴/s³]
    # F = g * V_dot * (T_s - T_a) / T_s
    # where V_dot is volumetric flow rate
    # alternatively: F = g * Q_heat / (π * ρ * Cp * T_s) ≈ g * Q_heat / (π * ρ * Cp * T)
    # Simplified: F ≈ g * Q_h * (T_s - T_a) / (π * T_s)
    # Standard form: F = g * Vs * d²/4 * (T_s - T_a) / T_s
    # But Q_heat = 0.25 * π * d² * Vs * ρ * Cp * ΔT
    # So we compute F from heat output:
    # F = (g * Q_heat) / (π * ρ_ambient * Cp * T_ambient)
    # Use approximate: ρ*Cp ≈ 1200 J/(m³·K) at ambient conditions
    rho_cp = 1200.0  # approximate volumetric heat capacity of air [J/(m³·K)]
    F_b = (g * Q_heat) / (math.pi * rho_cp * T_ambient)

    if F_b <= 0:
        # No buoyancy, try momentum
        F_b = 0.0

    # Momentum flux parameter F_m [m⁴/s²]
    # F_m = Vs² * d² * T_a / (4 * T_s)
    # For now, calculate from diameter if available
    F_m = 0.0

    # Determine if buoyancy or momentum dominates
    delta_T = T_stack - T_ambient
    if delta_T > 0:
        # Buoyancy-dominated plume rise
        # Briggs formula for unstable/neutral:
        # ΔH = 1.6 * F_b^(1/3) * x^(2/3) / u  for x < x_f
        # where x_f is the distance to final rise
        # Final rise: ΔH_fi = 39 * F_b^(2/5) / u  (for F_b < 55)
        #            ΔH_fi = 5 * F_b^(1/4) / u    (for F_b >= 55)

        # Buoyancy-dominated: find x_star = distance where atmospheric
        # turbulence begins to dominate
        if F_b < 55:
            # Unstable/neutral, F_b < 55
            x_star = 14 * F_b ** (5 / 8)
            if max_distance < x_star:
                # Early stage rise
                delta_h = 1.6 * F_b ** (1 / 3) * max_distance ** (2 / 3) / wind_speed
            else:
                delta_h = 1.6 * F_b ** (1 / 3) * x_star ** (2 / 3) / wind_speed
        else:
            # F_b >= 55
            s_param = (g * delta_T) / (T_ambient * wind_speed)
            x_star = 119 * F_b ** (2 / 5)
            if max_distance < x_star:
                delta_h = 1.6 * F_b ** (1 / 3) * max_distance ** (2 / 3) / wind_speed
            else:
                delta_h = 1.6 * F_b ** (1 / 3) * x_star ** (2 / 3) / wind_speed
    elif F_m > 0:
        # Momentum-dominated
        # ΔH = 3 * d * Vs / u (simplified Briggs momentum rise)
        delta_h = 3.0 * stack_diameter * (F_m ** 0.5) / wind_speed
    else:
        delta_h = 0.0

    # Briggs also gives formulas for stable conditions; skipped for now
    # (E, F classes use different formulas)

    return max(0.0, delta_h)


def plume_rise(
    Q_heat: float,
    wind_speed: float,
    stack_diameter: float,
    T_stack: float,
    T_ambient: float,
    stack_height: float = 0.0,
    max_distance: float = 1000.0,
) -> float:
    """Convenience wrapper for plume_rise_briggs.

    Args:
        Q_heat: Heat emission rate [W].
        wind_speed: Wind speed [m/s].
        stack_diameter: Stack diameter [m].
        T_stack: Stack gas temperature [K].
        T_ambient: Ambient temperature [K].
        stack_height: Physical stack height [m].
        max_distance: Distance for rise calculation [m].

    Returns:
        Plume rise ΔH [m].
    """
    return plume_rise_briggs(
        Q_heat=Q_heat,
        wind_speed=wind_speed,
        stack_diameter=stack_diameter,
        T_stack=T_stack,
        T_ambient=T_ambient,
        stack_height=stack_height,
        max_distance=max_distance,
    )


# ---------------------------------------------------------------------------
# Concentration Calculation — Single Point
# ---------------------------------------------------------------------------


def concentration_at_point(
    x: float,
    y: float,
    z: float,
    input: PlumeInput,
    sigma_y_val: Optional[float] = None,
    sigma_z_val: Optional[float] = None,
) -> float:
    """Calculate Gaussian plume concentration at a single (x, y, z) point.

    C(x,y,z) = Q/(2π·u·σy·σz) · exp(-y²/2σy²)
                · [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)]

    Args:
        x: Downwind distance from source [m].
        y: Cross-wind distance from centerline [m].
        z: Height above ground [m].
        input: PlumeInput with release and environmental parameters.
        sigma_y_val: Pre-calculated σy at distance x [m]. If None, computed.
        sigma_z_val: Pre-calculated σz at distance x [m]. If None, computed.

    Returns:
        Concentration [kg/m³] (SI units). Convert to mg/m³ for display.

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...                  release_height=0.0)
        >>> c = concentration_at_point(1000.0, 0.0, 0.0, inp)
        >>> c > 0
        True
    """
    if x <= 0:
        return 0.0

    u = max(input.wind_speed, 0.1)
    H = input.release_height

    # Get dispersion coefficients
    if sigma_y_val is None:
        sy = sigma_y(x, input.stability_class, input.terrain_type)
        # Apply sampling time correction if needed
        if input.sampling_time != input.reference_time:
            sy = sigma_y_corrected(
                x, input.stability_class, input.terrain_type,
                input.sampling_time, input.reference_time,
            )
    else:
        sy = sigma_y_val

    if sigma_z_val is None:
        sz = sigma_z(x, input.stability_class, input.terrain_type)
    else:
        sz = sigma_z_val

    # Apply building wake correction (Huber-Snyder)
    if input.building is not None and input.building.height > 0:
        sy, sz = building_wake_correction(x, sy, sz, input.building)

    if sy <= 0 or sz <= 0:
        return 0.0

    # Base concentration (without reflections, decay, or deposition)
    denom = 2.0 * math.pi * u * sy * sz
    if denom <= 0:
        return 0.0

    C = input.source_rate / denom

    # Lateral (cross-wind) term
    C *= math.exp(-0.5 * (y / sy) ** 2)

    # Vertical term with ground reflection
    if sz > 0:
        term1 = math.exp(-0.5 * ((z - H) / sz) ** 2)
        term2 = math.exp(-0.5 * ((z + H) / sz) ** 2)
        C *= (term1 + term2)

    # Chemical decay: exp(-λ * x / u)
    if input.decay_rate > 0:
        travel_time = x / u
        C *= math.exp(-input.decay_rate * travel_time)

    # Dry deposition correction (source depletion model, simplified)
    if input.deposition_velocity > 0:
        # Simplified deposition: reduces effective source rate with distance
        # More sophisticated models use source depletion; this is a first-order
        # approximation that reduces concentration for distances beyond ~100m
        if x > 100.0:
            # Approximate vertical mixing scale
            mix_scale = max(sz, 1.0)
            depletion_factor = math.exp(
                -math.sqrt(2.0 / math.pi)
                * input.deposition_velocity
                * x
                / (u * mix_scale)
            )
            C *= depletion_factor

    return max(0.0, C)


# ---------------------------------------------------------------------------
# Grid Calculation
# ---------------------------------------------------------------------------


def calculate_plume(
    input: PlumeInput,
    jet_velocity: float = 0.0,
    hole_diameter: float = 0.0,
    release_density: float = 1.2,
) -> PlumeResult:
    """Calculate full 3D concentration grid for a Gaussian plume.

    Computes concentration on a 3D grid specified by the input parameters.
    When jet_velocity and hole_diameter are provided (both > 0), delegates
    to the jet-enhanced model that accounts for initial jet momentum
    dispersion. This gives more accurate near-field results matching
    PHAST UDM behavior.

    Args:
        input: Complete PlumeInput with grid specification.
        jet_velocity: Exit velocity of the jet [m/s]. When > 0 together with
            hole_diameter > 0, jet-enhanced dispersion is used.
        hole_diameter: Release hole/rupture diameter [m].
        release_density: Density of released gas [kg/m³] (default air=1.2).

    Returns:
        PlumeResult with concentration grid, coordinates, and summary stats.

    Examples:
        >>> inp = PlumeInput(
        ...     source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...     release_height=0.0,
        ...     grid_x_range=(100, 5000, 50),
        ...     grid_y_range=(-500, 500, 51),
        ...     grid_z_range=(0, 200, 21),
        ... )
        >>> result = calculate_plume(inp)
        >>> result.max_concentration > 0
        True
        >>> result.concentration_grid.shape == (50, 51, 21)
        True
    """
    # Delegate to jet-enhanced model if jet parameters are provided
    if jet_velocity > 0 and hole_diameter > 0:
        return calculate_plume_with_jet(
            input=input,
            jet_velocity=jet_velocity,
            hole_diameter=hole_diameter,
            release_density=release_density,
        )

    # Create coordinate arrays
    x_min, x_max, n_x = input.grid_x_range
    y_min, y_max, n_y = input.grid_y_range
    z_min, z_max, n_z = input.grid_z_range

    x_coords = np.linspace(x_min, x_max, n_x)
    y_coords = np.linspace(y_min, y_max, n_y)
    z_coords = np.linspace(z_min, z_max, n_z)

    # Pre-compute sigma_y and sigma_z for each x position
    # (sigma depends only on x, not y/z)
    sy_arr = np.zeros(n_x)
    sz_arr = np.zeros(n_x)
    u = max(input.wind_speed, 0.1)

    for i, x in enumerate(x_coords):
        sy_arr[i] = sigma_y(x, input.stability_class, input.terrain_type)
        sz_arr[i] = sigma_z(x, input.stability_class, input.terrain_type)
        # Apply sampling time correction
        if input.sampling_time != input.reference_time:
            sy_arr[i] = sigma_y_corrected(
                x, input.stability_class, input.terrain_type,
                input.sampling_time, input.reference_time,
            )
        # Apply building wake correction (Huber-Snyder)
        if input.building is not None and input.building.height > 0:
            sy_arr[i], sz_arr[i] = building_wake_correction(x, sy_arr[i], sz_arr[i], input.building)

    # Initialize 3D concentration grid [mg/m³]
    C_grid = np.zeros((n_x, n_y, n_z), dtype=np.float64)
    ground_C = np.zeros((n_x, n_y), dtype=np.float64)
    centerline_C = np.zeros(n_x, dtype=np.float64)

    H = input.release_height
    Q = input.source_rate
    two_pi_u = 2.0 * math.pi * u

    # Vectorized calculation using broadcasting
    # C(x,y,z) = Q/(2π·u·σy·σz) · exp(-y²/2σy²) · [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)]

    # Precompute lateral (y) and vertical (z) components
    # Shape: (n_y,) and (n_z,)
    y_sq = y_coords ** 2  # (n_y,)
    z_minus_H_sq = (z_coords - H) ** 2  # (n_z,)
    z_plus_H_sq = (z_coords + H) ** 2  # (n_z,)

    for i in range(n_x):
        sy = sy_arr[i]
        sz = sz_arr[i]

        if sy <= 0 or sz <= 0:
            continue

        # Base factor: Q / (2π·u·σy·σz)
        base_factor = Q / (two_pi_u * sy * sz)

        # Lateral term: shape (n_y,)
        lat_term = np.exp(-0.5 * y_sq / (sy ** 2))

        # Vertical term with ground reflection: shape (n_z,)
        if sz > 0:
            term1 = np.exp(-0.5 * z_minus_H_sq / (sz ** 2))
            term2 = np.exp(-0.5 * z_plus_H_sq / (sz ** 2))
            vert_term = term1 + term2
        else:
            vert_term = np.zeros(n_z)

        # Outer product: (n_y,) × (n_z,) → (n_y, n_z)
        # C_2d = base_factor * lat_term[:, None] * vert_term[None, :]
        C_2d = base_factor * np.outer(lat_term, vert_term)

        # Apply chemical decay
        if input.decay_rate > 0:
            travel_time = x_coords[i] / u
            C_2d *= math.exp(-input.decay_rate * travel_time)

        # Apply dry deposition
        if input.deposition_velocity > 0 and x_coords[i] > 100.0:
            mix_scale = max(sz, 1.0)
            depletion_factor = math.exp(
                -math.sqrt(2.0 / math.pi)
                * input.deposition_velocity
                * x_coords[i]
                / (u * mix_scale)
            )
            C_2d *= depletion_factor

        # Store in grid
        C_grid[i, :, :] = C_2d

        # Extract ground-level (z ≈ 0, first z index)
        ground_C[i, :] = C_2d[:, 0]

        # Extract centerline (y=0, z=H or z=0 for ground)
        y_mid = n_y // 2
        z_mid = 0  # ground level
        centerline_C[i] = C_2d[y_mid, z_mid]

    # Convert from kg/m³ to mg/m³
    C_grid_mgm3 = C_grid * 1e6
    ground_C_mgm3 = ground_C * 1e6
    centerline_C_mgm3 = centerline_C * 1e6

    # Find max concentration and location
    max_flat_idx = np.argmax(C_grid)
    max_idx = np.unravel_index(max_flat_idx, C_grid.shape)
    max_C = C_grid_mgm3[max_idx]
    max_x = x_coords[max_idx[0]]
    max_y = y_coords[max_idx[1]]
    max_z = z_coords[max_idx[2]]

    return PlumeResult(
        concentration_grid=C_grid_mgm3,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        max_concentration=float(max_C),
        max_distance=float(max_x),
        max_y_distance=float(max_y),
        max_z_height=float(max_z),
        centerline_concentration=centerline_C_mgm3,
        ground_concentration=ground_C_mgm3,
        input=input,
        plume_rise_delta=0.0,
    )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def concentration_at(
    x: float,
    y: float,
    z: float,
    input: PlumeInput,
) -> float:
    """Calculate concentration at a single point, returning mg/m³.

    Convenience wrapper around concentration_at_point that converts
    SI internal units (kg/m³) to display units (mg/m³).

    Args:
        x: Downwind distance [m].
        y: Cross-wind distance [m].
        z: Height above ground [m].
        input: PlumeInput parameters.

    Returns:
        Concentration [mg/m³].

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...                  release_height=0.0)
        >>> c = concentration_at(1000, 0, 0, inp)
        >>> c > 0
        True
    """
    return concentration_at_point(x, y, z, input) * 1e6


def centerline_profile(
    x_coords: np.ndarray,
    input: PlumeInput,
    z: float = 0.0,
) -> np.ndarray:
    """Compute concentration along the plume centerline (y=0) at height z.

    Args:
        x_coords: Array of downwind distances [m].
        input: PlumeInput parameters.
        z: Height above ground [m] (default 0 = ground level).

    Returns:
        Array of concentrations [mg/m³] at each x.

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...                  release_height=0.0)
        >>> x = np.logspace(2, 4, 50)
        >>> c = centerline_profile(x, inp)
        >>> len(c) == 50
        True
    """
    n = len(x_coords)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = concentration_at_point(x_coords[i], 0.0, z, input) * 1e6
    return result


def crosswind_profile(
    x: float,
    y_coords: np.ndarray,
    input: PlumeInput,
    z: float = 0.0,
) -> np.ndarray:
    """Compute concentration across the plume at fixed downwind distance x.

    Args:
        x: Downwind distance [m].
        y_coords: Cross-wind coordinate array [m].
        input: PlumeInput parameters.
        z: Height above ground [m] (default 0 = ground level).

    Returns:
        Array of concentrations [mg/m³] at each y.

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...                  release_height=0.0)
        >>> y = np.linspace(-200, 200, 41)
        >>> c = crosswind_profile(1000.0, y, inp)
        >>> len(c) == 41
        True
    """
    n = len(y_coords)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = concentration_at_point(x, y_coords[i], z, input) * 1e6
    return result


def vertical_profile(
    x: float,
    z_coords: np.ndarray,
    input: PlumeInput,
    y: float = 0.0,
) -> np.ndarray:
    """Compute concentration vertically at fixed (x, y).

    Args:
        x: Downwind distance [m].
        z_coords: Vertical coordinate array [m].
        input: PlumeInput parameters.
        y: Cross-wind distance [m] (default 0 = centerline).

    Returns:
        Array of concentrations [mg/m³] at each z.

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0, stability_class='D',
        ...                  release_height=50.0)
        >>> z = np.linspace(0, 200, 41)
        >>> c = vertical_profile(1000.0, z, inp)
        >>> len(c) == 41
        True
    """
    n = len(z_coords)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = concentration_at_point(x, y, z_coords[i], input) * 1e6
    return result


# ---------------------------------------------------------------------------
# Isopleth / Contour Calculation
# ---------------------------------------------------------------------------


def isopleth_data(
    input: PlumeInput,
    target_concentrations: List[float],
    z: float = 0.0,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Compute isopleth (contour) data for ground-level concentrations.

    For each target concentration, finds the contour polygon by
    evaluating concentration on a dense grid and extracting the contour.

    Args:
        input: PlumeInput parameters.
        target_concentrations: List of target concentrations [mg/m³]
            (e.g., ERPG values, LFL).
        z: Height for isopleth plane [m] (default 0 = ground level).

    Returns:
        Dict mapping target concentration → dict with 'x_contour' and
        'y_contour' arrays defining the contour shape.

    Examples:
        >>> inp = PlumeInput(source_rate=10.0, wind_speed=3.0, stability_class='F',
        ...                  release_height=0.0)
        >>> contours = isopleth_data(inp, [100, 500, 1000])
        >>> isinstance(contours, dict)
        True
    """
    # Create a dense evaluation grid
    x_log = np.logspace(1, 5, 200)  # 10m to 100km
    y_lin = np.linspace(-2000, 2000, 200)

    # Evaluate ground concentration on this grid
    C_ground = np.zeros((len(x_log), len(y_lin)), dtype=np.float64)

    for i, x in enumerate(x_log):
        for j, y in enumerate(y_lin):
            C_ground[i, j] = concentration_at_point(x, y, z, input) * 1e6

    results: Dict[float, Dict[str, np.ndarray]] = {}

    for target in target_concentrations:
        # Find the contour at this concentration level
        # Simple approach: find the x-position where centerline C ≈ target
        centerline = C_ground[:, len(y_lin) // 2]
        x_idx = np.argmin(np.abs(centerline - target))

        # Create a simplified contour: for y at that x, find where C crosses target
        cross_profile = C_ground[x_idx, :]

        # Find y bounds where C > target
        above = cross_profile > target
        if not np.any(above):
            continue

        y_indices = np.where(above)[0]
        y_min = float(y_lin[y_indices[0]])
        y_max = float(y_lin[y_indices[-1]])

        # Build contour outline
        x_contour = float(x_log[x_idx])
        n_pts = 50
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
        # Approximate elliptical shape
        contour_x = x_contour + 0.1 * x_contour * np.cos(theta)
        contour_y = y_lin[0] + (y_lin[-1] - y_lin[0]) * np.sin(theta) / 2
        # Scale to actual width
        scale = abs(y_max - y_min) / (y_lin[-1] - y_lin[0])
        contour_y = np.linspace(y_min, y_max, n_pts)

        # Refine: iteratively find the boundary at multiple angles
        refined_x = []
        refined_y = []
        for angle in np.linspace(0, 2 * np.pi, 72):
            dx = math.cos(angle)
            dy = math.sin(angle)
            # Ray march from center
            r = x_contour * 0.1
            c_val = 0.0
            max_r = x_contour * 5
            while r < max_r:
                test_x = x_contour + dx * r
                test_y = dy * r * abs(y_max - y_min) / x_contour
                if test_x < 10:
                    break
                c_val = concentration_at_point(test_x, test_y, z, input) * 1e6
                if abs(c_val - target) / target < 0.05:
                    refined_x.append(test_x)
                    refined_y.append(test_y)
                    break
                r += x_contour * 0.02

            if len(refined_x) == 0:
                # Fallback to simple ellipse
                refined_x.append(x_contour + dx * x_contour * 0.3)
                refined_y.append(dy * (y_max - y_min) / 2)

        results[target] = {
            "x_contour": np.array(refined_x, dtype=np.float64),
            "y_contour": np.array(refined_y, dtype=np.float64),
        }

    return results


# ---------------------------------------------------------------------------
# Helper: max ground-level concentration
# ---------------------------------------------------------------------------


def max_ground_concentration(input: PlumeInput) -> Tuple[float, float]:
    """Find the maximum ground-level concentration and its location.

    Analytical formula for ground-level max:
        C_max = (2 * Q) / (π * e * u * H²) * (σz/σy)
    occurs at distance where σz ≈ H/√2.

    Args:
        input: PlumeInput parameters.

    Returns:
        Tuple of (max_concentration_mgm3, distance_m).

    References:
        Turner, D.B. (1994), Section 2.2.
    """
    H = max(input.release_height, 0.001)  # avoid division by zero
    u = max(input.wind_speed, 0.1)
    Q = input.source_rate

    # Estimate distance x_max where σz ≈ H/√2
    target_sigma_z = H / math.sqrt(2.0)
    x_guess = 100.0  # starting guess
    for _ in range(100):
        sz = sigma_z(x_guess, input.stability_class, input.terrain_type)
        if sz < target_sigma_z:
            x_guess *= 1.5
        else:
            break

    # Refine with binary search
    x_lo, x_hi = x_guess * 0.1, x_guess * 2.0
    for _ in range(50):
        x_mid = (x_lo + x_hi) / 2
        sz = sigma_z(x_mid, input.stability_class, input.terrain_type)
        if sz < target_sigma_z:
            x_lo = x_mid
        else:
            x_hi = x_mid

    x_max = (x_lo + x_hi) / 2.0

    # Evaluate maximum concentration
    sy = sigma_y(x_max, input.stability_class, input.terrain_type)
    sz = sigma_z(x_max, input.stability_class, input.terrain_type)

    # Apply time correction
    if input.sampling_time != input.reference_time:
        sy = sigma_y_corrected(
            x_max, input.stability_class, input.terrain_type,
            input.sampling_time, input.reference_time,
        )

    C_max = (2.0 * Q) / (math.pi * math.e * u * H ** 2) * (sz / sy)

    # Convert to mg/m³
    C_max_mgm3 = C_max * 1e6

    return float(C_max_mgm3), float(x_max)


# ---------------------------------------------------------------------------
# Dispersion Calculator Class
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Jet Momentum Dispersion — Near-Field Corrections
# ---------------------------------------------------------------------------

# Eddy diffusivity constant for jet dispersion (Briggs, Ooms)
_KY_JET = 0.1  # [m²/s per m/s wind speed]
_KZ_JET = 0.01  # [m²/s per m/s wind speed]


def sigma_y_jet(
    x: float,
    d_hole: float,
    wind_speed: float,
    stability_class: StabilityClass,
    terrain_type: TerrainType = "rural",
    jet_velocity: float = 0.0,
    release_density: float = 1.2,
) -> float:
    """Compute sigma_y accounting for jet momentum — REDUCES sigma near source.

    For a momentum-dominated release, the gas jet entrains ambient air slowly,
    keeping the plume narrow in the near-field. This gives SMALLER sigma than
    pure atmospheric dispersion, which is critical for flash fire accuracy.

    Approach: sigma = min(atmospheric_sigma, jet_spread_sigma)
    - jet_spread_sigma grows slowly: sigma_jet ≈ 0.1 * x (typical turbulent jet)
    - atmospheric_sigma grows faster (especially in unstable conditions)
    - In the near-field, jet_spread_sigma limits the cloud width
    - In the far-field, atmospheric dispersion takes over

    Args:
        x: Downwind distance [m].
        d_hole: Release hole diameter [m].
        wind_speed: Wind speed [m/s].
        stability_class: Pasquill-Gifford stability class.
        terrain_type: 'rural' or 'urban'.
        jet_velocity: Jet exit velocity [m/s].
        release_density: Release gas density [kg/m³].

    Returns:
        Effective sigma_y [m].
    """
    if x <= 0:
        return 0.0

    if d_hole <= 0 or jet_velocity <= 0:
        return sigma_y(x, stability_class, terrain_type)

    # Atmospheric sigma
    sy_atm = sigma_y(x, stability_class, terrain_type)

    # Jet spread: for a round turbulent jet, half-width grows as b ≈ 0.1 * x
    # sigma_y ≈ b / sqrt(2) for Gaussian fit to top-hat profile
    # But the jet core persists for L_j ≈ d * (v_jet/u) * sqrt(rho/air)
    # Within the core, the effective sigma is much smaller
    L_j = _jet_decay_length(jet_velocity, d_hole, wind_speed, release_density)

    if L_j > 0 and x < L_j:
        # In the momentum-dominated zone: sigma grows slowly
        # Jet half-width at x: b(x) ≈ d/2 + 0.08 * x  (slow growth)
        b_jet = d_hole / 2.0 + 0.08 * x
        sy_jet = b_jet / math.sqrt(2.0)
        # Take minimum: jet limits the spread
        return min(sy_atm, sy_jet)
    else:
        # Beyond jet decay: atmospheric dispersion dominates
        # But use virtual source correction for smoother transition
        # Effective start distance = where jet sigma = atmospheric sigma
        if L_j > 0:
            b_at_Lj = d_hole / 2.0 + 0.08 * L_j
            sy_at_Lj = b_at_Lj / math.sqrt(2.0)
            # Find x_eff where atmospheric sigma = sy_at_Lj
            # Use sy_at_Lj as initial spread, grow with atmospheric rate
            sy_eff = math.sqrt(sy_at_Lj**2 + (sigma_y(x, stability_class, terrain_type) - sigma_y(L_j, stability_class, terrain_type))**2)
            return max(sy_eff, sy_at_Lj)
        return sy_atm


def sigma_z_jet(
    x: float,
    d_hole: float,
    wind_speed: float,
    stability_class: StabilityClass,
    terrain_type: TerrainType = "rural",
    jet_velocity: float = 0.0,
    release_density: float = 1.2,
) -> float:
    """Compute sigma_z accounting for jet momentum — REDUCES sigma near source.

    Same approach as sigma_y_jet: jet limits vertical spread in near-field.

    Args:
        x: Downwind distance [m].
        d_hole: Release hole diameter [m].
        wind_speed: Wind speed [m/s].
        stability_class: Pasquill-Gifford stability class.
        terrain_type: 'rural' or 'urban'.
        jet_velocity: Jet exit velocity [m/s].
        release_density: Release gas density [kg/m³].

    Returns:
        Effective sigma_z [m].
    """
    if x <= 0:
        return 0.0

    if d_hole <= 0 or jet_velocity <= 0:
        return sigma_z(x, stability_class, terrain_type)

    sz_atm = sigma_z(x, stability_class, terrain_type)

    L_j = _jet_decay_length(jet_velocity, d_hole, wind_speed, release_density)

    if L_j > 0 and x < L_j:
        # Momentum zone: sigma_z grows slowly with jet
        b_jet = d_hole / 2.0 + 0.08 * x
        sz_jet = b_jet / math.sqrt(2.0)
        return min(sz_atm, sz_jet)
    else:
        if L_j > 0:
            b_at_Lj = d_hole / 2.0 + 0.08 * L_j
            sz_at_Lj = b_at_Lj / math.sqrt(2.0)
            sz_eff = math.sqrt(sz_at_Lj**2 + max(0, sigma_z(x, stability_class, terrain_type)**2 - sigma_z(L_j, stability_class, terrain_type)**2))
            return max(sz_eff, sz_at_Lj)
        return sz_atm


def _jet_decay_length(
    jet_velocity: float,
    hole_diameter: float,
    wind_speed: float,
    release_density: float = 1.2,
    air_density: float = 1.2,
) -> float:
    """Compute the jet decay (potential core) length.

    The jet transitions from momentum-dominated to atmosphere-dominated
    dispersion at approximately 5-10 jet decay lengths. This is the
    distance where the jet-to-wind velocity ratio drops below ~1.

    Simplified formula based on Chu & Lee (1996) and Wood (1993):
        L_j = d_hole * (jet_velocity / u) * √(ρ_release / ρ_air)

    Args:
        jet_velocity: Exit velocity of the jet [m/s].
        hole_diameter: Release hole diameter [m].
        wind_speed: Cross-wind speed [m/s].
        release_density: Density of released gas [kg/m³].
        air_density: Density of ambient air [kg/m³].

    Returns:
        Jet decay length [m].
    """
    if jet_velocity <= 0 or hole_diameter <= 0 or wind_speed <= 0:
        return 0.0

    density_ratio = math.sqrt(max(release_density / air_density, 0.001))
    return hole_diameter * (jet_velocity / wind_speed) * density_ratio


# ---------------------------------------------------------------------------
# Jet-Enhanced Plume Calculation
# ---------------------------------------------------------------------------


def calculate_plume_with_jet(
    input: PlumeInput,
    jet_velocity: float,
    hole_diameter: float,
    release_density: float = 1.2,
) -> PlumeResult:
    """Calculate 3D concentration grid using jet-enhanced dispersion.

    For near-field distances (x < 10 × jet_decay_length), uses jet-momentum
    dispersion coefficients. Beyond that, transitions to standard atmospheric
    Gaussian dispersion.

    The transition is smooth: at each x, we compute a blending weight
    based on the ratio x / (10 * L_j).

    Args:
        input: PlumeInput with release and grid parameters.
        jet_velocity: Exit velocity of the jet at the hole [m/s].
        hole_diameter: Release hole diameter [m].
        release_density: Density of the released gas [kg/m³] (default air=1.2).

    Returns:
        PlumeResult with jet-enhanced concentration grid.

    Examples:
        >>> inp = PlumeInput(
        ...     source_rate=2.0, wind_speed=3.0, stability_class='D',
        ...     release_height=0.0,
        ...     grid_x_range=(10, 5000, 50),
        ...     grid_y_range=(-200, 200, 21),
        ...     grid_z_range=(0, 50, 11),
        ... )
        >>> result = calculate_plume_with_jet(inp, jet_velocity=50.0,
        ...                                    hole_diameter=0.05)
        >>> result.max_concentration > 0
        True
    """
    # Compute jet decay length
    L_j = _jet_decay_length(
        jet_velocity=jet_velocity,
        hole_diameter=hole_diameter,
        wind_speed=input.wind_speed,
        release_density=release_density,
    )
    jet_transition_distance = 10.0 * L_j if L_j > 0 else 0.0

    # Create coordinate arrays
    x_min, x_max, n_x = input.grid_x_range
    y_min, y_max, n_y = input.grid_y_range
    z_min, z_max, n_z = input.grid_z_range

    x_coords = np.linspace(x_min, x_max, n_x)
    y_coords = np.linspace(y_min, y_max, n_y)
    z_coords = np.linspace(z_min, z_max, n_z)

    # Atmospheric sigma (standard)
    sy_atm = np.zeros(n_x)
    sz_atm = np.zeros(n_x)

    # Jet-enhanced sigma
    sy_jet = np.zeros(n_x)
    sz_jet = np.zeros(n_x)

    u = max(input.wind_speed, 0.1)

    for i, x in enumerate(x_coords):
        # Standard atmospheric dispersion coefficients
        sy_atm[i] = sigma_y(x, input.stability_class, input.terrain_type)
        sz_atm[i] = sigma_z(x, input.stability_class, input.terrain_type)

        # Apply sampling time correction
        if input.sampling_time != input.reference_time:
            sy_atm[i] = sigma_y_corrected(
                x, input.stability_class, input.terrain_type,
                input.sampling_time, input.reference_time,
            )

        # Apply building wake correction to atmospheric sigma
        if input.building is not None and input.building.height > 0:
            sy_atm[i], sz_atm[i] = building_wake_correction(x, sy_atm[i], sz_atm[i], input.building)

        # Jet-enhanced dispersion coefficients
        if hole_diameter > 0 and jet_velocity > 0:
            sy_jet[i] = sigma_y_jet(
                x, hole_diameter, u,
                input.stability_class, input.terrain_type,
                jet_velocity, release_density,
            )
            sz_jet[i] = sigma_z_jet(
                x, hole_diameter, u,
                input.stability_class, input.terrain_type,
                jet_velocity, release_density,
            )
        else:
            sy_jet[i] = sy_atm[i]
            sz_jet[i] = sz_atm[i]

    # Initialize grids
    C_grid = np.zeros((n_x, n_y, n_z), dtype=np.float64)
    ground_C = np.zeros((n_x, n_y), dtype=np.float64)
    centerline_C = np.zeros(n_x, dtype=np.float64)

    H = input.release_height
    Q = input.source_rate
    two_pi_u = 2.0 * math.pi * u

    y_sq = y_coords ** 2
    z_minus_H_sq = (z_coords - H) ** 2
    z_plus_H_sq = (z_coords + H) ** 2

    for i in range(n_x):
        x = x_coords[i]

        # Blend standard and jet sigma based on distance
        if jet_transition_distance > 0 and x < jet_transition_distance:
            # Smooth transition: weight goes from 1 (pure jet) at x=0
            # to 0 (pure atmospheric) at x = jet_transition_distance
            blend = 1.0 - (x / jet_transition_distance)
            # Use a smooth sigmoid-like transition (cosine blend)
            blend_smooth = 0.5 * (1.0 + math.cos(math.pi * (1.0 - blend)))

            sy = blend_smooth * sy_jet[i] + (1.0 - blend_smooth) * sy_atm[i]
            sz = blend_smooth * sz_jet[i] + (1.0 - blend_smooth) * sz_atm[i]
        else:
            sy = sy_atm[i]
            sz = sz_atm[i]

        if sy <= 0 or sz <= 0:
            continue

        base_factor = Q / (two_pi_u * sy * sz)
        lat_term = np.exp(-0.5 * y_sq / (sy ** 2))

        if sz > 0:
            term1 = np.exp(-0.5 * z_minus_H_sq / (sz ** 2))
            term2 = np.exp(-0.5 * z_plus_H_sq / (sz ** 2))
            vert_term = term1 + term2
        else:
            vert_term = np.zeros(n_z)

        C_2d = base_factor * np.outer(lat_term, vert_term)

        # Chemical decay
        if input.decay_rate > 0:
            travel_time = x / u
            C_2d *= math.exp(-input.decay_rate * travel_time)

        # Dry deposition
        if input.deposition_velocity > 0 and x > 100.0:
            mix_scale = max(sz, 1.0)
            depletion_factor = math.exp(
                -math.sqrt(2.0 / math.pi)
                * input.deposition_velocity
                * x
                / (u * mix_scale)
            )
            C_2d *= depletion_factor

        C_grid[i, :, :] = C_2d
        ground_C[i, :] = C_2d[:, 0]

        y_mid = n_y // 2
        z_mid = 0
        centerline_C[i] = C_2d[y_mid, z_mid]

    # Convert kg/m³ → mg/m³
    C_grid_mgm3 = C_grid * 1e6
    ground_C_mgm3 = ground_C * 1e6
    centerline_C_mgm3 = centerline_C * 1e6

    max_flat_idx = np.argmax(C_grid)
    max_idx = np.unravel_index(max_flat_idx, C_grid.shape)
    max_C = C_grid_mgm3[max_idx]
    max_x = x_coords[max_idx[0]]
    max_y = y_coords[max_idx[1]]
    max_z = z_coords[max_idx[2]]

    return PlumeResult(
        concentration_grid=C_grid_mgm3,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        max_concentration=float(max_C),
        max_distance=float(max_x),
        max_y_distance=float(max_y),
        max_z_height=float(max_z),
        centerline_concentration=centerline_C_mgm3,
        ground_concentration=ground_C_mgm3,
        input=input,
        plume_rise_delta=0.0,
    )


# ---------------------------------------------------------------------------
# Distance-to-Concentration Finder (Binary Search)
# ---------------------------------------------------------------------------


def find_distance_to_concentration(
    target_concentration: float,
    input: PlumeInput,
    jet_velocity: float = 0.0,
    hole_diameter: float = 0.0,
    release_density: float = 1.2,
    z: float = 0.0,
    y: float = 0.0,
    x_min: float = 0.1,
    x_max: float = 10000.0,
    tolerance: float = 0.01,
    max_iterations: int = 100,
) -> Optional[float]:
    """Find downwind distance where centerline concentration equals target.

    Uses binary search to locate the distance x where:
        C(x, y=0, z=0) ≈ target_concentration [mg/m³]

    This is critical for calculating flash-fire distances (50% LFL),
    toxic endpoint distances (ERPG), and flammable cloud dimensions.

    For ground-level releases (H=0), concentration decreases monotonically
    with distance, so a single binary search works. For elevated releases,
    the search finds the first (near-field) crossing.

    When jet parameters are provided, uses the jet-enhanced dispersion
    model for better near-field accuracy.

    Args:
        target_concentration: Target concentration [mg/m³].
        input: PlumeInput parameters.
        jet_velocity: Exit velocity of jet [m/s] (0 = no jet enhancement).
        hole_diameter: Release hole diameter [m] (0 = no jet enhancement).
        release_density: Density of released gas [kg/m³].
        z: Height above ground [m] (default 0 = ground level).
        y: Cross-wind offset [m] (default 0 = centerline).
        x_min: Lower bound for search [m].
        x_max: Upper bound for search [m].
        tolerance: Relative tolerance for convergence.
        max_iterations: Maximum binary search iterations.

    Returns:
        Distance to target concentration [m], or None if not found.

    Notes:
        - For ground-level releases, the ground-level centerline
          concentration decreases monotonically, so binary search
          is deterministic.
        - For elevated releases, concentration increases to a peak
          before decreasing. The search finds the crossing on the
          near-field side (shorter distance) first.

    Examples:
        >>> inp = PlumeInput(source_rate=1.0, wind_speed=5.0,
        ...                  stability_class='D', release_height=0.0)
        >>> dist = find_distance_to_concentration(10.0, inp)
        >>> dist is not None and dist > 0
        True
    """
    if target_concentration <= 0:
        return None

    def _conc_at_x(x_val: float) -> float:
        """Compute centerline concentration at distance x [mg/m³]."""
        if x_val <= 0:
            return float('inf')

        u = max(input.wind_speed, 0.1)
        H = input.release_height
        Q = input.source_rate

        # Get dispersion coefficients (with or without jet enhancement)
        if hole_diameter > 0 and jet_velocity > 0:
            sy = sigma_y_jet(
                x_val, hole_diameter, u,
                input.stability_class, input.terrain_type,
                jet_velocity, release_density,
            )
            sz = sigma_z_jet(
                x_val, hole_diameter, u,
                input.stability_class, input.terrain_type,
                jet_velocity, release_density,
            )

            # Apply transition blending
            L_j = _jet_decay_length(
                jet_velocity, hole_diameter, u, release_density,
            )
            jet_transition = 10.0 * L_j
            if jet_transition > 0 and x_val < jet_transition:
                # Compute atmospheric sigma for blending
                sy_atm = sigma_y(
                    x_val, input.stability_class, input.terrain_type,
                )
                sz_atm = sigma_z(
                    x_val, input.stability_class, input.terrain_type,
                )
                blend = 1.0 - (x_val / jet_transition)
                blend_smooth = 0.5 * (1.0 + math.cos(math.pi * (1.0 - blend)))
                sy = blend_smooth * sy + (1.0 - blend_smooth) * sy_atm
                sz = blend_smooth * sz + (1.0 - blend_smooth) * sz_atm
        else:
            sy = sigma_y(x_val, input.stability_class, input.terrain_type)
            sz = sigma_z(x_val, input.stability_class, input.terrain_type)

        if input.sampling_time != input.reference_time:
            # Apply sampling time correction ONLY to atmospheric component
            sy_atm = sigma_y_corrected(
                x_val, input.stability_class, input.terrain_type,
                input.sampling_time, input.reference_time,
            )
            # If using jet sigma and blended, re-blend with corrected atmospheric
            if hole_diameter > 0 and jet_velocity > 0 and x_val < jet_transition:
                sy = blend_smooth * sy + (1.0 - blend_smooth) * sy_atm

        if sy <= 0 or sz <= 0:
            return 0.0

        # Concentration at (x, y=0, z)
        C = Q / (2.0 * math.pi * u * sy * sz)
        C *= math.exp(-0.5 * (y / sy) ** 2)

        if sz > 0:
            term1 = math.exp(-0.5 * ((z - H) / sz) ** 2)
            term2 = math.exp(-0.5 * ((z + H) / sz) ** 2)
            C *= (term1 + term2)

        if input.decay_rate > 0:
            C *= math.exp(-input.decay_rate * x_val / u)

        # Convert to mg/m³
        return C * 1e6

    # Evaluate endpoints to ensure target is within range
    c_min = _conc_at_x(x_min)
    c_max = _conc_at_x(x_max)

    if target_concentration > c_min:
        # Target is higher than the near-field concentration →
        # target distance is less than x_min (but x_min is our floor)
        return x_min if c_min > 0 else None

    if target_concentration <= c_max:
        # Target is lower than far-field → target is beyond x_max
        return x_max

    # Binary search
    x_lo, x_hi = x_min, x_max
    for _ in range(max_iterations):
        x_mid = (x_lo + x_hi) / 2.0
        c_mid = _conc_at_x(x_mid)

        if abs(c_mid - target_concentration) / target_concentration < tolerance:
            return x_mid

        if c_mid > target_concentration:
            x_lo = x_mid
        else:
            x_hi = x_mid

    return (x_lo + x_hi) / 2.0


# ---------------------------------------------------------------------------
# Flash Fire Distance
# ---------------------------------------------------------------------------


def calculate_flash_fire_distance(
    source_rate: float,
    wind_speed: float,
    stability_class: StabilityClass,
    lfl: float,
    lfl_fraction: float = 0.5,
    hole_diameter: float = 0.0,
    jet_velocity: float = 0.0,
    release_density: float = 1.2,
    release_height: float = 0.0,
    temperature: float = 298.15,
    pressure: float = P_ATM,
    molecular_weight: float = 29.0,
    terrain_type: TerrainType = "rural",
    inventory_volume_m3: float = 0.0,
    release_duration_s: float = 0.0,
    sampling_time: float = 600.0,
) -> Optional[float]:
    """Calculate flash-fire hazard distance to a target LFL fraction.

    Flash fire distances are determined by the distance where the
    centerline ground-level concentration equals a fraction of the
    Lower Flammability Limit (LFL). Common fractions:
        - 100% LFL: flammable cloud boundary
        - 50% LFL: standard flash fire endpoint (CCPS, TNO Yellow Book)

    Uses the jet-enhanced Gaussian plume model for improved near-field
    accuracy compared to standard atmospheric dispersion. This
    significantly reduces overestimation at short distances.

    When inventory_volume_m3 or release_duration_s is provided, delegates
    to calculate_flash_fire_distance_v2 which accounts for:
        - Inventory depletion (reduces effective release rate)
        - Time averaging (short releases spread over sampling period)
        - Jet momentum near-field dilution (rapid concentration drop)

    These corrections are essential for matching PHAST results on
    inventory-limited releases (e.g., fullbore ruptures of ISO containers).

    Args:
        source_rate: Continuous release rate Q [kg/s].
        wind_speed: Mean wind speed at release height [m/s].
        stability_class: Pasquill-Gifford stability class ('A'-'F').
        lfl: Lower Flammability Limit [kg/m³] (NOT volume fraction!).
            Convert from vol%: LFL[kg/m³] = LFL[vol%] * MW / (100 * 24.45)
            where 24.45 L/mol at 25°C, 1 atm — or more precisely:
            LFL[kg/m³] = LFL_frac * (P * MW) / (R * T)
        lfl_fraction: Fraction of LFL for endpoint (default 0.5 = 50%).
        hole_diameter: Release hole/rupture diameter [m] (for jet model).
        jet_velocity: Exit velocity of the release [m/s] (for jet model).
        release_density: Density of released gas [kg/m³] at release conditions.
        release_height: Effective release height [m] (default 0 = ground).
        temperature: Ambient temperature [K].
        pressure: Ambient pressure [Pa].
        molecular_weight: Molecular weight of released gas [g/mol].
        terrain_type: 'rural' or 'urban'.
        inventory_volume_m3: Inventory volume [m³]. When > 0, uses V2 model
            with inventory depletion & time averaging.
        release_duration_s: Known release duration [s]. When > 0, enables
            time-averaged concentration.
        sampling_time: Reference sampling period [s] (default 600 s = 10 min).

    Returns:
        Distance to flash fire endpoint [m], or None if calculation fails.

    Notes:
        - Flash fire ignores delayed ignition scenarios (UVCE).
        - The model assumes continuous release at steady state.
        - For a more conservative assessment, use stability class 'F'
          (stable) which gives the longest distances.
        - PHAST UDM uses a similar approach but with more sophisticated
          near-field jet and heavy-gas corrections.

    Examples:
        >>> # Methane: LFL = 5 vol% ≈ 0.033 kg/m³, 50% LFL for endpoint
        >>> # 10 kg/s release, 5 m/s wind, stability D, 0.1 m hole, 50 m/s jet
        >>> dist = calculate_flash_fire_distance(
        ...     source_rate=10.0, wind_speed=5.0, stability_class='D',
        ...     lfl=0.033, lfl_fraction=0.5,
        ...     hole_diameter=0.1, jet_velocity=50.0,
        ... )
        >>> dist is not None
        True
        >>> # V2: ISO container fullbore rupture
        >>> dist_v2 = calculate_flash_fire_distance(
        ...     source_rate=108.0, wind_speed=5.0, stability_class='D',
        ...     lfl=0.033, lfl_fraction=0.5,
        ...     hole_diameter=0.2, jet_velocity=300.0,
        ...     release_density=30.0, inventory_volume_m3=0.564,
        ... )
        >>> dist_v2 is not None
        True
    """
    # Delegate to V2 when inventory data is available
    if inventory_volume_m3 > 0 or release_duration_s > 0:
        return calculate_flash_fire_distance_v2(
            source_rate=source_rate,
            wind_speed=wind_speed,
            stability_class=stability_class,
            lfl=lfl,
            lfl_fraction=lfl_fraction,
            hole_diameter=hole_diameter,
            jet_velocity=jet_velocity,
            release_density=release_density,
            release_height=release_height,
            temperature=temperature,
            pressure=pressure,
            molecular_weight=molecular_weight,
            terrain_type=terrain_type,
            inventory_volume_m3=inventory_volume_m3,
            release_duration_s=release_duration_s,
            sampling_time=sampling_time,
        )

    # Original method: steady-state with jet-enhanced sigma only
    # Build PlumeInput
    input = PlumeInput(
        source_rate=source_rate,
        wind_speed=wind_speed,
        stability_class=stability_class,
        release_height=release_height,
        terrain_type=terrain_type,
        temperature=temperature,
        pressure=pressure,
        molecular_weight=molecular_weight,
    )

    # Target concentration = LFL fraction × LFL [kg/m³ → mg/m³]
    target_kg_m3 = lfl * lfl_fraction
    target_mgm3 = target_kg_m3 * 1e6

    # Use the binary search distance finder with jet enhancement
    return find_distance_to_concentration(
        target_concentration=target_mgm3,
        input=input,
        jet_velocity=jet_velocity,
        hole_diameter=hole_diameter,
        release_density=release_density,
        x_min=0.05,
        x_max=20000.0,
    )


# ---------------------------------------------------------------------------
# Jet Dilution Factor — Near-Field Momentum Dilution
# ---------------------------------------------------------------------------


def jet_dilution_factor(
    x: float,
    hole_diameter: float,
    jet_velocity: float = 0.0,
    wind_speed: float = 1.0,
    k: float = 5.5,
) -> float:
    """Compute the jet dilution factor for near-field concentration correction.

    For a choked gas release at high velocity through a hole, the jet expands
    rapidly and entrains ambient air. In the first few meters, concentration
    drops by orders of magnitude due to air entrainment.

    Based on the classic round turbulent jet theory:
        C(x) / C0 ≈ (d / x) * K

    where:
        C(x) = concentration at distance x
        C0 = initial concentration at the source
        d = hole diameter
        K = entrainment constant (~5-6 for turbulent gas jets, ~10 for liquid jets)

    This gives:
        - At x = 10*d (e.g., 2 m for 200 mm hole): C/C0 ≈ 55%
        - At x = 50*d (e.g., 10 m for 200 mm hole): C/C0 ≈ 11%
        - At x = 100*d (e.g., 20 m for 200 mm hole): C/C0 ≈ 5.5%

    Beyond the jet core region (~20*d to 30*d), atmospheric dispersion
    dominates and this factor approaches 1.0.

    The factor is clamped to [0.01, 1.0] to prevent unrealistic values.

    Args:
        x: Downwind distance from release [m].
        hole_diameter: Release hole/rupture diameter [m].
        jet_velocity: Exit velocity of the jet [m/s] (unused, included for API
            compatibility).
        wind_speed: Wind speed [m/s] (unused, included for API compatibility).
        k: Jet entrainment constant (default 5.5 for turbulent gas jets).

    Returns:
        Dilution factor in [0.01, 1.0]. Multiply Gaussian plume concentration
        by this factor to account for near-field jet dilution.

    References:
        - Rajaratnam, N. (1976). Turbulent Jets. Elsevier.
        - Chen, C.J. & Rodi, W. (1980). Vertical Turbulent Buoyant Jets.
        - TNO Yellow Book, Chapter 4.5 (Jet Releases).

    Examples:
        >>> # 200 mm hole at 2 m distance → factor ≈ 0.55
        >>> factor = jet_dilution_factor(2.0, 0.2)
        >>> 0.4 < factor < 0.7
        True
        >>> # 200 mm hole at 20 m distance → factor ≈ 0.055
        >>> factor = jet_dilution_factor(20.0, 0.2)
        >>> 0.03 < factor < 0.08
        True
        >>> # Far-field: factor should be small for large jet, approaches 1.0 very far out
        >>> factor = jet_dilution_factor(200.0, 0.2)
        >>> 0.0 < factor <= 1.0
        True
        >>> # Small jet dissipates quickly: 10mm hole at 50m → factor ≈ 1.0
        >>> factor = jet_dilution_factor(50.0, 0.01)
        >>> factor > 0.9
        True
    """
    if x <= 0 or hole_diameter <= 0:
        return 1.0

    # Jet core length: region where initial momentum dominates
    # Typically ~20*d for turbulent round jets
    jet_core_length = 20.0 * hole_diameter

    if x <= hole_diameter:
        # Inside the hole itself: no dilution
        return 1.0

    # Near-field jet dilution: C/C0 = (d / x) * K
    # This is the classic round jet entrainment formula
    jet_ratio = (hole_diameter / x) * k

    # Transition zone (between jet core and far-field atmospheric dispersion):
    # The Gaussian plume model already accounts for atmospheric dispersion.
    # The jet dilution is an ADDITIONAL near-field effect accounting for
    # rapid momentum-driven mixing that the Gaussian model doesn't capture.
    #
    # Therefore, the jet dilution factor should approach 1.0 (no correction)
    # once we exit the momentum-dominated zone. The transition is smooth
    # over a long distance (up to ~50 × jet_core_length) to avoid sharp
    # discontinuities.
    if x > jet_core_length:
        # Beyond jet core: blend toward no correction over a long distance
        # Use 50 * jet_core_length as the full transition distance
        transition_distance = 50.0 * jet_core_length
        transition_ratio = min(1.0, (x - jet_core_length) / transition_distance)
        blend = 1.0 - transition_ratio
        # Use smooth cosine blend for nicer transition
        blend = 0.5 * (1.0 + math.cos(math.pi * (1.0 - blend)))
        dilution = blend * jet_ratio + (1.0 - blend) * 1.0
    else:
        dilution = jet_ratio

    # Clamp to reasonable range
    return max(0.01, min(1.0, dilution))


# ---------------------------------------------------------------------------
# Time-Averaged Concentration
# ---------------------------------------------------------------------------


def calculate_time_averaged_concentration(
    source_rate: float,
    duration_s: float,
    wind_speed: float,
    stability_class: StabilityClass,
    distance: float,
    hole_diameter: float = 0.0,
    jet_velocity: float = 0.0,
    release_density: float = 1.2,
    release_height: float = 0.0,
    temperature: float = 298.15,
    pressure: float = P_ATM,
    molecular_weight: float = 29.0,
    terrain_type: TerrainType = "rural",
    sampling_time: float = 600.0,
) -> float:
    """Calculate time-averaged concentration for a finite-duration release.

    For short-duration releases (e.g., inventory-limited fullbore ruptures),
    the cloud passes over a receptor quickly. The time-averaged concentration
    is lower than the instantaneous steady-state prediction because the
    cloud occupies only a fraction of the sampling period.

    PHAST and other advanced models account for this by integrating the
    time-varying release rate over the sampling time.

    Formula:
        C_avg = C_instantaneous * min(1.0, duration / sampling_time)

    where:
        - C_instantaneous is the steady-state Gaussian plume concentration
          at the given distance (using the full source_rate)
        - sampling_time is typically 600 s (10 min) per standard practice

    For releases shorter than sampling_time, the average concentration is
    reduced proportionally. For example, a 0.17 s release gives:
        C_avg = C_instantaneous * (0.17 / 600) = C_instantaneous * 0.000283

    This dramatically reduces the predicted concentration (and thus the
    flash-fire distance) for inventory-limited releases, matching PHAST's
    time-varying release model.

    Args:
        source_rate: Continuous release rate Q [kg/s].
        duration_s: Actual release duration [s]. For fullbore ruptures,
            this is inventory_mass / source_rate.
        wind_speed: Mean wind speed at release height [m/s].
        stability_class: Pasquill-Gifford stability class.
        distance: Downwind distance to evaluate concentration [m].
        hole_diameter: Release hole diameter [m] (for jet enhancement).
        jet_velocity: Exit velocity of jet [m/s] (for jet enhancement).
        release_density: Density of released gas [kg/m³].
        release_height: Effective release height [m].
        temperature: Ambient temperature [K].
        pressure: Ambient pressure [Pa].
        molecular_weight: Molecular weight [g/mol].
        terrain_type: 'rural' or 'urban'.
        sampling_time: Reference sampling/averaging time [s] (default 600 s).

    Returns:
        Time-averaged concentration [mg/m³].

    Examples:
        >>> # Long-duration release: no averaging effect
        >>> c = calculate_time_averaged_concentration(
        ...     source_rate=1.0, duration_s=3600.0,
        ...     wind_speed=5.0, stability_class='D', distance=100.0,
        ... )
        >>> c > 0
        True
    """
    if duration_s <= 0 or distance <= 0:
        return 0.0

    # Build PlumeInput for instantaneous concentration
    input = PlumeInput(
        source_rate=source_rate,
        wind_speed=wind_speed,
        stability_class=stability_class,
        release_height=release_height,
        terrain_type=terrain_type,
        temperature=temperature,
        pressure=pressure,
        molecular_weight=molecular_weight,
    )

    # Calculate instantaneous concentration at (distance, y=0, z=0)
    if hole_diameter > 0 and jet_velocity > 0:
        # Use jet-enhanced concentration
        c_instant = _concentration_at_centerline_with_jet(
            x=distance, input=input,
            hole_diameter=hole_diameter,
            jet_velocity=jet_velocity,
            release_density=release_density,
        )
    else:
        c_instant = concentration_at_point(distance, 0.0, 0.0, input)

    # Convert kg/m³ → mg/m³
    c_instant_mgm3 = c_instant * 1e6

    # Apply jet dilution factor for near-field momentum dilution
    if hole_diameter > 0:
        dilution = jet_dilution_factor(distance, hole_diameter)
        c_instant_mgm3 *= dilution

    # Time averaging: scale by duration / sampling_time
    time_scale = min(1.0, duration_s / sampling_time)

    return c_instant_mgm3 * time_scale


def _concentration_at_centerline_with_jet(
    x: float,
    input: PlumeInput,
    hole_diameter: float,
    jet_velocity: float,
    release_density: float = 1.2,
) -> float:
    """Internal: compute centerline concentration at (x, y=0, z=0) with jet enhancement.

    Args:
        x: Downwind distance [m].
        input: PlumeInput parameters.
        hole_diameter: Release hole diameter [m].
        jet_velocity: Exit velocity of jet [m/s].
        release_density: Density of released gas [kg/m³].

    Returns:
        Concentration [kg/m³].
    """
    if x <= 0:
        return 0.0

    u = max(input.wind_speed, 0.1)
    H = input.release_height
    Q = input.source_rate

    # Jet-enhanced sigma
    sy = sigma_y_jet(
        x, hole_diameter, u,
        input.stability_class, input.terrain_type,
        jet_velocity, release_density,
    )
    sz = sigma_z_jet(
        x, hole_diameter, u,
        input.stability_class, input.terrain_type,
        jet_velocity, release_density,
    )

    # Blend with atmospheric sigma based on proximity to source
    L_j = _jet_decay_length(jet_velocity, hole_diameter, u, release_density)
    jet_transition = 10.0 * L_j
    if jet_transition > 0 and x < jet_transition:
        sy_atm = sigma_y(x, input.stability_class, input.terrain_type)
        sz_atm = sigma_z(x, input.stability_class, input.terrain_type)
        blend = 1.0 - (x / jet_transition)
        blend_smooth = 0.5 * (1.0 + math.cos(math.pi * (1.0 - blend)))
        sy = blend_smooth * sy + (1.0 - blend_smooth) * sy_atm
        sz = blend_smooth * sz + (1.0 - blend_smooth) * sz_atm

    # Sampling time correction
    if input.sampling_time != input.reference_time:
        sy_atm = sigma_y_corrected(
            x, input.stability_class, input.terrain_type,
            input.sampling_time, input.reference_time,
        )
        if x < jet_transition:
            sy = blend_smooth * sy + (1.0 - blend_smooth) * sy_atm

    if sy <= 0 or sz <= 0:
        return 0.0

    C = Q / (2.0 * math.pi * u * sy * sz)

    # Lateral: y=0 → exp(0) = 1
    # Vertical: z=0, ground reflection
    if sz > 0:
        term1 = math.exp(-0.5 * ((0.0 - H) / sz) ** 2)
        term2 = math.exp(-0.5 * ((0.0 + H) / sz) ** 2)
        C *= (term1 + term2)

    if input.decay_rate > 0:
        C *= math.exp(-input.decay_rate * x / u)

    return max(0.0, C)


# ---------------------------------------------------------------------------
# Flash Fire Distance V2 — With Inventory & Duration
# ---------------------------------------------------------------------------


def calculate_flash_fire_distance_v2(
    source_rate: float,
    wind_speed: float,
    stability_class: StabilityClass,
    lfl: float,
    lfl_fraction: float = 0.5,
    hole_diameter: float = 0.0,
    jet_velocity: float = 0.0,
    release_density: float = 1.2,
    release_height: float = 0.0,
    temperature: float = 298.15,
    pressure: float = P_ATM,
    molecular_weight: float = 29.0,
    terrain_type: TerrainType = "rural",
    inventory_volume_m3: float = 0.0,
    release_duration_s: float = 0.0,
    sampling_time: float = 600.0,
) -> Optional[float]:
    """Calculate flash-fire distance with time-varying release & inventory depletion.

    This V2 implementation accounts for three key physics missing from the
    simple Gaussian plume approach:

    1. **Inventory Depletion**: For a fullbore rupture with limited inventory
       (e.g., ISO container with 0.564 m³), the release lasts only a fraction
       of a second. The effective average release rate is much lower than the
       initial rate because total mass is capped by inventory.

    2. **Time Averaging**: Short-duration releases pass over receptors quickly.
       The time-averaged concentration over a 10-minute sampling period is
       proportionally lower: C_avg = C_inst × (t_release / 600s).

    3. **Jet Dilution**: For high-velocity choked gas releases, near-field
       jet entrainment rapidly dilutes concentration by orders of magnitude.
       C(x)/C0 ≈ (d/x) × 5.5 for round turbulent jets.

    Algorithm:
        a. Calculate inventory mass: m_inv = volume × gas density (at conditions)
        b. If duration given, use it; else estimate: t_release = m_inv / source_rate
        c. Effective avg rate: mdot_eff = min(source_rate, m_inv / max(t_release, 0.01))
        d. Calculate dispersion with mdot_eff instead of source_rate
        e. Apply time-averaging factor: C_avg = C_inst × min(1, t_release / 600s)
        f. Apply jet dilution factor: C_eff = C_avg × (d/x) × K
        g. Binary search for distance where C_eff equals target concentration

    Args:
        source_rate: Initial (peak) release rate Q [kg/s].
        wind_speed: Mean wind speed at release height [m/s].
        stability_class: Pasquill-Gifford stability class.
        lfl: Lower Flammability Limit [kg/m³].
        lfl_fraction: Fraction of LFL for endpoint (default 0.5 = 50% LFL).
        hole_diameter: Release hole/rupture diameter [m].
        jet_velocity: Exit velocity of the release [m/s].
        release_density: Density of released gas [kg/m³] at release conditions.
        release_height: Effective release height [m].
        temperature: Ambient temperature [K].
        pressure: Ambient pressure [Pa].
        molecular_weight: Molecular weight [g/mol].
        terrain_type: 'rural' or 'urban'.
        inventory_volume_m3: Inventory volume [m³]. When > 0, the release is
            treated as inventory-limited. Effective mass = volume × density.
        release_duration_s: Known release duration [s]. If 0 and inventory
            is given, calculated as: inventory_mass / source_rate.
        sampling_time: Reference sampling/averaging time [s] (default 600 s).

    Returns:
        Distance to flash fire endpoint [m], or None if calculation fails.

    Examples:
        >>> # ISO container fullbore rupture: 200mm hole, 30 barg, methane
        >>> # source_rate=108 kg/s, inventory=0.564 m³, density~30 kg/m³
        >>> dist = calculate_flash_fire_distance_v2(
        ...     source_rate=108.0, wind_speed=5.0, stability_class='D',
        ...     lfl=0.033, lfl_fraction=0.5,
        ...     hole_diameter=0.2, jet_velocity=300.0,
        ...     release_density=30.0, inventory_volume_m3=0.564,
        ... )
        >>> dist is not None
        True
    """
    if source_rate <= 0 or lfl <= 0:
        return None

    # Step 1: Determine effective release duration and average rate
    effective_rate = source_rate
    effective_duration = release_duration_s

    if inventory_volume_m3 > 0:
        # Inventory-limited release
        inventory_mass = inventory_volume_m3 * release_density  # kg

        if effective_duration <= 0:
            # Calculate duration from inventory and rate
            effective_duration = inventory_mass / source_rate

        # Effective average rate: capped by total mass over duration
        # mdot_eff = min(initial rate, total mass / duration)
        # For very short releases (fraction of a second), the effective rate
        # equals the total inventory divided by the actual duration
        effective_rate = min(
            source_rate,
            inventory_mass / max(effective_duration, 0.001)
        )
    elif effective_duration > 0:
        # Duration specified without inventory: use time-averaging only
        pass
    else:
        # No inventory data: fall through to standard calculation
        pass

    # Step 2: Build PlumeInput with effective (reduced) release rate
    input = PlumeInput(
        source_rate=effective_rate,
        wind_speed=wind_speed,
        stability_class=stability_class,
        release_height=release_height,
        terrain_type=terrain_type,
        temperature=temperature,
        pressure=pressure,
        molecular_weight=molecular_weight,
    )

    # Step 3: Target concentration in mg/m³
    target_kg_m3 = lfl * lfl_fraction
    target_mgm3 = target_kg_m3 * 1e6

    # Step 4: Define the concentration-at-distance function with all corrections
    def _conc_with_corrections(x_val: float) -> float:
        """Compute corrected concentration at distance x [mg/m³]."""
        if x_val <= 0:
            return float('inf')

        # Base concentration (with jet-enhanced sigma if applicable)
        if hole_diameter > 0 and jet_velocity > 0:
            c_kgm3 = _concentration_at_centerline_with_jet(
                x=x_val, input=input,
                hole_diameter=hole_diameter,
                jet_velocity=jet_velocity,
                release_density=release_density,
            )
        else:
            c_kgm3 = concentration_at_point(x_val, 0.0, 0.0, input)

        c_mgm3 = c_kgm3 * 1e6

        # Apply jet dilution factor (near-field momentum entrainment)
        if hole_diameter > 0:
            jdf = jet_dilution_factor(x_val, hole_diameter)
            c_mgm3 *= jdf

        # Apply time averaging for short-duration releases
        if effective_duration > 0:
            time_scale = min(1.0, effective_duration / sampling_time)
            c_mgm3 *= time_scale

        return c_mgm3

    # Step 5: Binary search for distance
    x_lo = 0.05
    x_hi = 20000.0

    # Evaluate endpoints
    c_min = _conc_with_corrections(x_lo)
    c_max = _conc_with_corrections(x_hi)

    if target_mgm3 > c_min:
        # Target above maximum concentration (near-field still too low)
        return x_lo if c_min > 0 else None

    if target_mgm3 <= c_max:
        # Target beyond far-field range
        return x_hi

    # Binary search with tolerance
    tolerance = 0.01
    max_iterations = 100

    for _ in range(max_iterations):
        x_mid = (x_lo + x_hi) / 2.0
        c_mid = _conc_with_corrections(x_mid)

        if abs(c_mid - target_mgm3) / target_mgm3 < tolerance:
            return x_mid

        if c_mid > target_mgm3:
            x_lo = x_mid
        else:
            x_hi = x_mid

    return (x_lo + x_hi) / 2.0


# ---------------------------------------------------------------------------
# Flash Fire Distance — Updated with V2 Support
# ---------------------------------------------------------------------------


class GaussianPlumeCalculator:
    """Main calculator object for Gaussian plume dispersion.

    Usage:
        calc = GaussianPlumeCalculator()
        result = calc.calculate(plume_input)
        single_C = calc.concentration_at(500, 10, 0, plume_input)
    """

    def calculate(self, input: PlumeInput) -> PlumeResult:
        """Run full Gaussian plume dispersion calculation.

        Args:
            input: Complete PlumeInput with grid specification.

        Returns:
            PlumeResult with concentration grid and metadata.
        """
        return calculate_plume(input)

    def concentration_at(
        self, x: float, y: float, z: float, input: PlumeInput
    ) -> float:
        """Calculate concentration at a single point [mg/m³].

        Args:
            x: Downwind distance [m].
            y: Cross-wind distance [m].
            z: Height above ground [m].
            input: PlumeInput parameters.

        Returns:
            Concentration [mg/m³].
        """
        return concentration_at_point(x, y, z, input) * 1e6

    def isopleth(
        self,
        input: PlumeInput,
        concentrations: List[float],
        z: float = 0.0,
    ) -> Dict[float, Dict[str, np.ndarray]]:
        """Compute isopleth contours.

        Args:
            input: PlumeInput parameters.
            concentrations: Target concentrations [mg/m³].
            z: Height for contour plane [m].

        Returns:
            Contour data dictionary.
        """
        return isopleth_data(input, concentrations, z)

    def plume_rise(
        self,
        Q_heat: float,
        wind_speed: float,
        stack_diameter: float,
        T_stack: float,
        T_ambient: float,
    ) -> float:
        """Calculate Briggs plume rise.

        Args:
            Q_heat: Heat emission rate [W].
            wind_speed: Wind speed [m/s].
            stack_diameter: Stack diameter [m].
            T_stack: Stack gas temperature [K].
            T_ambient: Ambient temperature [K].

        Returns:
            Plume rise ΔH [m].
        """
        return plume_rise_briggs(
            Q_heat, wind_speed, stack_diameter,
            T_stack, T_ambient,
        )
