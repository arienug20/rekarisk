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


def calculate_plume(input: PlumeInput) -> PlumeResult:
    """Calculate full 3D concentration grid for a Gaussian plume.

    Computes concentration on a 3D grid specified by the input parameters.

    Args:
        input: Complete PlumeInput with grid specification.

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
