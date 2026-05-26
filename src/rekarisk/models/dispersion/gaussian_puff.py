"""
Rekarisk Dispersion — Instantaneous Gaussian Puff Model.

Models the dispersion of an instantaneous (or finite-duration) release
as a travelling Gaussian puff, advected by wind and growing in all
three dimensions via Pasquill-Gifford dispersion coefficients.

Core Equation (instantaneous release):
    C(x,y,z,t) = Q / [(2π)^(3/2)·σx·σy·σz]
                 · exp(-(x-xc)²/2σx²) · exp(-(y)²/2σy²)
                 · [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)]

    where xc = x_source + u·(t - t_release) is the puff center.

For finite-duration releases: superposition of multiple puffs.

References:
    - Turner, D.B. (1994). Workbook of Atmospheric Dispersion Estimates.
    - CCPS Guidelines for Consequence Analysis of Chemical Releases.
    - Hanna, S.R. & Drivas, P.J. (1987). Guidelines for Use of Vapor
      Cloud Dispersion Models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from rekarisk.core.constants import G, P_ATM
from rekarisk.meteorology.stability import (
    StabilityClass,
    TerrainType,
    sigma_y,
    sigma_z,
    sigma_y_corrected,
    sigma_z as sigma_z_calc,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PuffInput:
    """Input parameters for Gaussian puff dispersion calculation.

    Attributes:
        mass: Total mass released [kg].
        release_time: Time of release t0 [s] (typically 0).
        release_duration: Duration of release [s]. If > 0, modelled as
            a finite-duration release using multiple puffs.
        wind_speed: Mean wind speed [m/s].
        wind_direction: Wind direction from north [degrees].
        stability_class: Pasquill-Gifford stability class (A-F).
        release_height: Height of release [m].
        terrain_type: 'rural' or 'urban'.
        temperature: Ambient temperature [K].
        pressure: Ambient pressure [Pa].
        source_x: Source x-coordinate [m] (default 0).
        source_y: Source y-coordinate [m] (default 0).
        decay_rate: First-order chemical decay rate λ [1/s] (default 0).
        deposition_velocity: Dry deposition velocity [m/s] (default 0).
        sampling_time: Averaging time [s] for σ correction (default 600).
        reference_time: Reference time for σ correction [s] (default 600).
        time_start: Start time for calculation [s] (default 0).
        time_end: End time for calculation [s] (default 3600 = 1 hour).
        time_steps: Number of time steps (default 100).
        grid_x_range: (x_min, x_max, n_x) for evaluation grid [m].
        grid_y_range: (y_min, y_max, n_y) for evaluation grid [m].
        grid_z_range: (z_min, z_max, n_z) for evaluation grid [m].
        num_puffs: Number of puffs for finite-duration release (default 20).
        molecular_weight: MW [g/mol] for unit conversion context.
    """

    mass: float = 100.0  # total mass [kg]
    release_time: float = 0.0  # t0 [s]
    release_duration: float = 0.0  # [s], 0 = instantaneous
    wind_speed: float = 5.0  # u [m/s]
    wind_direction: float = 0.0  # degrees from north
    stability_class: StabilityClass = "D"
    release_height: float = 0.0  # H [m]
    terrain_type: TerrainType = "rural"
    temperature: float = 298.15  # [K]
    pressure: float = P_ATM  # [Pa]
    source_x: float = 0.0
    source_y: float = 0.0
    decay_rate: float = 0.0  # λ [1/s]
    deposition_velocity: float = 0.0  # v_d [m/s]
    sampling_time: float = 600.0
    reference_time: float = 600.0
    time_start: float = 0.0
    time_end: float = 3600.0
    time_steps: int = 100
    grid_x_range: Tuple[float, float, int] = field(
        default_factory=lambda: (0.0, 5000.0, 51)
    )
    grid_y_range: Tuple[float, float, int] = field(
        default_factory=lambda: (-500.0, 500.0, 51)
    )
    grid_z_range: Tuple[float, float, int] = field(
        default_factory=lambda: (0.0, 200.0, 21)
    )
    num_puffs: int = 20
    molecular_weight: float = 29.0

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError(f"Mass must be non-negative, got {self.mass}")
        if self.wind_speed < 0.1:
            raise ValueError(f"Wind speed must be ≥ 0.1 m/s")
        if self.time_steps < 2:
            raise ValueError("At least 2 time steps required")
        if self.time_end <= self.time_start:
            raise ValueError("time_end must be > time_start")


@dataclass
class PuffSnapshot:
    """Concentration snapshot at a single time step.

    Attributes:
        time: Time of snapshot [s].
        concentration_grid: 3D concentration [mg/m³] at this time.
        max_concentration: Peak concentration in grid [mg/m³].
        puff_center_x: x-position of puff center [m].
        puff_center_y: y-position of puff center [m].
        sigma_x, sigma_y, sigma_z: Dispersion coefficients at puff center [m].
    """

    time: float
    concentration_grid: np.ndarray
    max_concentration: float
    puff_center_x: float
    puff_center_y: float
    sigma_x: float
    sigma_y: float
    sigma_z: float


@dataclass
class PuffResult:
    """Results of Gaussian puff dispersion calculation.

    Attributes:
        time_series: List of PuffSnapshot at each time step.
        times: Array of evaluation times [s].
        max_concentration_over_time: Peak concentration at each time step [mg/m³].
        puff_center_positions: Array of (x, y) center positions over time.
        total_dose: Time-integrated concentration ∫C dt [mg·s/m³].
        ground_dose: Time-integrated ground-level concentration [mg·s/m³].
        input: The PuffInput used.
    """

    time_series: List[PuffSnapshot] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_concentration_over_time: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    puff_center_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    total_dose: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1, 1))
    )
    ground_dose: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1))
    )
    input: Optional[PuffInput] = None

    @property
    def concentration_units(self) -> str:
        return "mg/m³"

    @property
    def dose_units(self) -> str:
        return "mg·s/m³"

    def get_snapshot_at(self, time: float) -> Optional[PuffSnapshot]:
        """Get the snapshot closest to a given time."""
        idx = np.argmin(np.abs(self.times - time))
        if idx < len(self.time_series):
            return self.time_series[idx]
        return None

    def to_dict(self) -> dict:
        """Serialize key results."""
        return {
            "times": self.times.tolist(),
            "max_concentration_over_time": self.max_concentration_over_time.tolist(),
            "puff_center_positions": self.puff_center_positions.tolist(),
        }


# ---------------------------------------------------------------------------
# Puff Concentration — Single Point
# ---------------------------------------------------------------------------


def _sigma_x_puff(
    downwind_distance_m: float,
    stability: StabilityClass,
    terrain: TerrainType = "rural",
) -> float:
    """Estimate along-wind dispersion coefficient σx for puffs.

    In standard puff models, σx ≈ σy for isotropic horizontal turbulence.
    This uses the same σy function for along-wind dispersion.

    Args:
        downwind_distance_m: Distance from source [m].
        stability: PG stability class.
        terrain: 'rural' or 'urban'.

    Returns:
        σx [m].
    """
    return sigma_y(downwind_distance_m, stability, terrain)


def concentration_puff(
    x: float,
    y: float,
    z: float,
    t: float,
    input: PuffInput,
    mass_per_puff: Optional[float] = None,
) -> float:
    """Calculate puff concentration at a single (x, y, z, t) point.

    Args:
        x, y, z: Position coordinates [m].
        t: Time [s].
        input: PuffInput parameters.
        mass_per_puff: Mass for this individual puff [kg].
            If None, uses input.mass (instantaneous release).

    Returns:
        Concentration [kg/m³].
    """
    if t < input.release_time:
        return 0.0

    Q_eff = mass_per_puff if mass_per_puff is not None else input.mass
    if Q_eff <= 0:
        return 0.0

    u = max(input.wind_speed, 0.1)
    H = input.release_height

    # Puff center position (advected by wind)
    travel_time = t - input.release_time
    # Convert wind direction to velocity components
    theta = math.radians(input.wind_direction)
    u_x = -u * math.sin(theta)  # east-west
    u_y = -u * math.cos(theta)  # north-south

    xc = input.source_x + u_x * travel_time
    yc = input.source_y + u_y * travel_time

    # Travel distance ≈ wind speed * travel time (magnitude)
    travel_dist = u * travel_time

    # Dispersion coefficients at travel distance
    sx = _sigma_x_puff(travel_dist, input.stability_class, input.terrain_type)
    sy = sigma_y(travel_dist, input.stability_class, input.terrain_type)
    sz = sigma_z(travel_dist, input.stability_class, input.terrain_type)

    # Apply time correction
    if input.sampling_time != input.reference_time:
        sy = sigma_y_corrected(
            travel_dist, input.stability_class, input.terrain_type,
            input.sampling_time, input.reference_time,
        )

    # Initial cloud dispersion: estimate initial sigma from cloud volume
    if Q_eff > 0:
        default_density = 2.0
        cloud_volume = Q_eff / default_density
        r0_est = max(0.1, (cloud_volume * 3.0 / (4.0 * math.pi)) ** (1.0/3.0))
        sigma_0 = r0_est / math.sqrt(2.0)
    else:
        sigma_0 = 0.1

    sx = math.sqrt(sigma_0 ** 2 + sx ** 2)
    sy = math.sqrt(sigma_0 ** 2 + sy ** 2)
    sz = math.sqrt(sigma_0 ** 2 + sz ** 2)

    sx = max(sx, 0.5)
    sy = max(sy, 0.5)
    sz = max(sz, 0.1)

    # Puff normalization factor
    # C = Q / [(2π)^(3/2) · σx · σy · σz]
    prefix = Q_eff / (math.pow(2.0 * math.pi, 1.5) * sx * sy * sz)

    # Along-wind dispersion
    dx = x - xc
    prefix *= math.exp(-0.5 * (dx / sx) ** 2)

    # Cross-wind dispersion (using yc for center offset)
    dy = y - yc
    prefix *= math.exp(-0.5 * (dy / sy) ** 2)

    # Vertical dispersion with ground reflection
    if sz > 0:
        term1 = math.exp(-0.5 * ((z - H) / sz) ** 2)
        term2 = math.exp(-0.5 * ((z + H) / sz) ** 2)
        prefix *= (term1 + term2)

    # Chemical decay
    if input.decay_rate > 0:
        prefix *= math.exp(-input.decay_rate * travel_time)

    return max(0.0, prefix)


# ---------------------------------------------------------------------------
# Grid Calculation — Single Puff at One Time
# ---------------------------------------------------------------------------


def _puff_snapshot(
    input: PuffInput,
    t: float,
    mass_per_puff: Optional[float] = None,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    z_coords: Optional[np.ndarray] = None,
) -> PuffSnapshot:
    """Compute concentration grid for a single puff at time t.

    Args:
        input: PuffInput.
        t: Evaluation time [s].
        mass_per_puff: Mass for this puff [kg].
        x_coords, y_coords, z_coords: Optional pre-created coordinate arrays.

    Returns:
        PuffSnapshot.
    """
    if x_coords is None:
        x_min, x_max, n_x = input.grid_x_range
        x_coords = np.linspace(x_min, x_max, n_x)

    if y_coords is None:
        y_min, y_max, n_y = input.grid_y_range
        y_coords = np.linspace(y_min, y_max, n_y)

    if z_coords is None:
        z_min, z_max, n_z = input.grid_z_range
        z_coords = np.linspace(z_min, z_max, n_z)

    if t < input.release_time:
        C_grid = np.zeros((len(x_coords), len(y_coords), len(z_coords)))
        return PuffSnapshot(
            time=t,
            concentration_grid=C_grid * 1e6,
            max_concentration=0.0,
            puff_center_x=input.source_x,
            puff_center_y=input.source_y,
            sigma_x=0.0,
            sigma_y=0.0,
            sigma_z=0.0,
        )

    Q_eff = mass_per_puff if mass_per_puff is not None else input.mass
    u = max(input.wind_speed, 0.1)
    H = input.release_height

    travel_time = t - input.release_time
    theta = math.radians(input.wind_direction)
    u_x = -u * math.sin(theta)
    u_y = -u * math.cos(theta)
    xc = input.source_x + u_x * travel_time
    yc = input.source_y + u_y * travel_time
    travel_dist = u * travel_time

    sx = _sigma_x_puff(travel_dist, input.stability_class, input.terrain_type)
    sy = sigma_y(travel_dist, input.stability_class, input.terrain_type)
    sz = sigma_z(travel_dist, input.stability_class, input.terrain_type)

    if input.sampling_time != input.reference_time:
        sy = sigma_y_corrected(
            travel_dist, input.stability_class, input.terrain_type,
            input.sampling_time, input.reference_time,
        )

    # Initial cloud dispersion: estimate initial sigma from cloud volume
    # For puff: sigma_0 ≈ R_cloud / sqrt(2), where R_cloud is initial radius
    # Estimate cloud radius from mass and a default density if not specified
    if Q_eff > 0:
        default_density = 2.0  # [kg/m³] default for gas releases
        cloud_volume = Q_eff / default_density
        r0_est = max(0.1, (cloud_volume * 3.0 / (4.0 * math.pi)) ** (1.0/3.0))
        sigma_0 = r0_est / math.sqrt(2.0)
    else:
        sigma_0 = 0.1

    # Effective sigma = sqrt(sigma_initial² + sigma_atmospheric²)
    sx = math.sqrt(sigma_0 ** 2 + sx ** 2)
    sy = math.sqrt(sigma_0 ** 2 + sy ** 2)
    sz = math.sqrt(sigma_0 ** 2 + sz ** 2)

    sx = max(sx, 0.5)
    sy = max(sy, 0.5)
    sz = max(sz, 0.1)

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    C_grid = np.zeros((nx, ny, nz), dtype=np.float64)

    prefix = Q_eff / (math.pow(2.0 * math.pi, 1.5) * sx * sy * sz)

    # Chemical decay factor
    decay_factor = 1.0
    if input.decay_rate > 0:
        decay_factor = math.exp(-input.decay_rate * travel_time)

    for i, x in enumerate(x_coords):
        dx = x - xc
        x_term = math.exp(-0.5 * (dx / sx) ** 2)
        if x_term < 1e-15:
            continue

        for j, y_val in enumerate(y_coords):
            dy = y_val - yc
            y_term = math.exp(-0.5 * (dy / sy) ** 2)
            if y_term < 1e-15:
                continue

            for k, z_val in enumerate(z_coords):
                term1 = math.exp(-0.5 * ((z_val - H) / sz) ** 2)
                term2 = math.exp(-0.5 * ((z_val + H) / sz) ** 2)
                vert_term = term1 + term2

                C_grid[i, j, k] = prefix * x_term * y_term * vert_term * decay_factor

    C_grid_mgm3 = C_grid * 1e6

    return PuffSnapshot(
        time=t,
        concentration_grid=C_grid_mgm3,
        max_concentration=float(np.max(C_grid_mgm3)),
        puff_center_x=float(xc),
        puff_center_y=float(yc),
        sigma_x=float(sx),
        sigma_y=float(sy),
        sigma_z=float(sz),
    )


# ---------------------------------------------------------------------------
# Main Calculation
# ---------------------------------------------------------------------------


def calculate_puff(input: PuffInput) -> PuffResult:
    """Calculate Gaussian puff dispersion over time.

    For instantaneous releases: single puff advected by wind.
    For finite-duration releases: multiple puffs released over duration,
    with concentration being the superposition of all puffs.

    Args:
        input: Complete PuffInput.

    Returns:
        PuffResult with time series, dose, and summary stats.

    Examples:
        >>> inp = PuffInput(mass=100, wind_speed=5, stability_class='D',
        ...                 time_start=0, time_end=3600, time_steps=20)
        >>> result = calculate_puff(inp)
        >>> len(result.time_series) == 20
        True
        >>> result.max_concentration_over_time[0] >= 0
        True
    """
    times = np.linspace(input.time_start, input.time_end, input.time_steps)

    # Create coordinate arrays
    x_min, x_max, n_x = input.grid_x_range
    y_min, y_max, n_y = input.grid_y_range
    z_min, z_max, n_z = input.grid_z_range

    x_coords = np.linspace(x_min, x_max, n_x)
    y_coords = np.linspace(y_min, y_max, n_y)
    z_coords = np.linspace(z_min, z_max, n_z)

    # Total dose accumulator
    total_dose = np.zeros((n_x, n_y, n_z), dtype=np.float64)
    ground_dose = np.zeros((n_x, n_y), dtype=np.float64)

    time_series: List[PuffSnapshot] = []
    max_conc_over_time = np.zeros(input.time_steps, dtype=np.float64)
    center_positions = np.zeros((input.time_steps, 2), dtype=np.float64)

    if input.release_duration <= 0:
        # Instantaneous release — single puff
        for i, t in enumerate(times):
            snap = _puff_snapshot(input, t, x_coords=x_coords,
                                   y_coords=y_coords, z_coords=z_coords)
            time_series.append(snap)
            max_conc_over_time[i] = snap.max_concentration
            center_positions[i] = [snap.puff_center_x, snap.puff_center_y]

            # Integrate dose: trapz rule
            if i > 0:
                dt = times[i] - times[i - 1]
                avg_conc = (time_series[i - 1].concentration_grid +
                            snap.concentration_grid) / 2.0
                total_dose += avg_conc * dt
                ground_dose += (time_series[i - 1].concentration_grid[:, :, 0] +
                                snap.concentration_grid[:, :, 0]) / 2.0 * dt
    else:
        # Finite-duration release — multiple puffs
        n_puffs = input.num_puffs
        mass_per_puff = input.mass / n_puffs
        release_interval = input.release_duration / n_puffs

        # Generate puff release times
        puff_times = np.linspace(
            input.release_time,
            input.release_time + input.release_duration,
            n_puffs,
        )

        for i, t in enumerate(times):
            # Superposition: sum contributions from all released puffs
            C_combined = np.zeros((n_x, n_y, n_z), dtype=np.float64)

            for puff_idx, puff_t in enumerate(puff_times):
                if t < puff_t:
                    continue  # puff not yet released

                # Create a modified input for this puff
                puff_input = PuffInput(
                    mass=mass_per_puff,
                    release_time=puff_t,
                    release_duration=0.0,  # each puff is instantaneous
                    wind_speed=input.wind_speed,
                    wind_direction=input.wind_direction,
                    stability_class=input.stability_class,
                    release_height=input.release_height,
                    terrain_type=input.terrain_type,
                    temperature=input.temperature,
                    pressure=input.pressure,
                    source_x=input.source_x,
                    source_y=input.source_y,
                    decay_rate=input.decay_rate,
                    deposition_velocity=input.deposition_velocity,
                    sampling_time=input.sampling_time,
                    reference_time=input.reference_time,
                    grid_x_range=input.grid_x_range,
                    grid_y_range=input.grid_y_range,
                    grid_z_range=input.grid_z_range,
                )

                snap = _puff_snapshot(
                    puff_input, t,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    z_coords=z_coords,
                )
                C_combined += snap.concentration_grid

            # Build aggregate snapshot
            u = max(input.wind_speed, 0.1)
            mid_puff_t = input.release_time + input.release_duration / 2.0
            mid_travel = u * max(t - mid_puff_t, 0)
            theta = math.radians(input.wind_direction)
            xc = input.source_x - u * math.sin(theta) * max(t - mid_puff_t, 0)
            yc = input.source_y - u * math.cos(theta) * max(t - mid_puff_t, 0)

            sx_final = _sigma_x_puff(mid_travel, input.stability_class,
                                      input.terrain_type)

            snap_agg = PuffSnapshot(
                time=t,
                concentration_grid=C_combined,
                max_concentration=float(np.max(C_combined)),
                puff_center_x=float(xc),
                puff_center_y=float(yc),
                sigma_x=float(sx_final),
                sigma_y=float(sx_final),  # approximate
                sigma_z=float(sigma_z(mid_travel, input.stability_class,
                                       input.terrain_type)),
            )

            time_series.append(snap_agg)
            max_conc_over_time[i] = snap_agg.max_concentration
            center_positions[i] = [snap_agg.puff_center_x, snap_agg.puff_center_y]

        # Compute dose for finite-duration release
        for i in range(1, len(time_series)):
            dt = times[i] - times[i - 1]
            avg_conc = (time_series[i - 1].concentration_grid +
                        time_series[i].concentration_grid) / 2.0
            total_dose += avg_conc * dt
            ground_dose += (time_series[i - 1].concentration_grid[:, :, 0] +
                            time_series[i].concentration_grid[:, :, 0]) / 2.0 * dt

    return PuffResult(
        time_series=time_series,
        times=times,
        max_concentration_over_time=max_conc_over_time,
        puff_center_positions=center_positions,
        total_dose=total_dose,
        ground_dose=ground_dose,
        input=input,
    )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def concentration_at(
    x: float, y: float, z: float, t: float, input: PuffInput,
) -> float:
    """Puff concentration at a single (x, y, z, t) point [mg/m³].

    Args:
        x, y, z: Position [m].
        t: Time [s].
        input: PuffInput.

    Returns:
        Concentration [mg/m³].
    """
    return concentration_puff(x, y, z, t, input) * 1e6


def finite_duration_release(
    mass_rate: float,
    duration: float,
    input: PuffInput,
) -> PuffResult:
    """Calculate puff dispersion for a finite-duration continuous release.

    Uses superposition of multiple puffs released at intervals.

    Args:
        mass_rate: Mass release rate Q [kg/s].
        duration: Release duration [s].
        input: Base PuffInput (mass, release_duration will be overridden).

    Returns:
        PuffResult.

    Examples:
        >>> base = PuffInput(mass=100, wind_speed=5, stability_class='D')
        >>> result = finite_duration_release(10.0, 10.0, base)
        >>> result.total_dose.size > 0
        True
    """
    total_mass = mass_rate * duration
    modified = PuffInput(
        mass=total_mass,
        release_time=input.release_time,
        release_duration=duration,
        wind_speed=input.wind_speed,
        wind_direction=input.wind_direction,
        stability_class=input.stability_class,
        release_height=input.release_height,
        terrain_type=input.terrain_type,
        temperature=input.temperature,
        pressure=input.pressure,
        source_x=input.source_x,
        source_y=input.source_y,
        decay_rate=input.decay_rate,
        deposition_velocity=input.deposition_velocity,
        sampling_time=input.sampling_time,
        reference_time=input.reference_time,
        time_start=input.time_start,
        time_end=input.time_end,
        time_steps=input.time_steps,
        grid_x_range=input.grid_x_range,
        grid_y_range=input.grid_y_range,
        grid_z_range=input.grid_z_range,
        num_puffs=input.num_puffs,
    )
    return calculate_puff(modified)


def concentration_puff_time_series(
    x: float, y: float, z: float,
    times: np.ndarray,
    input: PuffInput,
) -> np.ndarray:
    """Compute concentration at a fixed (x, y, z) over time.

    Args:
        x, y, z: Fixed observation point [m].
        times: Time steps [s].
        input: PuffInput.

    Returns:
        Concentration [mg/m³] at each time.
    """
    result = np.zeros(len(times), dtype=np.float64)
    for i, t in enumerate(times):
        result[i] = concentration_puff(x, y, z, t, input) * 1e6
    return result


# ---------------------------------------------------------------------------
# Puff Calculator Class
# ---------------------------------------------------------------------------


class GaussianPuffCalculator:
    """Main calculator for Gaussian puff dispersion.

    Usage:
        calc = GaussianPuffCalculator()
        result = calc.calculate(puff_input)
        C = calc.concentration_at(500, 10, 0, 120, puff_input)
    """

    def calculate(self, input: PuffInput) -> PuffResult:
        """Run full puff dispersion calculation.

        Args:
            input: Complete PuffInput.

        Returns:
            PuffResult with time series and dose.
        """
        return calculate_puff(input)

    def concentration_at(
        self, x: float, y: float, z: float, t: float, input: PuffInput,
    ) -> float:
        """Calculate concentration at a single point [mg/m³]."""
        return concentration_puff(x, y, z, t, input) * 1e6

    def finite_duration_release(
        self, mass_rate: float, duration: float, input: PuffInput,
    ) -> PuffResult:
        """Calculate for a finite-duration continuous release."""
        return finite_duration_release(mass_rate, duration, input)
