"""
Rekarisk Dispersion — Auto-Dispatcher.

Automatically selects the appropriate dispersion model based on release
characteristics:

    1. Check if substance is dense gas (ρ/ρ_air > 1.1) → dense_gas
    2. Check if release is instantaneous → gaussian_puff
    3. Check if release is continuous → gaussian_plume
    4. Combine transitions for mixed scenarios (dense → passive)

Usage:
    dispatcher = DispersionDispatcher()
    result = dispatcher.dispatch(release_info, weather_info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from rekarisk.core.constants import G, P_ATM, T_0C

from .gaussian_plume import (
    GaussianPlumeCalculator,
    PlumeInput,
    PlumeResult,
    plume_rise_briggs,
)
from .gaussian_puff import (
    GaussianPuffCalculator,
    PuffInput,
    PuffResult,
)
from .dense_gas import (
    DenseGasCalculator,
    DenseGasInput,
    DenseGasResult,
    check_if_dense,
)

# Imperial to metric conversion helpers
try:
    from rekarisk.core.units import UnitConverter
    _HAS_UNITS = True
except ImportError:
    _HAS_UNITS = False


# ---------------------------------------------------------------------------
# Release Info Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReleaseInfo:
    """Unified release information for auto-dispatching.

    Provides all source term information needed by any dispersion model.

    Attributes:
        mass: Total mass released [kg] (for instantaneous releases).
        mass_rate: Mass release rate [kg/s] (for continuous releases).
        duration: Release duration [s]. 0 = instantaneous.
        substance_density: Density of released substance [kg/m³].
        molecular_weight: MW [g/mol].
        temperature: Release temperature [K].
        phase: 'gas', 'liquid', 'two_phase'.
        release_height: Height of release [m].
        release_velocity: Exit velocity [m/s] (for momentum rise).
        release_diameter: Orifice/stack diameter [m].
        heat_release_rate: Heat release rate [W] (for plume rise).
    """

    mass: float = 0.0
    mass_rate: float = 0.0
    duration: float = 0.0
    substance_density: float = 1.2
    molecular_weight: float = 29.0
    temperature: float = 298.15
    phase: str = "gas"
    release_height: float = 0.0
    release_velocity: float = 0.0
    release_diameter: float = 0.0
    heat_release_rate: float = 0.0

    @property
    def is_instantaneous(self) -> bool:
        """True for instantaneous releases (duration ≈ 0)."""
        return self.duration <= 0

    @property
    def is_continuous(self) -> bool:
        """True for continuous releases (duration > 0)."""
        return self.duration > 0


@dataclass
class WeatherInfo:
    """Unified weather/atmospheric information for auto-dispatching.

    Attributes:
        wind_speed: Wind speed at reference height [m/s].
        wind_direction: Wind direction from north [degrees].
        stability_class: PG stability class.
        terrain_type: 'rural' or 'urban'.
        temperature: Ambient temperature [K].
        pressure: Ambient pressure [Pa].
        relative_humidity: Relative humidity [%].
        surface_roughness: Surface roughness z0 [m].
        reference_height: Reference height for wind [m].
    """

    wind_speed: float = 5.0
    wind_direction: float = 0.0
    stability_class: str = "D"
    terrain_type: str = "rural"
    temperature: float = 298.15
    pressure: float = P_ATM
    relative_humidity: float = 50.0
    surface_roughness: float = 0.1
    reference_height: float = 10.0

    @property
    def air_density(self) -> float:
        """Calculate ambient air density [kg/m³]."""
        from rekarisk.meteorology.meteorology import atmospheric_density
        return atmospheric_density(
            temperature_k=self.temperature,
            pressure_pa=self.pressure,
            relative_humidity_pct=self.relative_humidity,
        )


# ---------------------------------------------------------------------------
# Dispatch Result
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Result of auto-dispatching.

    Attributes:
        model_used: Name of the model(s) applied.
        is_dense_gas: Whether dense gas model was used.
        dense_result: Result from dense gas phase (if applicable).
        plume_result: Result from Gaussian plume phase (if applicable).
        puff_result: Result from Gaussian puff phase (if applicable).
        transition_distance: Distance where dense→passive transition occurred [m].
        max_concentration: Overall peak concentration [mg/m³].
        max_distance: Distance to peak concentration [m].
        message: Diagnostic message about model selection.
    """

    model_used: str = "none"
    is_dense_gas: bool = False
    dense_result: Optional[DenseGasResult] = None
    plume_result: Optional[PlumeResult] = None
    puff_result: Optional[PuffResult] = None
    transition_distance: float = 0.0
    max_concentration: float = 0.0
    max_distance: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "model_used": self.model_used,
            "is_dense_gas": self.is_dense_gas,
            "transition_distance_m": self.transition_distance,
            "max_concentration_mgm3": self.max_concentration,
            "max_distance_m": self.max_distance,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Dispersion Dispatcher
# ---------------------------------------------------------------------------


class DispersionDispatcher:
    """Auto-dispatch to the appropriate dispersion model.

    Selection logic:
        1. If dense gas (ρ_c/ρ_a > 1.1) → dense_gas + optional Gaussian transition
        2. If instantaneous release → gaussian_puff
        3. If continuous release → gaussian_plume

    Usage:
        dispatcher = DispersionDispatcher()
        result = dispatcher.dispatch(release, weather)
    """

    def __init__(self):
        self._plume_calc = GaussianPlumeCalculator()
        self._puff_calc = GaussianPuffCalculator()
        self._dense_calc = DenseGasCalculator()
        self._dense_threshold = 1.1

    def dispatch(
        self,
        release: ReleaseInfo,
        weather: WeatherInfo,
        grid_x_range: Optional[tuple] = None,
        grid_y_range: Optional[tuple] = None,
        grid_z_range: Optional[tuple] = None,
        sampling_time: float = 600.0,
        decay_rate: float = 0.0,
        deposition_velocity: float = 0.0,
    ) -> DispatchResult:
        """Auto-select and run the appropriate dispersion model.

        Args:
            release: ReleaseInfo with source term data.
            weather: WeatherInfo with atmospheric data.
            grid_x_range: Optional (x_min, x_max, n_x) for grid.
            grid_y_range: Optional (y_min, y_max, n_y) for grid.
            grid_z_range: Optional (z_min, z_max, n_z) for grid.
            sampling_time: Averaging time [s].
            decay_rate: Chemical decay rate [1/s].
            deposition_velocity: Dry deposition velocity [m/s].

        Returns:
            DispatchResult with output from the selected model(s).
        """
        # Compute air density
        rho_air = weather.air_density

        # Determine grid ranges
        if grid_x_range is None:
            grid_x_range = (50.0, 5000.0, 51)
        if grid_y_range is None:
            grid_y_range = (-500.0, 500.0, 51)
        if grid_z_range is None:
            grid_z_range = (0.0, 200.0, 21)

        # --- Step 1: Check for dense gas ---
        is_dense = check_if_dense(
            release.substance_density, rho_air, self._dense_threshold
        )

        if is_dense:
            return self._dispatch_dense_gas(
                release, weather, grid_x_range, grid_y_range, grid_z_range,
                sampling_time, decay_rate, deposition_velocity,
            )

        # --- Step 2: Instantaneous vs Continuous ---
        if release.is_instantaneous:
            return self._dispatch_puff(
                release, weather, grid_x_range, grid_y_range, grid_z_range,
                sampling_time, decay_rate, deposition_velocity,
            )
        else:
            return self._dispatch_plume(
                release, weather, grid_x_range, grid_y_range, grid_z_range,
                sampling_time, decay_rate, deposition_velocity,
            )

    def _dispatch_plume(
        self,
        release: ReleaseInfo,
        weather: WeatherInfo,
        grid_x_range: tuple,
        grid_y_range: tuple,
        grid_z_range: tuple,
        sampling_time: float,
        decay_rate: float,
        deposition_velocity: float,
    ) -> DispatchResult:
        """Dispatch to Gaussian plume model."""
        # Calculate plume rise if applicable
        plume_dh = 0.0
        if release.heat_release_rate > 0 and release.release_diameter > 0:
            plume_dh = plume_rise_briggs(
                Q_heat=release.heat_release_rate,
                wind_speed=weather.wind_speed,
                stack_diameter=release.release_diameter,
                T_stack=release.temperature,
                T_ambient=weather.temperature,
            )
        elif release.release_velocity > 0 and release.release_diameter > 0:
            # Momentum rise
            plume_dh = 3.0 * release.release_diameter * release.release_velocity / max(weather.wind_speed, 0.1)

        effective_height = max(release.release_height + plume_dh, 0.0)

        plume_input = PlumeInput(
            source_rate=release.mass_rate,
            wind_speed=weather.wind_speed,
            stability_class=weather.stability_class,
            release_height=effective_height,
            terrain_type=weather.terrain_type,
            temperature=weather.temperature,
            pressure=weather.pressure,
            decay_rate=decay_rate,
            deposition_velocity=deposition_velocity,
            sampling_time=sampling_time,
            grid_x_range=grid_x_range,
            grid_y_range=grid_y_range,
            grid_z_range=grid_z_range,
            source_diameter=release.release_diameter,
            stack_exit_velocity=release.release_velocity,
            stack_temperature=release.temperature,
            molecular_weight=release.molecular_weight,
        )

        result = self._plume_calc.calculate(plume_input)
        result.plume_rise_delta = plume_dh

        return DispatchResult(
            model_used="gaussian_plume",
            is_dense_gas=False,
            plume_result=result,
            max_concentration=result.max_concentration,
            max_distance=result.max_distance,
            message=(
                f"Continuous release ({release.mass_rate:.3g} kg/s) "
                f"modeled with Gaussian plume. "
                f"Plume rise: {plume_dh:.1f} m. "
                f"Effective height: {effective_height:.1f} m."
                if plume_dh > 0 else
                f"Continuous release ({release.mass_rate:.3g} kg/s) "
                f"modeled with Gaussian plume. "
                f"Release height: {effective_height:.1f} m."
            ),
        )

    def _dispatch_puff(
        self,
        release: ReleaseInfo,
        weather: WeatherInfo,
        grid_x_range: tuple,
        grid_y_range: tuple,
        grid_z_range: tuple,
        sampling_time: float,
        decay_rate: float,
        deposition_velocity: float,
    ) -> DispatchResult:
        """Dispatch to Gaussian puff model."""
        puff_input = PuffInput(
            mass=release.mass,
            release_time=0.0,
            release_duration=release.duration,
            wind_speed=weather.wind_speed,
            wind_direction=weather.wind_direction,
            stability_class=weather.stability_class,
            release_height=release.release_height,
            terrain_type=weather.terrain_type,
            temperature=weather.temperature,
            pressure=weather.pressure,
            decay_rate=decay_rate,
            deposition_velocity=deposition_velocity,
            sampling_time=sampling_time,
            grid_x_range=grid_x_range,
            grid_y_range=grid_y_range,
            grid_z_range=grid_z_range,
            molecular_weight=release.molecular_weight,
        )

        result = self._puff_calc.calculate(puff_input)

        # Get max concentration across all time steps
        max_C = float(np.max(result.max_concentration_over_time))
        max_t_idx = int(np.argmax(result.max_concentration_over_time))
        if max_t_idx < len(result.times):
            max_t = float(result.times[max_t_idx])
        else:
            max_t = 0.0

        return DispatchResult(
            model_used="gaussian_puff",
            is_dense_gas=False,
            puff_result=result,
            max_concentration=max_C,
            max_distance=0.0,  # puffs move; distance not single-valued
            message=(
                f"Instantaneous release ({release.mass:.3g} kg) "
                f"modeled with Gaussian puff. "
                f"Peak at t ≈ {max_t:.0f} s. "
                f"Dose calculated over {len(result.time_series)} time steps."
            ),
        )

    def _dispatch_dense_gas(
        self,
        release: ReleaseInfo,
        weather: WeatherInfo,
        grid_x_range: tuple,
        grid_y_range: tuple,
        grid_z_range: tuple,
        sampling_time: float,
        decay_rate: float,
        deposition_velocity: float,
    ) -> DispatchResult:
        """Dispatch to dense gas model with possible transition."""
        rho_air = weather.air_density

        dense_input = DenseGasInput(
            source_mass=release.mass if release.is_instantaneous else 0.0,
            source_rate=release.mass_rate if not release.is_instantaneous else 0.0,
            release_type="instantaneous" if release.is_instantaneous else "continuous",
            release_duration=release.duration if not release.is_instantaneous else 0.0,
            cloud_density=release.substance_density,
            air_density=rho_air,
            wind_speed=weather.wind_speed,
            stability_class=weather.stability_class,
            release_height=release.release_height,
            terrain_type=weather.terrain_type,
            temperature_cloud=release.temperature,
            temperature_ambient=weather.temperature,
            enable_transition=True,
            molecular_weight=release.molecular_weight,
        )

        dense_result = self._dense_calc.calculate(dense_input)
        trans_dist = dense_result.transition_distance

        msg_parts = [
            f"Dense gas model applied "
            f"(ρ_c/ρ_a = {release.substance_density/rho_air:.2f}).",
        ]

        # After transition, optionally run Gaussian plume from transition point
        gaussian_result = None
        if trans_dist > 0 and dense_result.transition_time > 0:
            effective_rate = release.mass_rate if not release.is_instantaneous else 0.0
            if effective_rate <= 0 and release.mass > 0:
                # Approximate continuous rate from instantaneous
                effective_rate = release.mass / max(dense_result.transition_time, 1.0)

            if effective_rate > 0:
                try:
                    plume_input = PlumeInput(
                        source_rate=effective_rate,
                        wind_speed=weather.wind_speed,
                        stability_class=weather.stability_class,
                        release_height=max(release.release_height,
                                           dense_result.transition_height),
                        terrain_type=weather.terrain_type,
                        temperature=weather.temperature,
                        pressure=weather.pressure,
                        decay_rate=decay_rate,
                        deposition_velocity=deposition_velocity,
                        sampling_time=sampling_time,
                        grid_x_range=grid_x_range,
                        grid_y_range=grid_y_range,
                        grid_z_range=grid_z_range,
                        molecular_weight=release.molecular_weight,
                    )
                    gaussian_result = self._plume_calc.calculate(plume_input)
                    msg_parts.append(
                        f"Transitioned to Gaussian plume at "
                        f"{trans_dist:.0f} m ({dense_result.transition_time:.0f} s). "
                        f"Passive-phase max: {gaussian_result.max_concentration:.2f} "
                        f"mg/m³."
                    )
                except Exception:
                    msg_parts.append(
                        f"Transition at {trans_dist:.0f} m, "
                        f"but passive plume calculation failed."
                    )
            else:
                msg_parts.append(
                    f"Transitioned at {trans_dist:.0f} m "
                    f"({dense_result.transition_time:.0f} s) — "
                    f"cloud now passive."
                )

        max_C_dense = dense_result.max_concentration
        max_C_passive = (
            gaussian_result.max_concentration
            if gaussian_result is not None
            else 0.0
        )
        overall_max = max(max_C_dense, max_C_passive)

        return DispatchResult(
            model_used=(
                "dense_gas → gaussian_plume"
                if gaussian_result is not None
                else "dense_gas"
            ),
            is_dense_gas=True,
            dense_result=dense_result,
            plume_result=gaussian_result,
            transition_distance=trans_dist,
            max_concentration=overall_max,
            max_distance=(
                trans_dist
                if gaussian_result is None
                else (trans_dist + gaussian_result.max_distance)
            ),
            message=" ".join(msg_parts),
        )


# ---------------------------------------------------------------------------
# Quick Convenience Function
# ---------------------------------------------------------------------------


def quick_dispersion(
    mass_rate: float = 1.0,
    duration: float = 0.0,
    mass: float = 0.0,
    wind_speed: float = 5.0,
    stability: str = "D",
    release_height: float = 0.0,
    cloud_density: float = 1.2,
    temperature: float = 298.15,
) -> DispatchResult:
    """Quick one-shot dispersion calculation with minimal inputs.

    Args:
        mass_rate: Mass release rate [kg/s] (continuous).
        duration: Release duration [s] (0 = instantaneous).
        mass: Total mass [kg] (for instantaneous).
        wind_speed: Wind speed [m/s].
        stability: PG stability class.
        release_height: Release height [m].
        cloud_density: Cloud density [kg/m³].
        temperature: Ambient temperature [K].

    Returns:
        DispatchResult.

    Examples:
        >>> result = quick_dispersion(mass_rate=5.0, duration=60.0,
        ...                           wind_speed=5.0, stability='D')
        >>> result.model_used
        'gaussian_plume'
        >>> result.max_concentration > 0
        True
    """
    release = ReleaseInfo(
        mass=mass if mass > 0 else mass_rate * duration,
        mass_rate=mass_rate,
        duration=duration,
        substance_density=cloud_density,
        temperature=temperature,
        release_height=release_height,
    )
    weather = WeatherInfo(
        wind_speed=wind_speed,
        stability_class=stability,
        temperature=temperature,
    )
    dispatcher = DispersionDispatcher()
    return dispatcher.dispatch(release, weather)
