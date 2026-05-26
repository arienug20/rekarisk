"""
Rekarisk Dispersion Models — Atmospheric Dispersion.

Provides three dispersion model types and an auto-dispatcher:

    gaussian_plume  — Continuous steady-state releases (Gaussian plume)
    gaussian_puff   — Instantaneous/time-varying releases (Gaussian puff)
    dense_gas       — Dense gas dispersion with gravity spreading
    dispersion_dispatcher — Auto-select the appropriate model
"""

from .gaussian_plume import (
    GaussianPlumeCalculator,
    PlumeInput,
    PlumeResult,
    concentration_at as plume_concentration_at,
    calculate_plume,
    centerline_profile,
    crosswind_profile,
    isopleth_data,
    max_ground_concentration,
    plume_rise,
    plume_rise_briggs,
    vertical_profile,
)
from .gaussian_puff import (
    GaussianPuffCalculator,
    PuffInput,
    PuffResult,
    PuffSnapshot,
    calculate_puff,
    concentration_at as puff_concentration_at,
    finite_duration_release,
)
from .dense_gas import (
    DenseGasCalculator,
    DenseGasInput,
    DenseGasResult,
    DenseGasDensePhaseRecord,
    calculate_dense_gas,
    check_if_dense,
    transition_criteria,
)
from .dispersion_dispatcher import (
    DispersionDispatcher,
    DispatchResult,
    ReleaseInfo,
    WeatherInfo,
    quick_dispersion,
)

__all__ = [
    # Plume
    "GaussianPlumeCalculator",
    "PlumeInput",
    "PlumeResult",
    "plume_concentration_at",
    "calculate_plume",
    "centerline_profile",
    "crosswind_profile",
    "vertical_profile",
    "isopleth_data",
    "max_ground_concentration",
    "plume_rise",
    "plume_rise_briggs",
    # Puff
    "GaussianPuffCalculator",
    "PuffInput",
    "PuffResult",
    "PuffSnapshot",
    "calculate_puff",
    "puff_concentration_at",
    "finite_duration_release",
    # Dense Gas
    "DenseGasCalculator",
    "DenseGasInput",
    "DenseGasResult",
    "DenseGasDensePhaseRecord",
    "calculate_dense_gas",
    "check_if_dense",
    "transition_criteria",
    # Dispatcher
    "DispersionDispatcher",
    "DispatchResult",
    "ReleaseInfo",
    "WeatherInfo",
    "quick_dispersion",
]
