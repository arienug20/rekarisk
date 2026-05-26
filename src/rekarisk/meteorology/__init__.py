"""
Rekarisk Meteorology Module.

Provides Pasquill-Gifford stability classification, dispersion coefficients
(sigma-y, sigma-z), wind profile models, weather observation data handling,
and wind rose analysis for atmospheric dispersion modeling.

Modules:
    stability      — PG stability classification & sigma coefficients
    meteorology    — Wind profile, density, lapse rate, integrated state
    wind_rose      — Wind rose (binned wind speed × direction data)
    weather_data   — Weather observation time series & statistics
"""

from .stability import (
    StabilityClass,
    TerrainType,
    classify_stability,
    classify_stability_from_cloud,
    classify_stability_from_radiation,
    get_stability_description,
    list_terrain_types,
    mixing_height,
    power_law_exponent,
    sigma_y,
    sigma_y_corrected,
    sigma_z,
    surface_roughness,
)
from .meteorology import (
    MeteorologicalState,
    atmospheric_density,
    components_to_wind_direction,
    friction_velocity,
    get_lapse_rate,
    pressure_at_height,
    saturation_vapor_pressure,
    temperature_at_height,
    wind_direction_to_components,
    wind_log_law,
    wind_power_law,
)
from .wind_rose import (
    DIRECTION_NAMES,
    DEFAULT_SPEED_BINS,
    DEFAULT_SPEED_LABELS,
    N_DIRECTIONS,
    SECTOR_WIDTH_DEG,
    WindRoseData,
    direction_angle_from_index,
    direction_index,
    direction_name_from_angle,
    speed_class_index,
)
from .weather_data import (
    WeatherDataset,
    WeatherObservation,
)

__all__ = [
    # Stability
    "StabilityClass",
    "TerrainType",
    "classify_stability",
    "classify_stability_from_cloud",
    "classify_stability_from_radiation",
    "get_stability_description",
    "list_terrain_types",
    "mixing_height",
    "power_law_exponent",
    "sigma_y",
    "sigma_y_corrected",
    "sigma_z",
    "surface_roughness",
    # Meteorology
    "MeteorologicalState",
    "atmospheric_density",
    "components_to_wind_direction",
    "friction_velocity",
    "get_lapse_rate",
    "pressure_at_height",
    "saturation_vapor_pressure",
    "temperature_at_height",
    "wind_direction_to_components",
    "wind_log_law",
    "wind_power_law",
    # Wind Rose
    "DIRECTION_NAMES",
    "DEFAULT_SPEED_BINS",
    "DEFAULT_SPEED_LABELS",
    "N_DIRECTIONS",
    "SECTOR_WIDTH_DEG",
    "WindRoseData",
    "direction_angle_from_index",
    "direction_index",
    "direction_name_from_angle",
    "speed_class_index",
    # Weather Data
    "WeatherDataset",
    "WeatherObservation",
]
