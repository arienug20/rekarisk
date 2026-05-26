"""
Rekarisk Fire Models — Pool Fire, Jet Fire, BLEVE, Flash Fire.

Implements thermal radiation consequence models following CCPS, TNO Yellow Book,
HSE, API RP 521, and related industry guidance.

Modules:
  - pool_fire: Circular pool fire models (Mudan/Thomas, Shokri-Beyler)
  - jet_fire: Jet fire models (API RP 521, Kalghatgi, Considine & Grint)
  - bleve: BLEVE & fireball models (Roberts, CCPS, HSE)
  - flash_fire: Flash fire envelope from dispersion results
"""

from .pool_fire import (
    PoolFireInput,
    PoolFireResult,
    calculate_pool_fire,
    burning_rate_default,
    flame_length_thomas,
    flame_tilt_aga,
    surface_emissive_power,
    atmospheric_transmissivity,
    view_factor_cylinder_vertical,
    view_factor_cylinder_tilted,
    thermal_radiation_point_source,
    thermal_radiation_solid_flame,
    thermal_radiation_vs_distance,
    distance_to_thresholds,
)

from .jet_fire import (
    JetFireInput,
    JetFireResult,
    calculate_jet_fire,
    flame_length_kalghatgi,
    flame_length_vertical_jet,
    thermal_radiation_point_source as jet_point_source,
    thermal_radiation_solid_flame as jet_solid_flame,
)

from .bleve import (
    BLEVEInput,
    BLEVEResult,
    calculate_bleve,
    fireball_diameter_roberts,
    fireball_duration_roberts,
    fireball_sep,
    view_factor_sphere,
    fragment_throw_distance,
)

from .flash_fire import (
    FlashFireInput,
    FlashFireResult,
    calculate_flash_fire,
    find_lfl_contour,
    find_ufl_contour,
    lfl_area,
    max_lfl_distance,
    flash_fire_thermal_radiation,
)

__all__ = [
    # Pool Fire
    "PoolFireInput", "PoolFireResult", "calculate_pool_fire",
    "burning_rate_default", "flame_length_thomas", "flame_tilt_aga",
    "surface_emissive_power", "atmospheric_transmissivity",
    "view_factor_cylinder_vertical", "view_factor_cylinder_tilted",
    "thermal_radiation_point_source", "thermal_radiation_solid_flame",
    "thermal_radiation_vs_distance", "distance_to_thresholds",
    # Jet Fire
    "JetFireInput", "JetFireResult", "calculate_jet_fire",
    "flame_length_kalghatgi", "flame_length_vertical_jet",
    "jet_point_source", "jet_solid_flame",
    # BLEVE
    "BLEVEInput", "BLEVEResult", "calculate_bleve",
    "fireball_diameter_roberts", "fireball_duration_roberts",
    "fireball_sep", "view_factor_sphere", "fragment_throw_distance",
    # Flash Fire
    "FlashFireInput", "FlashFireResult", "calculate_flash_fire",
    "find_lfl_contour", "find_ufl_contour", "lfl_area",
    "max_lfl_distance", "flash_fire_thermal_radiation",
]
