"""
Rekarisk Source Term / Discharge Engine — Phase 2.

Provides discharge rate, blowdown, two-phase flow, relief valve sizing,
pipe rupture, and pool evaporation models for consequence analysis.

Modules:
  - orifice: Liquid, gas, and two-phase orifice discharge
  - vessel_depressur: Vessel blowdown (gas, two-phase, API 521)
  - two_phase: HEM, slip models (Fauske, Moody), Omega method
  - relief_valve: PSV sizing per API 520 (gas, liquid, steam, two-phase)
  - pipe_flow: Pipe flow, full-bore rupture, hole-in-pipe, Lockhart-Martinelli
  - pool_evaporation: Pool spreading (gravity-inertia/viscous) and evaporation
"""

from .orifice import (
    ReleasePhase,
    FlowRegime as OrificeFlowRegime,
    OrificeInput,
    OrificeResult,
    calculate_choked_pressure,
    liquid_orifice_discharge,
    gas_orifice_discharge,
    calculate_flashing_fraction,
    estimate_omega_parameter,
    two_phase_orifice_discharge,
    calculate_orifice,
    quick_liquid_orifice,
    quick_gas_orifice,
)

from .vessel_depressur import (
    VesselInput,
    VesselResult,
    calculate_vessel_blowdown,
)

from .two_phase import (
    TwoPhaseModel,
    TwoPhaseRegime,
    TwoPhaseInput,
    TwoPhaseResult,
    calculate_flash_fraction as tp_calculate_flash_fraction,
    calculate_saturated_flash_fraction,
    calculate_omega_parameter as tp_calculate_omega_parameter,
    hem_mass_flux,
    fauske_slip_ratio,
    moody_slip_ratio,
    calculate_two_phase_flow,
)

from .relief_valve import (
    ReliefScenario,
    ValveType,
    API_ORIFICE_AREAS,
    API_ORIFICE_ORDER,
    select_orifice_designation,
    ReliefValveInput,
    ReliefValveResult,
    size_gas_vapor_relief,
    size_liquid_relief,
    size_steam_relief,
    size_two_phase_relief,
    calculate_relief_valve,
)

from .pipe_flow import (
    RuptureType,
    PipeInput,
    PipeResult,
    colebrook_friction_factor,
    swamee_jain_friction_factor,
    darcy_weisbach_pressure_drop,
    lockhart_martinelli_multiplier,
    two_phase_pipe_pressure_drop,
    calculate_pipe_flow,
)

from .pool_evaporation import (
    PoolSurface,
    PoolRegime,
    PoolInput,
    PoolResult,
    gravity_inertia_radius,
    gravity_viscous_radius,
    transition_time,
    minimum_pool_thickness,
    mass_transfer_coefficient,
    evaporation_rate,
    boiling_evaporation_rate,
    simulate_pool,
)

__all__ = [
    # orifice
    "ReleasePhase", "OrificeFlowRegime", "OrificeInput", "OrificeResult",
    "calculate_choked_pressure", "liquid_orifice_discharge", "gas_orifice_discharge",
    "calculate_flashing_fraction", "estimate_omega_parameter",
    "two_phase_orifice_discharge", "calculate_orifice",
    "quick_liquid_orifice", "quick_gas_orifice",
    # vessel_depressur
    "VesselInput", "VesselResult", "calculate_vessel_blowdown",
    # two_phase
    "TwoPhaseModel", "TwoPhaseRegime", "TwoPhaseInput", "TwoPhaseResult",
    "tp_calculate_flash_fraction", "calculate_saturated_flash_fraction",
    "tp_calculate_omega_parameter", "hem_mass_flux",
    "fauske_slip_ratio", "moody_slip_ratio", "calculate_two_phase_flow",
    # relief_valve
    "ReliefScenario", "ValveType", "API_ORIFICE_AREAS", "API_ORIFICE_ORDER",
    "select_orifice_designation", "ReliefValveInput", "ReliefValveResult",
    "size_gas_vapor_relief", "size_liquid_relief", "size_steam_relief",
    "size_two_phase_relief", "calculate_relief_valve",
    # pipe_flow
    "RuptureType", "PipeInput", "PipeResult",
    "colebrook_friction_factor", "swamee_jain_friction_factor",
    "darcy_weisbach_pressure_drop", "lockhart_martinelli_multiplier",
    "two_phase_pipe_pressure_drop", "calculate_pipe_flow",
    # pool_evaporation
    "PoolSurface", "PoolRegime", "PoolInput", "PoolResult",
    "gravity_inertia_radius", "gravity_viscous_radius", "transition_time",
    "minimum_pool_thickness", "mass_transfer_coefficient", "evaporation_rate",
    "boiling_evaporation_rate", "simulate_pool",
]
