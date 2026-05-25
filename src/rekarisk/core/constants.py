"""
Rekarisk — Physical Constants & Regulatory Thresholds.

All constants in SI base units unless otherwise noted.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Fundamental Physical Constants
# ══════════════════════════════════════════════════════════════════════════════

# Universal gas constant [J/(mol·K)]
R = 8.314462618

# Standard gravitational acceleration [m/s²]
G = 9.80665

# Standard atmospheric pressure [Pa]
P_ATM = 101325.0

# Absolute zero (0°C in Kelvin)
T_0C = 273.15

# Avogadro constant [mol⁻¹]
N_A = 6.02214076e23

# Stefan-Boltzmann constant [W/(m²·K⁴)]
SIGMA_SB = 5.670374419e-8

# Boltzmann constant [J/K]
K_B = 1.380649e-23

# Speed of light [m/s]
C = 2.99792458e8

# Planck constant [J·s]
H = 6.62607015e-34


# ══════════════════════════════════════════════════════════════════════════════
# Thermophysical Constants
# ══════════════════════════════════════════════════════════════════════════════

# Standard conditions (IUPAC)
T_STP = 273.15            # 0 °C [K]
P_STP = 100000.0          # 1 bar [Pa]

# Normal conditions (NTP / EPA)
T_NTP = 293.15            # 20 °C [K]
P_NTP = 101325.0          # 1 atm [Pa]

# Molar volume at STP [m³/mol]
V_MOLAR_STP = 0.02241396954

# Molar volume at NTP [m³/mol]
V_MOLAR_NTP = 0.024055068

# Air properties at STP
AIR_MOLECULAR_WEIGHT = 28.9647            # [g/mol] (or kg/kmol)
AIR_DENSITY_STP = 1.2754                  # [kg/m³] at 0 °C, 1 atm
AIR_DENSITY_NTP = 1.2041                  # [kg/m³] at 20 °C, 1 atm
AIR_SPECIFIC_HEAT = 1005.0                # Cp [J/(kg·K)]
AIR_THERMAL_CONDUCTIVITY = 0.02624        # [W/(m·K)] at 300 K
AIR_DYNAMIC_VISCOSITY = 1.846e-5          # [Pa·s] at 300 K
AIR_KINEMATIC_VISCOSITY = 1.568e-5        # [m²/s] at 300 K

# Water properties
WATER_MOLECULAR_WEIGHT = 18.01528         # [g/mol]
WATER_DENSITY = 998.2071                  # [kg/m³] at 20 °C
WATER_SURFACE_TENSION = 0.07286           # [N/m] at 20 °C
WATER_LATENT_HEAT = 2264705.0             # [J/kg] at 100 °C
WATER_BOILING_POINT = 373.15              # [K]
WATER_FREEZING_POINT = 273.15             # [K]
WATER_SPECIFIC_HEAT = 4184.0              # [J/(kg·K)] liquid at 20 °C


# ══════════════════════════════════════════════════════════════════════════════
# Combustion Constants
# ══════════════════════════════════════════════════════════════════════════════

# Fraction of heat radiated (surface emissive power fraction)
# Typical values for common fuels
RADIANT_FRACTION_POOL_FIRE = {
    "LNG": 0.23,
    "LPG": 0.27,
    "gasoline": 0.30,
    "kerosene": 0.35,
    "diesel": 0.30,
    "crude_oil": 0.25,
    "methanol": 0.17,
    "ethanol": 0.22,
    "benzene": 0.36,
    "default": 0.30,
}

# Combustion efficiency [-]
COMBUSTION_EFFICIENCY = {
    "liquid_pool": 0.70,   # typical for large pool fires
    "jet_fire": 0.90,
    "bleve": 1.00,         # fireball consumes everything
    "default": 0.80,
}

# Soot surface emissive power [kW/m²]
SOOT_SEP = 20.0             # typically 20 kW/m² for sooty flames
CLEAR_FLAME_SEP = 150.0     # clear (non-sooty) flame


# ══════════════════════════════════════════════════════════════════════════════
# Wind / Meteorology Constants
# ══════════════════════════════════════════════════════════════════════════════

# Pasquill-Gifford stability classes
PG_STABILITY_CLASSES = ("A", "B", "C", "D", "E", "F")

# Description of each stability class
PG_STABILITY_DESCRIPTIONS = {
    "A": "Very unstable — sunny, light wind (< 2 m/s)",
    "B": "Moderately unstable — sunny, moderate wind (2-3 m/s)",
    "C": "Slightly unstable — sunny, moderate wind (3-5 m/s) or cloudy with wind 2-5 m/s",
    "D": "Neutral — overcast day/night or windy (> 5 m/s)",
    "E": "Slightly stable — night, light wind (2-3 m/s)",
    "F": "Moderately stable — clear night, very light wind (< 2 m/s)",
}

# Reference height for wind speed measurement [m]
WIND_REFERENCE_HEIGHT = 10.0

# Typical surface roughness lengths [m]
SURFACE_ROUGHNESS = {
    "calm_sea": 0.0002,
    "smooth_desert": 0.0003,
    "short_grass": 0.01,
    "long_grass": 0.05,
    "agricultural": 0.10,
    "suburban_light": 0.30,
    "suburban_dense": 1.00,
    "urban": 2.00,
    "industrial_complex": 3.00,
    "city_center": 5.00,
}

# Wind profile power-law exponents by stability class
WIND_PROFILE_EXPONENT = {
    "A": 0.07,
    "B": 0.07,
    "C": 0.10,
    "D": 0.15,
    "E": 0.35,
    "F": 0.55,
}


# ══════════════════════════════════════════════════════════════════════════════
# Dispersion Constants
# ══════════════════════════════════════════════════════════════════════════════

# Averaging time for dispersion calculations [s]
# Typical: 600 s (10 min) for toxic, 1800 s (30 min) for chronic
AVERAGING_TIME_DEFAULT = 600.0

# Minimum wind speed for dispersion calculations [m/s]
MIN_WIND_SPEED = 0.5

# Maximum wind speed for dispersion calculations [m/s]
MAX_WIND_SPEED = 25.0

# Default ambient temperature [K]
DEFAULT_AMBIENT_TEMP = 298.15    # 25 °C

# Default ambient pressure [Pa]
DEFAULT_AMBIENT_PRESSURE = P_ATM

# Default relative humidity [%]
DEFAULT_HUMIDITY = 50.0


# ══════════════════════════════════════════════════════════════════════════════
# Regulatory Thresholds — Thermal Radiation (kW/m²)
# ══════════════════════════════════════════════════════════════════════════════

# Kepmen LH No. 13/2000 (Indonesia) — titik ukur bahaya kebakaran
RADIATION_KEPMEN_LH = {
    "kebakaran_pipa": 12.5,         # pipa berpotensi kebakaran
    "rumah_penduduk": 12.5,         # rumah penduduk terdekat
    "jalan_umum": 18.0,             # jalan umum
    "jalan_kereta_api": 15.0,       # jalan kereta api
    "area_komersial": 12.5,         # area komersial / industri
    "border_plant": 12.5,           # batas plant
}

# Common consequence endpoints (HSE UK, API RP 521)
RADIATION_ENDPOINTS = {
    "piloted_ignition_wood": 37.5,       # spontaneously ignites wood
    "damage_process_equipment": 37.5,    # damage to process equipment
    "piloted_ignition_wood_20s": 25.0,   # ignites wood after 20 s
    "steel_failure": 25.0,               # structural steel failure
    "cable_insulation_degrade": 18.0,    # cable insulation degradation
    "significant_injury_30s": 12.5,      # 1% lethality, significant injury
    "1pct_lethality_60s": 12.5,         # 1% fatality in 60 s
    "pain_threshold_10s": 6.3,           # pain threshold ~10 s
    "pain_threshold_20s": 5.0,           # pain threshold ~20 s
    "safe_for_personnel": 1.58,          # safe for continuously exposed personnel
    "solar_radiation": 1.0,              # typical solar radiation at equator
}


# ══════════════════════════════════════════════════════════════════════════════
# Regulatory Thresholds — Overpressure (kPa)
# ══════════════════════════════════════════════════════════════════════════════

OVERPRESSURE_ENDPOINTS = {
    "building_total_destruction": 70.0,      # total destruction
    "heavy_machine_damage": 50.0,            # heavy machine damage
    "building_serious_damage": 35.0,         # serious structural damage
    "building_repairable_damage": 20.0,      # repairable structural damage
    "steel_structure_damage": 15.0,          # steel frame distortion
    "minor_structural_damage": 10.0,         # minor structural damage
    "glass_breakage_95pct": 7.0,             # 95% window breakage
    "glass_breakage_50pct": 3.5,             # 50% window breakage
    "glass_breakage_10pct": 2.0,             # 10% window breakage
    "minor_glass_breakage": 1.0,             # minor glass breakage
    "safe_distance": 0.5,                    # safe distance (no effect)
}


# ══════════════════════════════════════════════════════════════════════════════
# Regulatory Thresholds — Toxic Exposure
# ══════════════════════════════════════════════════════════════════════════════

# ERPG levels (Emergency Response Planning Guidelines)
# ERPG-1: mild transient effects
# ERPG-2: irreversible or serious health effects
# ERPG-3: life-threatening effects

# AEGL levels (Acute Exposure Guideline Levels)
# AEGL-1: notable discomfort
# AEGL-2: irreversible or serious long-lasting effects
# AEGL-3: life-threatening or death

# IDLH (Immediately Dangerous to Life or Health) — NIOSH

# Toxic endpoint defaults when specific data unavailable
TOXIC_DEFAULTS = {
    "erpg_2_factor": 10.0,      # default safety factor for ERPG-2
    "aegl_2_factor": 10.0,      # default safety factor for AEGL-2
}


# ══════════════════════════════════════════════════════════════════════════════
# QRA Constants
# ══════════════════════════════════════════════════════════════════════════════

# Risk criteria (Indonesia — Kepmen LH draft QRA guideline)
RISK_CRITERIA = {
    "individual_risk_max": 1.0e-6,              # maximum IRPA [/year] for public
    "individual_risk_negligible": 1.0e-7,       # negligible IRPA [/year]
    "individual_risk_worker_max": 1.0e-5,       # worker maximum [/year]
    "fn_slope": -1.0,                           # societal FN slope
    "fn_intercept_max": 1.0e-3,                 # unacceptable line intercept
    "fn_intercept_negligible": 1.0e-5,          # negligible line intercept
}

# Default ignition probabilities (IP Research, UK HSE)
IGNITION_PROB_IMMEDIATE = {
    "gas_continuous_small": 0.02,
    "gas_continuous_large": 0.05,
    "gas_instantaneous_small": 0.05,
    "gas_instantaneous_large": 0.20,
    "liquid_continuous_small": 0.01,
    "liquid_continuous_large": 0.03,
    "liquid_instantaneous_small": 0.02,
    "liquid_instantaneous_large": 0.10,
}

IGNITION_PROB_DELAYED_FACTOR = {
    "gas": 0.5,
    "liquid_volatile": 0.7,
    "liquid_non_volatile": 0.1,
}

# Explosion probability given flammable cloud (TNO)
EXPLOSION_PROB_FACTOR = {
    "confined_high_reactivity": 0.8,
    "confined_low_reactivity": 0.4,
    "unconfined_high_reactivity": 0.3,
    "unconfined_low_reactivity": 0.05,
}


# ══════════════════════════════════════════════════════════════════════════════
# Source Term Constants
# ══════════════════════════════════════════════════════════════════════════════

# Discharge coefficient for orifice calculations
DISCHARGE_COEFFICIENT = {
    "sharp_edge": 0.62,
    "rounded": 0.85,
    "full_bore_rupture": 1.00,
    "default": 0.62,
}

# Specific heat ratio (Cp/Cv) defaults
CP_CV_DEFAULTS = {
    "monatomic": 5.0 / 3.0,
    "diatomic": 7.0 / 5.0,
    "triatomic": 4.0 / 3.0,
    "hydrocarbon_light": 1.3,
    "hydrocarbon_heavy": 1.1,
}


# ══════════════════════════════════════════════════════════════════════════════
# Pool Fire Constants
# ══════════════════════════════════════════════════════════════════════════════

# View factor maximum (for vertical cylinder to nearby receiver)
VIEW_FACTOR_MAX = 1.0

# Default atmospheric transmissivity (when detailed data unavailable)
DEFAULT_TRANSMISSIVITY = 0.8

# Burning rate correlation constants
BURNING_RATE_DEFAULTS = {
    "k": 0.001,   # burning rate coefficient [m/s] (Burgess & Hertzberg)
    "beta": 0.0,  # wind effect coefficient
}


# ══════════════════════════════════════════════════════════════════════════════
# Explosion Constants
# ══════════════════════════════════════════════════════════════════════════════

# TNT equivalency — heat of detonation [J/kg]
TNT_HEAT_OF_DETONATION = 4.68e6

# TNT equivalency empirical yield factor
TNT_YIELD_FACTOR = 0.03   # typical for vapor cloud, range 0.01-0.10

# Blast efficiency factors for different explosion types
BLAST_EFFICIENCY = {
    "tnt": 1.0,
    "vce_unconfined": 0.03,   # Vapor Cloud Explosion, unconfined
    "vce_partly_confined": 0.10,
    "vce_confined": 0.20,
    "physical_explosion": 0.40,
    "bleve": 0.50,
}


# ══════════════════════════════════════════════════════════════════════════════
# Numerical Constants
# ══════════════════════════════════════════════════════════════════════════════

# Small number threshold for floating-point comparisons
EPSILON = 1e-12

# Maximum iterations for iterative solvers
MAX_ITER_SOLVER = 500

# Convergence tolerance for iterative solvers
TOL_SOLVER = 1e-8
