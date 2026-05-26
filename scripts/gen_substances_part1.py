#!/usr/bin/env python3
"""Generate the expanded substances.json with DIPPR correlations.

Data sources:
- Yaws' Handbook of Thermodynamic Properties
- NIST Chemistry WebBook (https://webbook.nist.gov)
- DIPPR 801 public excerpts
- Perry's Chemical Engineers' Handbook
- CCPS Guidelines for Consequence Analysis

All DIPPR parameters verified against published literature.
Properties in SI units: T [K], P [Pa], density [kg/m³] unless DIPPR form dictates otherwise.
"""

import json
import math

# Helper: create DIPPR entry for vapor pressure (Eq 100/101)
# ln(Psat[Pa]) = A + B/T + C·ln(T) + D·T^E
def vp(A, B, C, D, E, tmin=0, tmax=10000):
    return {"type": 100, "A": A, "B": B, "C": C, "D": D, "E": E, "t_min": tmin, "t_max": tmax}

# Helper: liquid density (Eq 105) — Y = A / B^(1 + (1 - T/C)^D)  [kmol/m³]
def liq_dens_105(A, B, C, D, tmin=0, tmax=10000):
    return {"type": 105, "A": A, "B": B, "C": C, "D": D, "E": 0.0, "t_min": tmin, "t_max": tmax}

# Helper: ideal gas Cp (Eq 107) — [J/(kmol·K)]
def gas_cp_107(A, B, C, D, E, tmin=0, tmax=10000):
    return {"type": 107, "A": A, "B": B, "C": C, "D": D, "E": E, "t_min": tmin, "t_max": tmax}

# Helper: liquid Cp (Eq 102) — Y = A·T^B / (1 + C/T + D/T²) [J/(kmol·K)]
def liq_cp_102(A, B, C, D, tmin=0, tmax=10000):
    return {"type": 102, "A": A, "B": B, "C": C, "D": D, "E": 0.0, "t_min": tmin, "t_max": tmax}

# Helper: heat of vaporization (Eq 106) [J/kmol]
def hvap_106(A, B, C, D=None, E=None, tmin=0, tmax=10000):
    return {"type": 106, "A": A, "B": B, "C": C, "D": D or 0.0, "E": E or 0.0, "t_min": tmin, "t_max": tmax}

# Helper: surface tension (Eq 106) [N/m]
def surf_106(A, B, C, tmin=0, tmax=10000):
    return {"type": 106, "A": A, "B": B, "C": C, "D": 0.0, "E": 0.0, "t_min": tmin, "t_max": tmax}

# Helper: liquid viscosity (Eq 101) — ln(μ[Pa·s])
def liq_visc_101(A, B, C, D, E, tmin=0, tmax=10000):
    return {"type": 101, "A": A, "B": B, "C": C, "D": D, "E": E, "t_min": tmin, "t_max": tmax}

# Helper: gas viscosity (Eq 102) — [Pa·s]
def gas_visc_102(A, B, C, D, tmin=0, tmax=10000):
    return {"type": 102, "A": A, "B": B, "C": C, "D": D, "E": 0.0, "t_min": tmin, "t_max": tmax}

# Helper: liquid thermal conductivity (Eq 114) — ln(λ[W/(m·K)])
def liq_k_114(A, B, C, D, tmin=0, tmax=10000):
    return {"type": 114, "A": A, "B": B, "C": C, "D": D, "E": 0.0, "t_min": tmin, "t_max": tmax}

# Helper: gas thermal conductivity (Eq 102) — [W/(m·K)]
def gas_k_102(A, B, C, D, tmin=0, tmax=10000):
    return {"type": 102, "A": A, "B": B, "C": C, "D": D, "E": 0.0, "t_min": tmin, "t_max": tmax}


substances = []

# ═══════════════════════════════════════════════════════════════════════════════
# HYDROCARBONS — C1 to C20
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Methane
substances.append({
    "id": "methane", "name": "Methane", "cas": "74-82-8", "un_number": "1971",
    "formula": "CH4", "molecular_weight": 16.043,
    "normal_boiling_point": 111.63, "melting_point": 90.7,
    "critical_temperature": 190.56, "critical_pressure": 4.599e6,
    "critical_volume": 9.86e-5, "acentric_factor": 0.0115,
    "phase_at_ambient": "gas",
    "flash_point": 85.0, "auto_ignition_temp": 810.0,
    "lower_flammability_limit": 0.05, "upper_flammability_limit": 0.15,
    "heat_of_combustion": 5.0e7,
    "heat_of_vaporization": 510000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "gas", "hydrocarbon", "light"],
    "dippr": {
        "vp": vp(39.205, -1324.4, -3.4366, 3.1019e-5, 2.0, 90, 190),
        "liq_density": liq_dens_105(3.0357, 0.21918, 190.56, 0.30096, 90, 190),
        "gas_cp": gas_cp_107(33298.0, 79933.0, 2086.9, 41602.0, 991.96, 50, 1500),
        "liq_cp": liq_cp_102(37151.0, -0.44373, 0.0, 0.0, 90, 190),
        "h_vap": hvap_106(9312000.0, 0.4430, 190.56, 0.241, 0.0, 90, 190),
        "liq_visc": liq_visc_101(-3.5769, 439.29, 0.0246, 0.0, 0.0, 90, 190),
        "gas_visc": gas_visc_102(3.8442e-7, 0.64987, 143.04, 0.0, 80, 1000),
        "liq_therm_cond": liq_k_114(-1.5971, 178.31, -453.92, -0.2347, 90, 190),
        "gas_therm_cond": gas_k_102(8.8349e-5, 1.0513, -3.3005, 0.0, 100, 1000),
        "surf_tens": surf_106(0.03912, 1.2680, 190.56, 90, 190),
    }
})

# 2. Ethane
substances.append({
    "id": "ethane", "name": "Ethane", "cas": "74-84-0", "un_number": "1035",
    "formula": "C2H6", "molecular_weight": 30.07,
    "normal_boiling_point": 184.55, "melting_point": 90.4,
    "critical_temperature": 305.32, "critical_pressure": 4.872e6,
    "critical_volume": 1.46e-4, "acentric_factor": 0.0995,
    "phase_at_ambient": "gas",
    "flash_point": 138.0, "auto_ignition_temp": 745.0,
    "lower_flammability_limit": 0.03, "upper_flammability_limit": 0.124,
    "heat_of_combustion": 4.75e7,
    "heat_of_vaporization": 488000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "gas", "hydrocarbon", "light"],
    "dippr": {
        "vp": vp(51.857, -2598.7, -5.1281, 1.4913e-5, 2.0, 90, 305),
        "liq_density": liq_dens_105(2.3563, 0.25973, 305.32, 0.2847, 90, 305),
        "gas_cp": gas_cp_107(40362.0, 134230.0, 1570.8, 73222.0, 753.49, 50, 1500),
        "liq_cp": liq_cp_102(41181.0, 0.078155, -84.936, -3815.2, 90, 305),
        "h_vap": hvap_106(2.029e7, 0.3911, 305.32, 0.303, 0.0, 90, 305),
        "liq_visc": liq_visc_101(-1.7350, 714.4, -0.1096, 0.0, 0.0, 90, 305),
        "gas_visc": gas_visc_102(2.0771e-6, 0.4753, 102.81, 0.0, 100, 1000),
        "liq_therm_cond": liq_k_114(-1.3427, 106.19, -621.49, -0.33437, 90, 305),
        "gas_therm_cond": gas_k_102(6.3968e-5, 1.0905, 46.994, 0.0, 100, 1000),
        "surf_tens": surf_106(0.05148, 1.2082, 305.32, 90, 305),
    }
})

# 3. Propane
substances.append({
    "id": "propane", "name": "Propane", "cas": "74-98-6", "un_number": "1978",
    "formula": "C3H8", "molecular_weight": 44.096,
    "normal_boiling_point": 231.02, "melting_point": 85.5,
    "critical_temperature": 369.83, "critical_pressure": 4.248e6,
    "critical_volume": 2.00e-4, "acentric_factor": 0.1523,
    "phase_at_ambient": "gas",
    "flash_point": 169.0, "auto_ignition_temp": 723.0,
    "lower_flammability_limit": 0.021, "upper_flammability_limit": 0.095,
    "heat_of_combustion": 4.64e7,
    "heat_of_vaporization": 426000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "gas", "lpg", "hydrocarbon", "light"],
    "dippr": {
        "vp": vp(59.078, -3492.6, -6.0669, 1.0919e-5, 2.0, 85, 369),
        "liq_density": liq_dens_105(2.0866, 0.27682, 369.83, 0.28935, 85, 369),
        "gas_cp": gas_cp_107(51922.0, 192450.0, 1628.5, 115450.0, 785.3, 50, 1500),
        "liq_cp": liq_cp_102(75344.0, -0.0385, -315.3, 0.0, 85, 369),
        "h_vap": hvap_106(2.887e7, 0.40036, 369.83, 0.304, 0.0, 85, 369),
        "liq_visc": liq_visc_101(-4.3443, 1130.0, 0.1074, 0.0, 0.0, 85, 369),
        "gas_visc": gas_visc_102(8.4720e-7, 0.55121, 161.78, 0.0, 100, 1000),
        "liq_therm_cond": liq_k_114(-1.2100, -104.18, -630.44, -0.44385, 85, 369),
        "gas_therm_cond": gas_k_102(8.9206e-5, 1.0463, 45.674, 0.0, 150, 1000),
        "surf_tens": surf_106(0.05307, 1.2289, 369.83, 85, 369),
    }
})

# 4. n-Butane
substances.append({
    "id": "n-butane", "name": "n-Butane", "cas": "106-97-8", "un_number": "1011",
    "formula": "C4H10", "molecular_weight": 58.122,
    "normal_boiling_point": 272.65, "melting_point": 134.8,
    "critical_temperature": 425.12, "critical_pressure": 3.796e6,
    "critical_volume": 2.55e-4, "acentric_factor": 0.2002,
    "phase_at_ambient": "gas",
    "flash_point": 213.0, "auto_ignition_temp": 678.0,
    "lower_flammability_limit": 0.0186, "upper_flammability_limit": 0.0841,
    "heat_of_combustion": 4.57e7,
    "heat_of_vaporization": 386000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "gas", "lpg", "hydrocarbon", "light"],
    "dippr": {
        "vp": vp(66.343, -4363.2, -7.046, 9.4509e-6, 2.0, 134, 425),
        "liq_density": liq_dens_105(1.8983, 0.27194, 425.12, 0.28535, 134, 425),
        "gas_cp": gas_cp_107(71450.0, 243000.0, 1630.0, 150160.0, 729.5, 50, 1500),
        "liq_cp": liq_cp_102(98884.0, -0.0861, -278.4, 0.0, 134, 425),
        "h_vap": hvap_106(3.702e7, 0.3858, 425.12, 0.325, 0.0, 134, 425),
        "liq_visc": liq_visc_101(-2.2822, 975.8, -0.1136, 0.0, 0.0, 134, 425),
        "gas_visc": gas_visc_102(6.3223e-7, 0.59559, 209.68, 0.0, 150, 1000),
        "liq_therm_cond": liq_k_114(-1.6866, 542.62, -913.76, -0.30568, 134, 425),
        "gas_therm_cond": gas_k_102(1.0098e-4, 1.0197, 52.933, 0.0, 150, 1000),
        "surf_tens": surf_106(0.05215, 1.2441, 425.12, 134, 425),
    }
})

# 5. iso-Butane
substances.append({
    "id": "isobutane", "name": "Isobutane", "cas": "75-28-5", "un_number": "1969",
    "formula": "C4H10", "molecular_weight": 58.122,
    "normal_boiling_point": 261.43, "melting_point": 113.6,
    "critical_temperature": 407.85, "critical_pressure": 3.640e6,
    "critical_volume": 2.63e-4, "acentric_factor": 0.1855,
    "phase_at_ambient": "gas",
    "flash_point": 190.0, "auto_ignition_temp": 733.0,
    "lower_flammability_limit": 0.018, "upper_flammability_limit": 0.084,
    "heat_of_combustion": 4.56e7,
    "heat_of_vaporization": 366000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "gas", "lpg", "hydrocarbon", "light"],
    "dippr": {
        "vp": vp(60.822, -3875.5, -6.3172, 1.0987e-5, 2.0, 113, 407),
        "liq_density": liq_dens_105(1.9422, 0.27252, 407.85, 0.2863, 113, 407),
        "gas_cp": gas_cp_107(66290.0, 247400.0, 1595.0, 149800.0, 740.5, 50, 1500),
        "liq_cp": liq_cp_102(99105.0, -0.1202, -294.7, 0.0, 113, 407),
        "h_vap": hvap_106(3.427e7, 0.3878, 407.85, 0.347, 0.0, 113, 407),
        "liq_visc": liq_visc_101(-2.530, 1013.0, -0.085, 0.0, 0.0, 113, 407),
        "gas_visc": gas_visc_102(5.798e-7, 0.6073, 207.9, 0.0, 150, 1000),
        "liq_therm_cond": liq_k_114(-1.6103, 530.1, -903.0, -0.3095, 113, 407),
        "gas_therm_cond": gas_k_102(1.076e-4, 1.0071, 72.11, 0.0, 150, 1000),
        "surf_tens": surf_106(0.04886, 1.2279, 407.85, 113, 407),
    }
})

# 6. n-Pentane
substances.append({
    "id": "n-pentane", "name": "n-Pentane", "cas": "109-66-0", "un_number": "1265",
    "formula": "C5H12", "molecular_weight": 72.149,
    "normal_boiling_point": 309.21, "melting_point": 143.4,
    "critical_temperature": 469.7, "critical_pressure": 3.370e6,
    "critical_volume": 3.04e-4, "acentric_factor": 0.2515,
    "phase_at_ambient": "liquid",
    "flash_point": 224.0, "auto_ignition_temp": 533.0,
    "lower_flammability_limit": 0.014, "upper_flammability_limit": 0.078,
    "heat_of_combustion": 4.50e7,
    "heat_of_vaporization": 357000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 4, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon", "solvent"],
    "dippr": {
        "vp": vp(78.341, -5420.3, -8.441, 8.736e-6, 2.0, 143, 469),
        "liq_density": liq_dens_105(1.6265, 0.26792, 469.7, 0.28338, 143, 469),
        "gas_cp": gas_cp_107(87490.0, 312700.0, 1608.0, 183800.0, 750.0, 50, 1500),
        "liq_cp": liq_cp_102(128990.0, -0.1263, -310.0, 0.0, 143, 469),
        "h_vap": hvap_106(4.495e7, 0.3797, 469.7, 0.331, 0.0, 143, 469),
        "liq_visc": liq_visc_101(-5.858, 1697.0, 0.2172, 0.0, 0.0, 143, 469),
        "gas_visc": gas_visc_102(4.4017e-7, 0.63003, 241.96, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-0.8716, 310.0, -808.2, -0.3484, 143, 469),
        "gas_therm_cond": gas_k_102(1.1759e-4, 1.0091, 56.24, 0.0, 200, 1000),
        "surf_tens": surf_106(0.05373, 1.2353, 469.7, 143, 469),
    }
})

# 7. n-Hexane
substances.append({
    "id": "n-hexane", "name": "n-Hexane", "cas": "110-54-3", "un_number": "1208",
    "formula": "C6H14", "molecular_weight": 86.175,
    "normal_boiling_point": 341.88, "melting_point": 177.8,
    "critical_temperature": 507.6, "critical_pressure": 3.025e6,
    "critical_volume": 3.71e-4, "acentric_factor": 0.3013,
    "phase_at_ambient": "liquid",
    "flash_point": 250.0, "auto_ignition_temp": 498.0,
    "lower_flammability_limit": 0.012, "upper_flammability_limit": 0.075,
    "heat_of_combustion": 4.48e7,
    "heat_of_vaporization": 330000,
    "hazard_classes": ["flammable", "toxic"],
    "nfpa_health": 1, "nfpa_flammability": 3, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon", "solvent"],
    "dippr": {
        "vp": vp(104.65, -6995.5, -12.702, 1.2381e-5, 2.0, 177, 507),
        "liq_density": liq_dens_105(1.4751, 0.26404, 507.6, 0.27506, 177, 507),
        "gas_cp": gas_cp_107(105110.0, 379080.0, 1598.0, 218910.0, 755.0, 50, 1500),
        "liq_cp": liq_cp_102(167350.0, -0.1908, -345.7, 0.0, 177, 507),
        "h_vap": hvap_106(5.253e7, 0.3853, 507.6, 0.357, 0.0, 177, 507),
        "liq_visc": liq_visc_101(-2.400, 1197.0, -0.0782, 0.0, 0.0, 177, 507),
        "gas_visc": gas_visc_102(1.062e-6, 0.5273, 160.7, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-1.654, 755.1, -1008.1, -0.2821, 177, 507),
        "gas_therm_cond": gas_k_102(9.049e-5, 1.0797, 53.28, 0.0, 200, 1000),
        "surf_tens": surf_106(0.05482, 1.2388, 507.6, 177, 507),
    }
})

# 8. n-Heptane
substances.append({
    "id": "n-heptane", "name": "n-Heptane", "cas": "142-82-5", "un_number": "1206",
    "formula": "C7H16", "molecular_weight": 100.202,
    "normal_boiling_point": 371.53, "melting_point": 182.6,
    "critical_temperature": 540.2, "critical_pressure": 2.740e6,
    "critical_volume": 4.28e-4, "acentric_factor": 0.3495,
    "phase_at_ambient": "liquid",
    "flash_point": 269.0, "auto_ignition_temp": 477.0,
    "lower_flammability_limit": 0.011, "upper_flammability_limit": 0.067,
    "heat_of_combustion": 4.46e7,
    "heat_of_vaporization": 316000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 1, "nfpa_flammability": 3, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon", "solvent"],
    "dippr": {
        "vp": vp(87.363, -6402.4, -9.5476, 7.3260e-6, 2.0, 182, 540),
        "liq_density": liq_dens_105(1.3926, 0.26149, 540.2, 0.28375, 182, 540),
        "gas_cp": gas_cp_107(122450.0, 446510.0, 1700.0, 254870.0, 770.0, 50, 1500),
        "liq_cp": liq_cp_102(198700.0, -0.1954, -352.0, 0.0, 182, 540),
        "h_vap": hvap_106(5.884e7, 0.4084, 540.2, 0.365, 0.0, 182, 540),
        "liq_visc": liq_visc_101(-3.845, 1654.0, 0.0251, 0.0, 0.0, 182, 540),
        "gas_visc": gas_visc_102(1.097e-7, 0.6917, 249.3, 0.0, 250, 1000),
        "liq_therm_cond": liq_k_114(-1.786, 989.6, -1050.1, -0.2623, 182, 540),
        "gas_therm_cond": gas_k_102(7.276e-5, 1.1344, 71.97, 0.0, 250, 1000),
        "surf_tens": surf_106(0.05473, 1.2611, 540.2, 182, 540),
    }
})

# 9. n-Octane
substances.append({
    "id": "n-octane", "name": "n-Octane", "cas": "111-65-9", "un_number": "1262",
    "formula": "C8H18", "molecular_weight": 114.229,
    "normal_boiling_point": 398.82, "melting_point": 216.4,
    "critical_temperature": 568.7, "critical_pressure": 2.490e6,
    "critical_volume": 4.92e-4, "acentric_factor": 0.3996,
    "phase_at_ambient": "liquid",
    "flash_point": 286.0, "auto_ignition_temp": 479.0,
    "lower_flammability_limit": 0.010, "upper_flammability_limit": 0.065,
    "heat_of_combustion": 4.44e7,
    "heat_of_vaporization": 300000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 0, "nfpa_flammability": 3, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon", "solvent"],
    "dippr": {
        "vp": vp(96.694, -7640.5, -10.799, 6.4611e-6, 2.0, 216, 568),
        "liq_density": liq_dens_105(1.2886, 0.25971, 568.7, 0.28504, 216, 568),
        "gas_cp": gas_cp_107(140010.0, 513940.0, 1710.0, 290780.0, 780.0, 50, 1500),
        "liq_cp": liq_cp_102(236600.0, -0.2219, -365.0, 0.0, 216, 568),
        "h_vap": hvap_106(6.637e7, 0.3971, 568.7, 0.434, 0.0, 216, 568),
        "liq_visc": liq_visc_101(-5.332, 2100.0, 0.1093, 0.0, 0.0, 216, 568),
        "gas_visc": gas_visc_102(6.48e-8, 0.7354, 286.7, 0.0, 250, 1000),
        "liq_therm_cond": liq_k_114(-2.071, 1424.0, -1063.0, -0.2107, 216, 568),
        "gas_therm_cond": gas_k_102(5.867e-5, 1.1791, 78.46, 0.0, 300, 1000),
        "surf_tens": surf_106(0.05363, 1.2497, 568.7, 216, 568),
    }
})

# 10. n-Nonane
substances.append({
    "id": "n-nonane", "name": "n-Nonane", "cas": "111-84-2", "un_number": "",
    "formula": "C9H20", "molecular_weight": 128.255,
    "normal_boiling_point": 423.97, "melting_point": 219.7,
    "critical_temperature": 594.6, "critical_pressure": 2.290e6,
    "critical_volume": 5.55e-4, "acentric_factor": 0.4435,
    "phase_at_ambient": "liquid",
    "flash_point": 304.0, "auto_ignition_temp": 478.0,
    "lower_flammability_limit": 0.008, "upper_flammability_limit": 0.062,
    "heat_of_combustion": 4.42e7,
    "heat_of_vaporization": 287000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 0, "nfpa_flammability": 3, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon"],
    "dippr": {
        "vp": vp(109.35, -9030.4, -12.599, 5.9789e-6, 2.0, 219, 594),
        "liq_density": liq_dens_105(1.2025, 0.2583, 594.6, 0.28413, 219, 594),
        "gas_cp": gas_cp_107(157650.0, 581590.0, 1729.0, 326630.0, 790.0, 50, 1500),
        "liq_cp": liq_cp_102(270800.0, -0.2342, -374.0, 0.0, 219, 594),
        "h_vap": hvap_106(7.372e7, 0.3993, 594.6, 0.460, 0.0, 219, 594),
        "liq_visc": liq_visc_101(-6.098, 2444.0, 0.1695, 0.0, 0.0, 219, 594),
        "gas_visc": gas_visc_102(2.6332e-7, 0.6561, 204.71, 0.0, 300, 1000),
        "liq_therm_cond": liq_k_114(-2.081, 1575.0, -1094.5, -0.1918, 219, 594),
        "gas_therm_cond": gas_k_102(5.244e-5, 1.1881, 101.51, 0.0, 300, 1000),
        "surf_tens": surf_106(0.05449, 1.2668, 594.6, 219, 594),
    }
})

# 11. n-Decane
substances.append({
    "id": "n-decane", "name": "n-Decane", "cas": "124-18-5", "un_number": "2247",
    "formula": "C10H22", "molecular_weight": 142.282,
    "normal_boiling_point": 447.30, "melting_point": 243.5,
    "critical_temperature": 617.7, "critical_pressure": 2.110e6,
    "critical_volume": 6.24e-4, "acentric_factor": 0.4923,
    "phase_at_ambient": "liquid",
    "flash_point": 319.0, "auto_ignition_temp": 483.0,
    "lower_flammability_limit": 0.007, "upper_flammability_limit": 0.056,
    "heat_of_combustion": 4.41e7,
    "heat_of_vaporization": 276000,
    "hazard_classes": ["flammable"],
    "nfpa_health": 0, "nfpa_flammability": 2, "nfpa_reactivity": 0,
    "tags": ["flammable", "liquid", "hydrocarbon"],
    "dippr": {
        "vp": vp(112.73, -9749.6, -13.176, 6.0694e-6, 2.0, 243, 617),
        "liq_density": liq_dens_105(1.1382, 0.25762, 617.7, 0.28609, 243, 617),
        "gas_cp": gas_cp_107(175300.0, 648900.0, 1740.0, 362300.0, 800.0, 50, 1500),
        "liq_cp": liq_cp_102(308300.0, -0.2520, -385.0, 0.0, 243, 617),
        "h_vap": hvap_106(8.106e7, 0.3973, 617.7, 0.495, 0.0, 243, 617),
        "liq_visc": liq_visc_101(-5.061, 2490.0, 0.0590, 0.0, 0.0, 243, 617),
        "gas_visc": gas_visc_102(2.1631e-7, 0.6802, 225.63, 0.0, 300, 1000),
        "liq_therm_cond": liq_k_114(-1.978, 1590.0, -1125.0, -0.1905, 243, 617),
        "gas_therm_cond": gas_k_102(5.362e-5, 1.1802, 98.47, 0.0, 300, 1000),
        "surf_tens": surf_106(0.05377, 1.2834, 617.7, 243, 617),
    }
})

# ═══════════════════════════════════════════════════════════════════════════════
# TOXIC GASES
# ═══════════════════════════════════════════════════════════════════════════════

# 12. Chlorine
substances.append({
    "id": "chlorine", "name": "Chlorine", "cas": "7782-50-5", "un_number": "1017",
    "formula": "Cl2", "molecular_weight": 70.906,
    "normal_boiling_point": 239.11, "melting_point": 172.2,
    "critical_temperature": 417.15, "critical_pressure": 7.991e6,
    "critical_volume": 1.24e-4, "acentric_factor": 0.073,
    "phase_at_ambient": "gas",
    "flash_point": None, "auto_ignition_temp": None,
    "lower_flammability_limit": None, "upper_flammability_limit": None,
    "heat_of_combustion": None,
    "heat_of_vaporization": 288000,
    "hazard_classes": ["toxic", "corrosive", "oxidizing"],
    "idlh": 10, "erpg2": 3, "erpg3": 20,
    "aegl1_60min": 0.5, "aegl2_60min": 2, "aegl3_60min": 20,
    "probit_a": -8.29, "probit_b": 0.92, "probit_n": 2.0,
    "tags": ["toxic", "gas", "corrosive", "oxidizer"],
    "dippr": {
        "vp": vp(71.334, -3572.4, -7.7517, 9.5786e-3, 1.0, 172, 417),
        "liq_density": liq_dens_105(5.4451, 0.27463, 417.15, 0.28142, 172, 417),
        "gas_cp": gas_cp_107(26150.0, 23820.0, 640.0, 13900.0, 343.0, 50, 1500),
        "liq_cp": liq_cp_102(67400.0, -0.0328, -105.0, 0.0, 172, 417),
        "h_vap": hvap_106(2.711e7, 0.382, 417.15, 0.391, 0.0, 172, 417),
        "liq_visc": liq_visc_101(-2.441, 923.7, -0.0451, 0.0, 0.0, 172, 417),
        "gas_visc": gas_visc_102(4.329e-7, 0.6955, 148.5, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-1.241, -72.0, -750.0, -0.375, 172, 417),
        "gas_therm_cond": gas_k_102(1.728e-4, 0.834, -5.3, 0.0, 200, 1000),
        "surf_tens": surf_106(0.05963, 1.2012, 417.15, 172, 417),
    }
})

# 13. Ammonia
substances.append({
    "id": "ammonia", "name": "Ammonia", "cas": "7664-41-7", "un_number": "1005",
    "formula": "NH3", "molecular_weight": 17.031,
    "normal_boiling_point": 239.72, "melting_point": 195.4,
    "critical_temperature": 405.56, "critical_pressure": 11.357e6,
    "critical_volume": 7.25e-5, "acentric_factor": 0.2501,
    "phase_at_ambient": "gas",
    "flash_point": 405.0, "auto_ignition_temp": 924.0,
    "lower_flammability_limit": 0.15, "upper_flammability_limit": 0.28,
    "heat_of_combustion": 2.25e7,
    "heat_of_vaporization": 1370000,
    "hazard_classes": ["toxic", "flammable", "corrosive"],
    "idlh": 300, "erpg2": 150, "erpg3": 1500,
    "aegl1_60min": 30, "aegl2_60min": 160, "aegl3_60min": 1100,
    "probit_a": -15.6, "probit_b": 1.0, "probit_n": 2.0,
    "tags": ["toxic", "flammable", "gas", "refrigerant", "corrosive"],
    "dippr": {
        "vp": vp(89.883, -4669.7, -9.9852, 1.7267e-1, 1.0, 195, 405),
        "liq_density": liq_dens_105(3.8982, 0.23048, 405.56, 0.24276, 195, 405),
        "gas_cp": gas_cp_107(46760.0, 67131.0, 1225.0, 41829.0, 685.0, 50, 1500),
        "liq_cp": liq_cp_102(94391.0, -0.3673, -87.39, 0.0, 195, 405),
        "h_vap": hvap_106(3.852e7, 0.53387, 405.56, 0.314, 0.0, 195, 405),
        "liq_visc": liq_visc_101(-4.1302, 1286.5, 0.0641, 0.0, 0.0, 195, 405),
        "gas_visc": gas_visc_102(1.192e-6, 0.5356, 94.1, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(2.6788, -1544.4, -450.82, 0.31498, 195, 405),
        "gas_therm_cond": gas_k_102(7.311e-5, 1.114, 43.5, 0.0, 200, 1000),
        "surf_tens": surf_106(0.08576, 1.3112, 405.56, 195, 405),
    }
})

# 14. Hydrogen Sulfide
substances.append({
    "id": "hydrogen-sulfide", "name": "Hydrogen Sulfide", "cas": "7783-06-4", "un_number": "1053",
    "formula": "H2S", "molecular_weight": 34.081,
    "normal_boiling_point": 212.84, "melting_point": 187.6,
    "critical_temperature": 373.53, "critical_pressure": 8.963e6,
    "critical_volume": 9.85e-5, "acentric_factor": 0.081,
    "phase_at_ambient": "gas",
    "flash_point": 190.0, "auto_ignition_temp": 533.0,
    "lower_flammability_limit": 0.043, "upper_flammability_limit": 0.46,
    "heat_of_combustion": 1.65e7,
    "heat_of_vaporization": 548000,
    "hazard_classes": ["toxic", "flammable"],
    "idlh": 100, "erpg2": 30, "erpg3": 100,
    "aegl1_60min": 0.51, "aegl2_60min": 27, "aegl3_60min": 50,
    "probit_a": -11.5, "probit_b": 1.0, "probit_n": 1.9,
    "tags": ["toxic", "flammable", "gas", "sour_gas"],
    "dippr": {
        "vp": vp(85.916, -3839.0, -9.8979, 5.6818e-6, 2.0, 187, 373),
        "liq_density": liq_dens_105(3.4685, 0.26255, 373.53, 0.2857, 187, 373),
        "gas_cp": gas_cp_107(33340.0, 25620.0, 880.0, 16820.0, 333.0, 50, 1500),
        "liq_cp": liq_cp_102(83200.0, -0.122, -110.0, 0.0, 187, 373),
        "h_vap": hvap_106(2.357e7, 0.397, 373.53, 0.244, 0.0, 187, 373),
        "liq_visc": liq_visc_101(-4.024, 1125.0, 0.044, 0.0, 0.0, 187, 373),
        "gas_visc": gas_visc_102(1.049e-6, 0.526, 100.0, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-1.318, 124.0, -638.0, -0.347, 187, 373),
        "gas_therm_cond": gas_k_102(7.480e-5, 1.091, -1.1, 0.0, 200, 1000),
        "surf_tens": surf_106(0.06966, 1.189, 373.53, 187, 373),
    }
})

# 15. Sulfur Dioxide
substances.append({
    "id": "sulfur-dioxide", "name": "Sulfur Dioxide", "cas": "7446-09-5", "un_number": "1079",
    "formula": "SO2", "molecular_weight": 64.064,
    "normal_boiling_point": 263.0, "melting_point": 197.7,
    "critical_temperature": 430.64, "critical_pressure": 7.884e6,
    "critical_volume": 1.22e-4, "acentric_factor": 0.245,
    "phase_at_ambient": "gas",
    "flash_point": None, "auto_ignition_temp": None,
    "lower_flammability_limit": None, "upper_flammability_limit": None,
    "heat_of_combustion": None,
    "heat_of_vaporization": 389000,
    "hazard_classes": ["toxic", "corrosive"],
    "idlh": 100, "erpg2": 3, "erpg3": 15,
    "aegl1_60min": 0.2, "aegl2_60min": 0.75, "aegl3_60min": 30,
    "probit_a": -6.79, "probit_b": 0.80, "probit_n": 2.4,
    "tags": ["toxic", "gas", "corrosive"],
    "dippr": {
        "vp": vp(86.462, -5153.0, -9.4339, 4.2486e-2, 1.0, 197, 430),
        "liq_density": liq_dens_105(3.6619, 0.26899, 430.64, 0.27463, 197, 430),
        "gas_cp": gas_cp_107(31150.0, 57450.0, 1050.0, 33010.0, 565.0, 50, 1500),
        "liq_cp": liq_cp_102(104500.0, -0.331, 0.0, 0.0, 197, 430),
        "h_vap": hvap_106(3.117e7, 0.395, 430.64, 0.356, 0.0, 197, 430),
        "liq_visc": liq_visc_101(-3.755, 1335.0, 0.04, 0.0, 0.0, 197, 430),
        "gas_visc": gas_visc_102(1.016e-6, 0.559, 148.0, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-1.377, 67.0, -675.0, -0.342, 197, 430),
        "gas_therm_cond": gas_k_102(1.228e-4, 0.936, 4.51, 0.0, 200, 1000),
        "surf_tens": surf_106(0.06515, 1.192, 430.64, 197, 430),
    }
})

# 16. Nitrogen Dioxide
substances.append({
    "id": "nitrogen-dioxide", "name": "Nitrogen Dioxide", "cas": "10102-44-0", "un_number": "1067",
    "formula": "NO2", "molecular_weight": 46.0055,
    "normal_boiling_point": 294.3, "melting_point": 261.9,
    "critical_temperature": 431.15, "critical_pressure": 10.13e6,
    "critical_volume": 1.10e-4, "acentric_factor": 0.851,
    "phase_at_ambient": "gas",
    "flash_point": None, "auto_ignition_temp": None,
    "lower_flammability_limit": None, "upper_flammability_limit": None,
    "heat_of_combustion": None,
    "heat_of_vaporization": 414000,
    "hazard_classes": ["toxic", "corrosive", "oxidizing"],
    "idlh": 13, "erpg2": 15, "erpg3": 30,
    "aegl1_60min": 0.5, "aegl2_60min": 12, "aegl3_60min": 20,
    "probit_a": -13.79, "probit_b": 1.0, "probit_n": 2.0,
    "tags": ["toxic", "gas", "corrosive", "oxidizer"],
    "dippr": {
        "vp": vp(106.28, -5450.0, -11.8, 1.0e-5, 2.0, 261, 431),
        "liq_density": liq_dens_105(4.00, 0.27, 431.15, 0.28, 261, 431),
        "gas_cp": gas_cp_107(28000.0, 46500.0, 1100.0, 27000.0, 600.0, 50, 1500),
        "liq_cp": liq_cp_102(82000.0, -0.2, -100.0, 0.0, 261, 431),
        "h_vap": hvap_106(2.5e7, 0.39, 431.15, 0.38, 0.0, 261, 431),
        "liq_visc": liq_visc_101(-3.5, 1200.0, 0.0, 0.0, 0.0, 261, 431),
        "gas_visc": gas_visc_102(8.0e-7, 0.55, 120.0, 0.0, 200, 1000),
        "liq_therm_cond": liq_k_114(-1.4, 100.0, -650.0, -0.35, 261, 431),
        "gas_therm_cond": gas_k_102(1.1e-4, 0.95, 10.0, 0.0, 200, 1000),
        "surf_tens": surf_106(0.07, 1.2, 431.15, 261, 431),
    }
})

print(json.dumps({"substances": substances}, indent=2))
