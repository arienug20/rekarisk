#!/usr/bin/env python3
"""
QRA Comparison: Rekarisk vs SAFETI (from FNKT-20-P1-SR-007)

Uses the same input data as the NKT QRA study and compares
LSIR, IRPA, and PLL results against SAFETI/PHAST results.
"""
import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

OUT = '/home/arienugraha-rei/.openclaw/workspace/outputs/qra_comparison'
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA FROM QRA DOCUMENT (FNKT-20-P1-SR-007)
# ══════════════════════════════════════════════════════════════════════════════

# 11 location points with SAFETI LSIR results (per year)
SAFETI_LSIR = {
    "Process Area NKT":         1.86e-4,
    "Process Area CPPG North":  4.09e-5,
    "Process Area CPPG South":  2.54e-5,
    "Substation Building":      1.95e-4,
    "Control Room NKT":         3.32e-4,
    "Control Room CPPG":        2.32e-4,
    "Support Area":             2.16e-4,
    "Security & Guard West":    2.19e-4,
    "Security & Guard North":   3.32e-4,
    "Metering Area":            3.44e-4,
    "Utility Area":             3.23e-4,
}

# SAFETI IRPA per worker (process hazard only)
SAFETI_IRPA_PROCESS = {
    "Operator":                  7.16e-5,
    "Asmen Production":          2.92e-5,
    "Asmen RAM":                 2.92e-5,
    "Shift Supervisor Day":      1.58e-5,
    "Shift Supervisor Night":    1.58e-5,
    "Sr Operator DCS Day":       5.81e-5,
    "Sr Operator DCS Night":     5.81e-5,
    "Sr Operator Process Day":   1.58e-5,
    "Sr Operator Process Night": 1.58e-5,
    "Operator DCS Day":          5.81e-5,
    "Operator DCS Night":        5.81e-5,
    "Field Operator Day":        1.58e-5,
    "Field Operator Night":      1.58e-5,
    "Supervisor Well":           5.53e-6,
    "Operator Well Day":         8.29e-6,
    "Operator Well Night":       8.29e-6,
}

# Total IRPA (process + work + transport)
SAFETI_IRPA_TOTAL = {
    "Operator NKT":              1.34e-4,
    "Asmen Production":          7.96e-5,
    "Asmen RAM":                 7.96e-5,
    "Sr Operator DCS Day":       9.64e-5,
    "Sr Operator DCS Night":     9.64e-5,
    "Guard Post 2 Day":          1.21e-4,
    "Warehouse":                 1.04e-4,
}

# SAFETI total PLL
SAFETI_PLL_TOTAL = 8.98e-3

# 7 Isolatable Sections (same as FERA)
ISO_SECTIONS = {
    "ISO 1": {"P_pa": 450.05*6894.76+101325, "T_K": (90.1-32)*5/9+273.15, 
              "vol_m3": 0.564, "phase": "gas", "MW": 20.5},
    "ISO 2G": {"P_pa": 450.05*6894.76+101325, "T_K": (90.1-32)*5/9+273.15, 
               "vol_m3": 2.6232, "phase": "gas", "MW": 20.5},
    "ISO 2L": {"P_pa": 450.05*6894.76+101325, "T_K": (90.1-32)*5/9+273.15, 
               "vol_m3": 2.40264, "phase": "liquid", "MW": 20.5},
    "ISO 3G": {"P_pa": 440.05*6894.76+101325, "T_K": (89.4-32)*5/9+273.15, 
               "vol_m3": 0.8268, "phase": "gas", "MW": 20.5},
    "ISO 3L": {"P_pa": 440.05*6894.76+101325, "T_K": (89.4-32)*5/9+273.15, 
               "vol_m3": 0.792, "phase": "liquid", "MW": 20.5},
    "ISO 4":  {"P_pa": 150.0*6894.76+101325, "T_K": (86.01-32)*5/9+273.15, 
               "vol_m3": 0.655, "phase": "liquid", "MW": 20.5},
    "ISO 5":  {"P_pa": 430.05*6894.76+101325, "T_K": (88.7-32)*5/9+273.15, 
               "vol_m3": 15.767, "phase": "gas", "MW": 20.5},
}

# Hole sizes from QRA doc (note: QRA uses 152.4mm fullbore, not 200mm)
HOLE_SIZES = {"Small": 5, "Medium": 50, "Large": 100, "Fullbore": 152.4}

# Leak frequencies from OGP/HSE (typical for QRA)
# Frequency per year per hole size per isolatable section
LEAK_FREQ = {
    "Small":    5.0e-4,
    "Medium":   5.0e-5,
    "Large":    2.0e-5,
    "Fullbore": 5.0e-6,
}

# Location coordinates (approximate, relative to NKT process area center)
LOCATIONS = {
    "Process Area NKT":         (0, 0),
    "Process Area CPPG North":  (-80, 60),
    "Process Area CPPG South":  (-80, -40),
    "Substation Building":      (-30, 50),
    "Control Room NKT":         (15, 25),
    "Control Room CPPG":        (-60, 40),
    "Support Area":             (-20, -30),
    "Security & Guard West":    (-100, 0),
    "Security & Guard North":   (0, 80),
    "Metering Area":            (25, -10),
    "Utility Area":             (-15, -50),
}

# Shelter factors (SAFETI accounts for building protection)
SHELTER_FACTOR = {
    "Process Area NKT":         1.0,   # outdoor
    "Process Area CPPG North":  1.0,
    "Process Area CPPG South":  1.0,
    "Substation Building":      0.3,   # inside building
    "Control Room NKT":         0.2,   # blast-rated control room
    "Control Room CPPG":        0.2,
    "Support Area":             0.8,   # partial cover
    "Security & Guard West":    0.7,   # guard post
    "Security & Guard North":   0.7,
    "Metering Area":            1.0,   # outdoor
    "Utility Area":             0.8,
}

# ESD/isolation probability
# SAFETI typically includes all scenarios — ESD reduces consequence duration, not frequency
# But for this comparison, we apply a small discount for large holes where ESD is effective
ESD_EFFECTIVENESS = {"Small": 1.0, "Medium": 1.0, "Large": 0.7, "Fullbore": 0.5}

# Probability of immediate ignition varies by hole size
P_IMM_IGNITION = {
    "Small":    0.01,
    "Medium":   0.06,
    "Large":    0.15,
    "Fullbore": 0.30,
}

# Probability of delayed ignition (given no immediate ignition)
P_DEL_IGNITION = {
    "Small":    0.02,
    "Medium":   0.10,
    "Large":    0.30,
    "Fullbore": 0.50,
}

# Weather scenarios
WEATHER = [
    ("1.35C", 1.35, "C"),  # Pasquill C, 1.35 m/s
    ("5.5D",  5.5,  "D"),  # Pasquill D, 5.5 m/s
]
WEATHER_PROB = {"1.35C": 0.15, "5.5D": 0.85}  # typical probability

# Release locations (approximate, relative to site center)
# Each ISO section has equipment at different locations
RELEASE_LOCATIONS = {
    "ISO 1":  (10, 0),       # Near metering skid
    "ISO 2G": (5, -5),       # Separator gas outlet
    "ISO 2L": (5, -5),       # Separator liquid outlet
    "ISO 3G": (0, 5),        # Cooler gas side
    "ISO 3L": (0, 5),        # Cooler liquid side
    "ISO 4":  (-5, -10),     # Export pump area
    "ISO 5":  (20, 0),       # Metering/run pipe
}

# Worker occupancy at each location (fraction of time)
# Approximated from typical NKT operations
WORKER_OCCUPANCY = {
    "Operator": {
        "Process Area NKT": 0.40, "Metering Area": 0.15, "Control Room NKT": 0.20,
        "Utility Area": 0.10, "Support Area": 0.15,
    },
    "Sr Operator DCS Day": {
        "Control Room NKT": 0.70, "Process Area NKT": 0.20, "Metering Area": 0.10,
    },
    "Sr Operator DCS Night": {
        "Control Room NKT": 0.70, "Process Area NKT": 0.20, "Metering Area": 0.10,
    },
    "Shift Supervisor Day": {
        "Control Room NKT": 0.50, "Process Area NKT": 0.30, "Utility Area": 0.20,
    },
    "Field Operator Day": {
        "Process Area NKT": 0.50, "Metering Area": 0.20, "Utility Area": 0.30,
    },
    "Supervisor Well": {
        "Process Area NKT": 0.10, "Support Area": 0.60, "Utility Area": 0.30,
    },
    "Operator Well Day": {
        "Process Area NKT": 0.15, "Support Area": 0.50, "Utility Area": 0.35,
    },
    "Guard Post 2 Day": {
        "Security & Guard North": 0.90, "Process Area NKT": 0.10,
    },
    "Warehouse": {
        "Support Area": 0.60, "Process Area NKT": 0.10, "Utility Area": 0.30,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# REKARISK CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def calc_release_rate(iso, d_hole_mm):
    """Calculate release rate for given ISO section and hole size."""
    from rekarisk.models.source_term.orifice import gas_orifice_discharge
    
    P_up = iso["P_pa"]
    P_down = 101325.0
    T = iso["T_K"]
    d = d_hole_mm / 1000.0
    A = math.pi * (d/2)**2
    
    if iso["phase"] == "gas":
        try:
            result = gas_orifice_discharge(Cd=0.62, area=A, P_up=P_up, P_down=P_down,
                                           k=1.3, T=T, MW=iso["MW"]/1000)
            return result.get("mdot", result.get("mass_flow_rate", 0))
        except:
            return 0
    else:
        # Liquid Bernoulli
        rho_l = 550 if "2L" in list(ISO_SECTIONS.keys())[list(ISO_SECTIONS.values()).index(iso)] else 700
        return 0.61 * A * math.sqrt(2 * 550 * (P_up - P_down))


def calc_jet_fire_distance(mdot_kg_s, d_hole_mm, threshold_kWm2):
    """Calculate distance to thermal radiation threshold."""
    from rekarisk.models.fire.jet_fire import (
        flame_length_chamberlain_hrr, distance_to_thresholds_jet_multipoint,
    )
    
    dHc = 47.5e6  # J/kg
    Q_kW = mdot_kg_s * dHc / 1000
    Q_W = Q_kW * 1000
    
    try:
        L = flame_length_chamberlain_hrr(Q_kW)
    except:
        L = 0.235 * Q_kW**0.385
    
    try:
        thresholds = distance_to_thresholds_jet_multipoint(
            total_heat_release=Q_W, radiative_fraction=0.35,
            flame_length=L, flame_width=0.12*L,
            tilt_deg=0, center_height=4.0,
            ambient_temperature=300.65, relative_humidity=82.32,
            thresholds=[threshold_kWm2],
        )
        return thresholds.get(threshold_kWm2, 0)
    except:
        return 0


def calc_flash_fire_distance(mdot_kg_s, d_hole_mm, wind_ms, stability):
    """Calculate flash fire distance (50% LFL)."""
    from rekarisk.models.dispersion.gaussian_plume import sigma_y, sigma_z
    
    LFL = 0.05 * (20.5/1000) * 101325 / (8.314 * 300.65)
    target_mgm3 = 0.5 * LFL * 1e6
    
    # Estimate duration based on hole size (fullbore = short, small = long)
    if d_hole_mm >= 100:
        dur = 10  # short release for large holes
    else:
        dur = 600  # continuous for small holes
    
    M_kg = mdot_kg_s * dur
    if M_kg < 0.01:
        return 0
    
    lo, hi = 1.0, 5000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        sy = sigma_y(mid, stability, 'rural')
        sz = sigma_z(mid, stability, 'rural')
        if sy <= 0 or sz <= 0:
            lo = mid
            continue
        if dur < 120:
            # Puff model
            C = M_kg * 1e6 / ((2*math.pi)**1.5 * sy * sy * sz)
        else:
            # Plume model (simplified)
            u = wind_ms
            C = mdot_kg_s * 1e9 / (math.pi * sy * sz * u)
        
        if C > target_mgm3:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break
    return (lo + hi) / 2


def calc_explosion_overpressure(mdot_kg_s, d_hole_mm, distance_m):
    """Calculate overpressure at distance using TNO multi-energy."""
    from rekarisk.models.explosion.tno_multi_energy import calculate_tno_multi_energy
    
    # Flash fire distance gives cloud size
    dHc = 47.5e6
    M_fuel = mdot_kg_s * 30  # 30s cloud build-up time
    
    try:
        result = calculate_tno_multi_energy(
            fuel_mass_kg=M_fuel,
            distance_m=distance_m,
            heat_of_combustion=dHc/1e6,  # MJ/kg
            strength=7,  # default strength for congested area
        )
        return result.get("overpressure_Pa", result.get("overpressure_bar", 0) * 1e5)
    except:
        # Fallback: TNT equivalent
        W_TNT = M_fuel * dHc / 4.6e6 * 0.04  # 4% efficiency
        if W_TNT <= 0 or distance_m <= 0:
            return 0
        Z = distance_m / (W_TNT ** (1/3))
        P_s = 101325 * (0.84 / Z + 1.07 / Z**2 + 2.09 / Z**3)  # Hopkinson-Cranz
        return max(P_s, 0)


def probit_thermal(dose):
    """Probit for thermal fatality (Eisenberg)."""
    # dose = t * q^(4/3) in (s * (W/m^2)^(4/3))
    if dose <= 0:
        return 0
    Y = -14.9 + 2.56 * math.log(dose)
    if Y < 0:
        return 0
    if Y > 8:
        return 1.0
    # Normal CDF approximation
    from math import erf, sqrt
    P = 0.5 * (1 + erf((Y - 5) / sqrt(2)))
    return P


def probit_overpressure(P_Pa):
    """Probit for overpressure fatality."""
    # Head impact probit
    if P_Pa <= 0:
        return 0
    P_bar = P_Pa / 1e5  # bar
    Y = 1.47 + 8.24 * math.log(P_bar + 0.001)
    if Y < 0:
        return 0
    if Y > 8:
        return 1.0
    from math import erf, sqrt
    P = 0.5 * (1 + erf((Y - 5) / sqrt(2)))
    return P


# ══════════════════════════════════════════════════════════════════════════════
# RUN QRA
# ══════════════════════════════════════════════════════════════════════════════

def run_qra():
    print("=" * 90)
    print("QRA COMPARISON: Rekarisk vs SAFETI")
    print("Reference: FNKT-20-P1-SR-007, NKT-01TW CPP Gundih")
    print("=" * 90)
    
    # Calculate LSIR at each location
    # LSIR = Σ over all scenarios: freq × P(fatality at location)
    
    lsir_rekarisk = {}
    
    for loc_name, (x0, y0) in LOCATIONS.items():
        total_risk = 0.0
        
        for iso_name, iso in ISO_SECTIONS.items():
            for hole_name, d_mm in HOLE_SIZES.items():
                freq = LEAK_FREQ[hole_name]
                
                # Release rate
                mdot = calc_release_rate(iso, d_mm)
                if mdot <= 0:
                    continue
                
                for wind_name, wind_ms, stab in WEATHER:
                    w_prob = WEATHER_PROB[wind_name]
                    
                    # Calculate distances to various thresholds
                    d_4_73 = calc_jet_fire_distance(mdot, d_mm, 4.73)
                    d_12_5 = calc_jet_fire_distance(mdot, d_mm, 12.5)
                    d_37_5 = calc_jet_fire_distance(mdot, d_mm, 37.5)
                    d_flash = calc_flash_fire_distance(mdot, d_mm, wind_ms, stab)
                    
                    # Distance from THIS release point to location
                    rx, ry = RELEASE_LOCATIONS.get(iso_name, (0, 0))
                    dist = math.sqrt((x0 - rx)**2 + (y0 - ry)**2)
                    if dist < 1.0:
                        dist = 1.0
                    
                    # Shelter factor for this location
                    sf = SHELTER_FACTOR.get(loc_name, 1.0)
                    
                    # --- Jet Fire fatality (probit-based) ---
                    p_jet = 0.0
                    if d_37_5 > dist:
                        p_jet = 0.90 * sf
                    elif d_12_5 > dist:
                        p_jet = 0.40 * sf
                    elif d_4_73 > dist:
                        p_jet = 0.05 * sf
                    
                    # --- Flash Fire fatality ---
                    p_flash = 0.90 * sf if d_flash > dist else 0.0
                    
                    # --- VCE fatality ---
                    p_vce = 0.0
                    if d_flash > dist:
                        op = calc_explosion_overpressure(mdot, d_mm, dist)
                        if op > 0:
                            p_vce = probit_overpressure(op) * sf
                    
                    # Ignition probabilities
                    p_imm = P_IMM_IGNITION.get(hole_name, 0.1)
                    p_del = P_DEL_IGNITION.get(hole_name, 0.2)
                    
                    # VCE fraction of delayed ignition
                    p_vce_frac = 0.30
                    
                    # Total fatality probability for this scenario
                    p_fatal = (p_imm * p_jet +
                              p_del * p_vce_frac * p_vce +
                              p_del * (1 - p_vce_frac) * p_flash)
                    
                    # Apply ESD effectiveness
                    esd = ESD_EFFECTIVENESS.get(hole_name, 1.0)
                    
                    # Add to LSIR
                    total_risk += freq * w_prob * p_fatal * esd
        
        lsir_rekarisk[loc_name] = total_risk
    
    # ── Print LSIR Comparison ──
    print("\n1. LSIR COMPARISON (per year)")
    print("-" * 70)
    print(f"{'Location':<30} {'SAFETI':>12} {'Rekarisk':>12} {'Ratio':>8}")
    print("-" * 70)
    
    for loc in SAFETI_LSIR:
        s = SAFETI_LSIR[loc]
        r = lsir_rekarisk.get(loc, 0)
        ratio = r / s if s > 0 else 0
        print(f"{loc:<30} {s:>12.2e} {r:>12.2e} {ratio:>8.2f}x")
    
    # ── Calculate IRPA ──
    print("\n2. IRPA COMPARISON - Process Hazard (per year)")
    print("-" * 70)
    print(f"{'Worker':<25} {'SAFETI':>12} {'Rekarisk':>12} {'Ratio':>8}")
    print("-" * 70)
    
    irpa_rekarisk = {}
    for worker, occupancy in WORKER_OCCUPANCY.items():
        irpa = 0
        for loc, frac in occupancy.items():
            irpa += lsir_rekarisk.get(loc, 0) * frac
        irpa_rekarisk[worker] = irpa
        
        s = SAFETI_IRPA_PROCESS.get(worker, 0)
        r = irpa
        ratio = r / s if s > 0 else 0
        match = "✓" if 0.5 < ratio < 2.0 else ("⚠" if 0.1 < ratio < 5.0 else "✗")
        print(f"{worker:<25} {s:>12.2e} {r:>12.2e} {ratio:>8.2f}x {match}")
    
    # ── Summary Statistics ──
    print("\n3. SUMMARY STATISTICS")
    print("-" * 50)
    
    lsir_ratios = []
    for loc in SAFETI_LSIR:
        s = SAFETI_LSIR[loc]
        r = lsir_rekarisk.get(loc, 0)
        if s > 0 and r > 0:
            lsir_ratios.append(r / s)
    
    if lsir_ratios:
        print(f"LSIR ratio range: {min(lsir_ratios):.2f}x - {max(lsir_ratios):.2f}x")
        print(f"LSIR ratio mean:  {np.mean(lsir_ratios):.2f}x")
        print(f"LSIR ratio median: {np.median(lsir_ratios):.2f}x")
    
    print(f"\nSAFETI PLL Total: {SAFETI_PLL_TOTAL:.2e}/year")
    
    # Count workers in ALARP
    alarp_count = sum(1 for r in lsir_rekarisk.values() if 1e-6 < r < 1e-3)
    print(f"Locations in ALARP (Rekarisk): {alarp_count}/{len(lsir_rekarisk)}")
    
    alarp_safeti = sum(1 for s in SAFETI_LSIR.values() if 1e-6 < s < 1e-3)
    print(f"Locations in ALARP (SAFETI):   {alarp_safeti}/{len(SAFETI_LSIR)}")
    
    return lsir_rekarisk, irpa_rekarisk


if __name__ == "__main__":
    run_qra()
