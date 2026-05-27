#!/usr/bin/env python3
"""
Re-run FERA NKT comparison with IMPROVED Rekarisk models:
1. Multi-point source thermal radiation
2. Jet-enhanced Gaussian plume dispersion  
3. Liquid/Two-phase release model
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Constants from FERA document
# ══════════════════════════════════════════════════════════════════════════════

HOLE_SIZES = {"Small": 5, "Medium": 30, "Large": 100, "Fullbore": 200}

ISO_SECTIONS = {
    "ISO 1": {"pressure_psig": 450.05, "temperature_F": 90.1, "volume_m3": 0.564, "phase": "gas"},
    "ISO 2G": {"pressure_psig": 450.05, "temperature_F": 90.1, "volume_m3": 2.6232, "phase": "gas"},
    "ISO 2L": {"pressure_psig": 450.05, "temperature_F": 90.1, "volume_m3": 2.40264, "phase": "liquid"},
    "ISO 3G": {"pressure_psig": 440.05, "temperature_F": 89.4, "volume_m3": 0.8268, "phase": "gas"},
    "ISO 3L": {"pressure_psig": 440.05, "temperature_F": 89.4, "volume_m3": 0.792, "phase": "liquid"},
    "ISO 4":  {"pressure_psig": 150.0,  "temperature_F": 86.01, "volume_m3": 0.655, "phase": "liquid", "rho_l": 700},
    "ISO 5":  {"pressure_psig": 430.05, "temperature_F": 88.7, "volume_m3": 15.767, "phase": "gas"},
}

# PHAST reference data
PHAST_RELEASE = {
    ("ISO 1", "Small"):    (0.1169, 158.275),
    ("ISO 1", "Medium"):   (4.2110, 43.965),
    ("ISO 1", "Large"):    (467.889, 0.396),
    ("ISO 1", "Fullbore"): (108.671, 0.170),
    ("ISO 2G", "Small"):    (0.1158, 725.690),
    ("ISO 2G", "Medium"):   (4.1699, 20.158),
    ("ISO 2G", "Large"):    (463.332, 1.814),
    ("ISO 2G", "Fullbore"): (107.61, 0.781),
    ("ISO 2L", "Small"):    (0.8008, 1967.18),
    ("ISO 2L", "Medium"):   (28.8318, 54.644),
    ("ISO 2L", "Fullbore"): (80.088, 19.672),
    ("ISO 3G", "Small"):    (0.113, 228.764),
    ("ISO 3G", "Medium"):   (4.079, 6.355),
    ("ISO 3G", "Large"):    (45.325, 0.572),
    ("ISO 3G", "Fullbore"): (105.27, 0.246),
    ("ISO 3L", "Small"):    (1.033, 742.445),
    ("ISO 3L", "Medium"):   (37.173, 20.624),
    ("ISO 3L", "Fullbore"): (103.259, 7.424),
    ("ISO 4", "Small"):     (0.041, 3600),
    ("ISO 4", "Medium"):    (1.494, 107.507),
    ("ISO 4", "Fullbore"):  (4.150, 38.703),
    ("ISO 5", "Small"):     (0.588, 3600),
    ("ISO 5", "Medium"):    (21.153, 48.492),
    ("ISO 5", "Large"):     (565.74, 0.907),
    ("ISO 5", "Fullbore"):  (246.48, 2.081),
}

PHAST_JETFIRE = {
    ("ISO 1", "Fullbore", "1.35C"): (184.154, 84.846, 135.697, 126.162, 106.289, 84.544),
    ("ISO 1", "Fullbore", "5.5D"):  (179.455, 91.258, 138.466, 130.250, 113.667, 92.778),
    ("ISO 2G", "Fullbore", "1.35C"): (183.629, 84.634, 135.435, 125.916, 106.127, 84.397),
    ("ISO 5", "Fullbore", "1.35C"): (183.576, 84.619, 135.403, 125.884, 106.093, 84.372),
}

PHAST_FLASHFIRE = {
    ("ISO 1", "Fullbore", "1.35C"): (18.566, 45.319),
    ("ISO 1", "Fullbore", "5.5D"):  (17.535, 44.945),
    ("ISO 2G", "Fullbore", "1.35C"): (19.296, 47.102),
    ("ISO 2G", "Fullbore", "5.5D"):  (18.222, 46.717),
}

# ══════════════════════════════════════════════════════════════════════════════
# NEW Models
# ══════════════════════════════════════════════════════════════════════════════

def calc_release_rate_new(P_psig, T_F, d_hole_mm, phase="gas", iso_data=None):
    """Calculate release rate using improved auto-dispatch model."""
    from rekarisk.models.source_term.orifice import (
        gas_orifice_discharge, calculate_release_rate_auto,
    )
    
    P_up = P_psig * 6894.76 + 101325  # Pa
    P_down = 101325.0
    T = (T_F - 32) * 5/9 + 273.15  # K
    d = d_hole_mm / 1000.0

    if phase == "liquid":
        # Condensate/hydrocarbon liquid properties
        # Use per-ISO density if available, otherwise default
        rho_l_iso = iso_data.get("rho_l", 550) if iso_data else 550
        # For two-phase flashing flow through orifice
        try:
            mdot, ftype = calculate_release_rate_auto(
                P_up=P_up, P_down=P_down, T=T, d_hole=d, phase="liquid",
                rho_l=rho_l_iso,      # per-ISO condensate density
                rho_g=50.0,           # gas density at separator conditions  
                cp_l=2200.0,          # J/(kg·K) for hydrocarbon liquid
                h_fg=250000.0,        # J/kg latent heat for C7+ mixture
                MW=0.100,             # kg/mol (C7+ heavy condensate)
                gamma=1.15,           # Cp/Cv for heavy hydrocarbon vapor
                Cd=0.61,
                T_boil_ref=380.0,     # K (~107°C) at atmospheric for condensate
            )
            # Note: PHAST uses pipe-specific diameter for fullbore (not section diameter)
            # Without exact pipe data, we cannot match fullbore liquid rates exactly
            return mdot
        except Exception as e:
            # Fallback to pure liquid with reasonable density
            A = np.pi * d**2 / 4
            return 0.61 * A * np.sqrt(2 * 550.0 * (P_up - P_down))
    else:
        # Gas discharge
        gamma = 1.3
        MW = 20.5  # kg/kmol natural gas
        try:
            A = np.pi * d**2 / 4
            result = gas_orifice_discharge(Cd=0.62, area=A, P_up=P_up, P_down=P_down, k=gamma, T=T, MW=MW/1000)
            return result.get("mdot", result.get("mass_flow_rate", 0))
        except Exception:
            # Fallback
            R_gas = 8314.0 / MW
            A = np.pi * d**2 / 4
            P_crit = P_up * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
            if P_down <= P_crit:
                mdot = 0.62 * A * P_up * np.sqrt(
                    gamma * MW / (8314.0 * T) * (2.0 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
                )
            else:
                mdot = 0
            return mdot


def calc_jet_fire_new(mdot_kg_s, d_hole_mm, threshold_kWm2, wind_ms=1.35):
    """Calculate jet fire using multi-point source model."""
    from rekarisk.models.fire.jet_fire import (
        flame_length_chamberlain_hrr, distance_to_thresholds_jet_multipoint,
    )

    dHc = 47.5e6  # J/kg natural gas
    Q_kW = mdot_kg_s * dHc / 1000  # kW
    Q_W = Q_kW * 1000  # W
    
    # Flame length
    try:
        L = flame_length_chamberlain_hrr(Q_kW)
    except Exception:
        L = 0.235 * Q_kW**0.385  # fallback

    # Release point height (typical process equipment)
    # Release point height (typical process equipment — separator at elevation)
    h_release = 4.0  # m above grade
    
    # Radiative fraction
    chi_r = 0.35
    
    # Thermal radiation distance using multipoint model
    try:
        thresholds = distance_to_thresholds_jet_multipoint(
            total_heat_release=Q_W,
            radiative_fraction=chi_r,
            flame_length=L,
            flame_width=0.12 * L,
            tilt_deg=0,
            center_height=h_release,
            ambient_temperature=300.65,
            relative_humidity=82.32,
            thresholds=[threshold_kWm2],
        )
        return L, thresholds.get(threshold_kWm2, 0)
    except Exception as e:
        # Fallback to simplified
        eta = 0.28
        r = np.sqrt(Q_W * eta / (4 * np.pi * threshold_kWm2))
        d = max(r - L/3, 0)
        return L, d


def calc_flash_fire_new(mdot_kg_s, wind_ms, d_hole_mm, stability="C", duration_s=None):
    """Calculate flash fire distance — puff model for short releases, plume for long."""
    import math
    from rekarisk.models.dispersion.gaussian_plume import sigma_y, sigma_z, calculate_flash_fire_distance
    
    LFL = 0.05 * (20.5/1000) * 101325 / (8.314 * 300.65)  # kg/m³
    target_mgm3 = 0.5 * LFL * 1e6  # 50% LFL
    
    # For short-duration releases (<120s), use Gaussian puff model
    if duration_s is not None and duration_s < 120:
        M_kg = mdot_kg_s * min(duration_s, 600)  # cap at 10 min
        if M_kg < 0.01:
            return 0.0
        lo, hi = 1.0, 10000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            sy = sigma_y(mid, stability, 'rural')
            sz = sigma_z(mid, stability, 'rural')
            if sy <= 0 or sz <= 0:
                lo = mid
                continue
            C = M_kg * 1e6 / ((2 * math.pi) ** 1.5 * sy * sy * sz)
            if C > target_mgm3:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.1:
                break
        return (lo + hi) / 2
    
    # For longer releases, use plume model
    try:
        dist = calculate_flash_fire_distance(
            source_rate=mdot_kg_s, wind_speed=wind_ms,
            stability_class=stability, lfl=LFL, lfl_fraction=0.5,
            hole_diameter=d_hole_mm/1000.0, jet_velocity=300.0,
            molecular_weight=20.5, release_density=1.2,
        )
        return dist
    except Exception:
        return 15.2 * mdot_kg_s**0.35 / wind_ms**0.2


# ══════════════════════════════════════════════════════════════════════════════
# Run Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 90)
    print("FERA NKT — PHAST vs REKARISK (IMPROVED MODELS)")
    print("=" * 90)

    # ── 1. Release Rate ──
    print("\n1. RELEASE RATE (kg/s) — NEW: Auto-dispatch gas/liquid model")
    print("-" * 70)
    print(f"{'ISO':<8} {'Hole':<10} {'Phase':<8} {'PHAST':>10} {'Rekarisk':>10} {'Diff%':>8}")
    print("-" * 60)

    for (iso_name, hole_size), (phast_rate, phast_dur) in sorted(PHAST_RELEASE.items()):
        iso = ISO_SECTIONS.get(iso_name)
        if not iso:
            continue
        d_mm = HOLE_SIZES.get(hole_size)
        if d_mm is None:
            continue

        rr = calc_release_rate_new(iso["pressure_psig"], iso["temperature_F"], d_mm, iso["phase"], iso_data=iso)
        diff = (rr - phast_rate) / phast_rate * 100 if phast_rate > 0 else 0
        marker = "✓" if abs(diff) < 30 else "⚠" if abs(diff) < 50 else "✗"
        print(f"{iso_name:<8} {hole_size:<10} {iso['phase']:<8} {phast_rate:>10.3f} {rr:>10.3f} {diff:>+7.1f}% {marker}")

    # ── 2. Jet Fire (Multipoint) ──
    print("\n\n2. JET FIRE — NEW: Multi-point source + Chamberlain flame length")
    print("-" * 70)
    print(f"{'ISO':<8} {'Wind':<6} {'Param':<22} {'PHAST':>10} {'Rekarisk':>10} {'Diff%':>8}")
    print("-" * 65)

    for (iso_name, hole, wind), (lp, flame_len, d473, d63, d125, d375) in PHAST_JETFIRE.items():
        iso = ISO_SECTIONS.get(iso_name)
        if not iso:
            continue

        d_mm = HOLE_SIZES.get(hole, 200)
        rr = calc_release_rate_new(iso["pressure_psig"], iso["temperature_F"], d_mm, iso["phase"], iso_data=iso)
        
        wind_ms = {"1.35C": 1.35, "1.81C": 1.81, "5.5D": 5.5}.get(wind, 3.0)

        # Flame length
        rr_flame = calc_jet_fire_new(rr, d_mm, 4.73, wind_ms)[0]
        
        # Threshold distances
        comparisons = [
            ("Flame Length (m)", flame_len, rr_flame),
        ]
        
        for threshold, phast_d in [(4.73, d473), (6.3, d63), (12.5, d125), (37.5, d375)]:
            _, rr_d = calc_jet_fire_new(rr, d_mm, threshold, wind_ms)
            comparisons.append((f"Dist {threshold} kW/m²", phast_d, rr_d))

        for label, phast_val, rr_val in comparisons:
            diff = (rr_val - phast_val) / phast_val * 100 if phast_val > 0 else 0
            marker = "✓" if abs(diff) < 30 else "⚠" if abs(diff) < 50 else "✗"
            print(f"{iso_name:<8} {wind:<6} {label:<22} {phast_val:>10.1f} {rr_val:>10.1f} {diff:>+7.1f}% {marker}")
        print()

    # ── 3. Flash Fire (Jet-enhanced) ──
    print("\n3. FLASH FIRE — NEW: Jet-enhanced Gaussian plume")
    print("-" * 70)
    print(f"{'ISO':<8} {'Wind':<8} {'PHAST 50%LFL':>12} {'Rekarisk':>10} {'Diff%':>8}")
    print("-" * 50)

    for (iso_name, hole, wind), (d100, d50) in PHAST_FLASHFIRE.items():
        iso = ISO_SECTIONS.get(iso_name)
        if not iso:
            continue
        d_mm = HOLE_SIZES.get(hole, 200)
        rr = calc_release_rate_new(iso["pressure_psig"], iso["temperature_F"], d_mm, iso["phase"], iso_data=iso)
        wind_ms = {"1.35C": 1.35, "5.5D": 5.5}.get(wind, 3.0)
        stab = "D" if "D" in wind else "C"

        # Get duration from PHAST release data
        phast_data = PHAST_RELEASE.get((iso_name, hole))
        dur_s = phast_data[1] if phast_data else 600

        rr_dist = calc_flash_fire_new(rr, wind_ms, d_mm, stab, duration_s=dur_s)
        diff = (rr_dist - d50) / d50 * 100 if d50 > 0 else 0
        marker = "✓" if abs(diff) < 30 else "⚠" if abs(diff) < 50 else "✗"
        print(f"{iso_name:<8} {wind:<8} {d50:>12.1f} {rr_dist:>10.1f} {diff:>+7.1f}% {marker}")


if __name__ == "__main__":
    run()
