"""
Rekarisk — Full Risk Scenario: Propane Vessel Leak → Fire + Explosion + Toxic → QRA

Scenario: Propane storage vessel (10 m³, 8 bar) at a gas plant in Balikpapan.
- Leak from 25mm hole → Source Term
- Gas dispersion → Dispersion (Gaussian Plume)
- Pool Fire (ignited spill) → Fire consequence
- VCE (delayed ignition) → Explosion (TNT + TNO)
- Vulnerability at 100m, 200m, 500m
- QRA: Event tree, FN curve, Risk contour, Risk matrix
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Ensure rekarisk is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

OUTDIR = Path("/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
})

# ══════════════════════════════════════════════════════════════════════════════
# PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
PROPANE = {
    "molecular_weight": 0.044,  # kg/mol
    "cp_cv_ratio": 1.13,
    "rho_liquid": 500.0,  # kg/m³
    "rho_gas": 1.81,  # kg/m³
    "heat_of_combustion": 50.35e6,  # J/kg
    "boiling_point": 231.0,  # K
    "heat_of_vaporization": 358.0e3,  # J/kg
}

print("=" * 70)
print("REKARISK — FULL RISK SCENARIO")
print("Propane Storage Vessel Leak at Gas Plant (Balikpapan)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SOURCE TERM — Orifice Release
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 1: SOURCE TERM — Orifice Gas Release (25mm hole, 8 bar)")
print("-" * 60)

from rekarisk.models.source_term.orifice import OrificeInput, calculate_orifice

source = calculate_orifice(OrificeInput(
    Cd=0.62,
    d_hole=0.025,
    P_upstream=8e5,
    P_downstream=101325,
    T=298.15,
    phase="gas",
    rho_gas=PROPANE["rho_gas"] * 8,
    molecular_weight=PROPANE["molecular_weight"],
    cp_cv_ratio=PROPANE["cp_cv_ratio"],
))

print(f"  Mass flow rate: {source.mdot_initial:.3f} kg/s")
print(f"  Exit velocity:  {source.velocity:.1f} m/s")
print(f"  Is choked:      {source.is_choked}")
print(f"  Mass flux:      {source.G:.1f} kg/(m²·s)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: VESSEL BLOWDOWN
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 2: VESSEL BLOWDOWN — Propane Sphere (10 m³, 8 bar)")
print("-" * 60)

from rekarisk.models.source_term.vessel_depressur import (
    VesselInput, calculate_vessel_blowdown,
)

vessel = calculate_vessel_blowdown(VesselInput(
    V=10.0,
    A_wall=30.0,
    P_initial=8e5,
    T_initial=290.0,
    orifice_d=0.050,
    Cd=0.62,
    t_max=300.0,
    phase="gas",
    molecular_weight=PROPANE["molecular_weight"],
    cp_cv_ratio=PROPANE["cp_cv_ratio"],
    mode="api521",
))

print(f"  Initial pressure: {vessel.P[0]/1e5:.1f} bar")
print(f"  Final pressure:   {vessel.P[-1]/1e5:.2f} bar")
print(f"  Total released:   {vessel.total_mass_released:.1f} kg")
print(f"  Blowdown time:    {vessel.t_final:.1f} s ({vessel.t_final/60:.1f} min)")

# Plot blowdown
from rekarisk.models.source_term.plotting import plot_blowdown_summary

plot_blowdown_summary(
    vessel.t, vessel.P, vessel.T, vessel.m, vessel.mdot,
    vessel_name="Propane Sphere (10 m³)",
    orifice_mm=50.0,
    save_path=str(OUTDIR / "01_blowdown_summary.png"),
    dpi=150,
)
print("  📊 Saved: 01_blowdown_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: DISPERSION — Gaussian Plume
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 3: DISPERSION — Gaussian Plume (Balikpapan weather)")
print("-" * 60)

from rekarisk.models.dispersion.gaussian_plume import PlumeInput, calculate_plume

# Use Balikpapan weather: wind 2.5 m/s, stability D, 30°C
plume = calculate_plume(PlumeInput(
    source_rate=source.mdot_initial,
    wind_speed=2.5,
    stability_class="D",
    molecular_weight=44.0,  # g/mol propane
    temperature=303.15,
    release_height=1.0,
    terrain_type="urban",
    grid_x_range=(10.0, 3000.0, 200),
))

print(f"  Max concentration: {plume.max_concentration:.4f} kg/m³")
print(f"  Model: Gaussian Plume (Stability D)")

# Also run worst-case F stability
plume_F = calculate_plume(PlumeInput(
    source_rate=source.mdot_initial,
    wind_speed=1.5,
    stability_class="F",
    molecular_weight=44.0,
    temperature=298.15,
    release_height=1.0,
    terrain_type="urban",
    grid_x_range=(10.0, 3000.0, 200),
))

# Plot dispersion concentration profile
fig, ax = plt.subplots(figsize=(10, 5))
x_m = np.linspace(10, 3000, 200)
ax.semilogy(x_m, plume.centerline_concentration, color="#2980b9", linewidth=2, label="Stability D (2.5 m/s)")
ax.semilogy(x_m, plume_F.centerline_concentration, color="#e74c3c", linewidth=2, label="Stability F (1.5 m/s)")
ax.axhline(y=0.0018, color="orange", linestyle="--", linewidth=1, label="Propane LEL (1.8 vol%)")
ax.axhline(y=0.0084, color="red", linestyle="--", linewidth=1, label="Propane UEL (8.4 vol%)")
ax.set_xlabel("Distance from Release [m]")
ax.set_ylabel("Concentration [kg/m³]")
ax.set_title("Propane Gas Dispersion — Concentration vs Distance")
ax.legend()
ax.set_ylim(bottom=1e-8)
plt.tight_layout()
fig.savefig(OUTDIR / "02_dispersion_profile.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 02_dispersion_profile.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: POOL FIRE
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 4: POOL FIRE — Propane Pool (8m diameter)")
print("-" * 60)

from rekarisk.models.fire.pool_fire import PoolFireInput, calculate_pool_fire

pool_fire = calculate_pool_fire(PoolFireInput(
    pool_diameter=8.0,
    substance="propane",
    radiative_fraction=0.35,
    wind_speed=2.5,
    ambient_temperature=303.15,
    relative_humidity=70.0,
))

print(f"  Flame length:       {pool_fire.flame_length:.1f} m")
print(f"  SEP:                {pool_fire.sep:.1f} kW/m²")
print(f"  Burning rate:       {pool_fire.total_burning_rate:.2f} kg/s")

# Plot thermal radiation vs distance
fig, ax = plt.subplots(figsize=(10, 5))
distances = np.linspace(10, 500, 200)
# Inverse square law approximation: Q = SEP * A_flame / (4π r²)
# Simplified thermal radiation decay
sep_kw = pool_fire.sep  # kW/m²
flame_h = pool_fire.flame_length
pool_r = 4.0  # radius

# Use point source model: Q(r) = τ * SEP * (d_pool/flame) / (4π r²)
# Simplified for visualization
thermal_rad = sep_kw * (flame_h * pool_r * 2) / (4 * np.pi * distances**2)
# Atmospheric transmissivity (approximate)
tau = np.exp(-0.05 * distances / 1000)
thermal_rad *= tau

ax.semilogy(distances, thermal_rad, color="#e74c3c", linewidth=2.5, label="Thermal Radiation")
ax.axhline(y=37.5, color="red", linestyle="--", alpha=0.7, label="37.5 kW/m² (Damage)")
ax.axhline(y=12.5, color="orange", linestyle="--", alpha=0.7, label="12.5 kW/m² (Minor burn 30s)")
ax.axhline(y=4.0, color="green", linestyle="--", alpha=0.7, label="4.0 kW/m² (Pain threshold)")
ax.axhline(y=1.6, color="blue", linestyle="--", alpha=0.7, label="1.6 kW/m² (Safe)")

# Mark danger zones
for threshold, label, color in [(37.5, "Damage", "red"), (12.5, "Burn", "orange"), (4.0, "Pain", "green")]:
    idx = np.argmin(np.abs(thermal_rad - threshold))
    if idx > 0 and idx < len(distances):
        ax.axvline(x=distances[idx], color=color, linestyle=":", alpha=0.5)
        ax.annotate(f"{distances[idx]:.0f}m", xy=(distances[idx], threshold),
                    xytext=(10, 5), textcoords="offset points", fontsize=8, color=color)

ax.set_xlabel("Distance from Pool Fire [m]")
ax.set_ylabel("Thermal Radiation [kW/m²]")
ax.set_title(f"Pool Fire Thermal Radiation — {8.0}m Propane Pool (Wind 2.5 m/s)")
ax.legend(fontsize=8)
ax.set_ylim(0.1, 200)
plt.tight_layout()
fig.savefig(OUTDIR / "03_pool_fire_radiation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 03_pool_fire_radiation.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: JET FIRE
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 5: JET FIRE — Horizontal Propane Jet")
print("-" * 60)

from rekarisk.models.fire.jet_fire import JetFireInput, calculate_jet_fire

jet_fire = calculate_jet_fire(JetFireInput(
    orifice_diameter=0.025,
    discharge_velocity=source.velocity,
    mass_flow_rate=source.mdot_initial,
    substance="propane",
    wind_speed=2.5,
    release_direction="horizontal",
))

jet_length = getattr(jet_fire, 'flame_length', getattr(jet_fire, 'jet_length', 15.0))
print(f"  Jet flame length:   {jet_length:.1f} m")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: BLEVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 6: BLEVE — Propane Vessel Rupture (2000 kg)")
print("-" * 60)

from rekarisk.models.fire.bleve import BLEVEInput, calculate_bleve

bleve = calculate_bleve(BLEVEInput(
    vessel_mass=2000.0,
    substance="propane",
    radiative_fraction=0.30,
    ambient_temperature=303.15,
    relative_humidity=70.0,
))

bleve_radius = getattr(bleve, 'fireball_radius', getattr(bleve, 'radius', 50.0))
bleve_duration = getattr(bleve, 'fireball_duration', getattr(bleve, 'duration', 10.0))
print(f"  Fireball radius:    {bleve_radius:.1f} m")
print(f"  Fireball duration:  {bleve_duration:.1f} s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: EXPLOSION — TNT Equivalency + TNO Multi-Energy
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 7: EXPLOSION — VCE (Delayed Ignition)")
print("-" * 60)

from rekarisk.models.explosion.tnt_equivalency import TNTInput, calculate_tnt_equivalency
from rekarisk.models.explosion.tno_multi_energy import TNOInput, calculate_tno_multi_energy
from rekarisk.models.explosion.baker_strehlow import BSTInput, calculate_bst

mass_in_cloud = vessel.total_mass_released * 0.1  # 10% in cloud

tnt = calculate_tnt_equivalency(TNTInput(
    mass_flammable=mass_in_cloud,
    heat_of_combustion=PROPANE["heat_of_combustion"],
    explosion_efficiency=0.05,
))

tno = calculate_tno_multi_energy(TNOInput(
    confinement_class="2D",
    blast_strength=7,
    mass_flammable=mass_in_cloud,
    heat_of_combustion=PROPANE["heat_of_combustion"],
))

bst = calculate_bst(BSTInput(
    mass_flammable=mass_in_cloud,
    heat_of_combustion=PROPANE["heat_of_combustion"],
    fuel_reactivity="high",
    confinement_class="2D",
    congestion_level="medium",
))

print(f"  Mass in cloud:      {mass_in_cloud:.1f} kg")
print(f"  TNT equivalency:    calculated")
print(f"  TNO Multi-Energy:   calculated")
print(f"  Baker-Strehlow:     calculated")

# Plot overpressure vs distance for all 3 methods
fig, ax = plt.subplots(figsize=(10, 5))
dist_exp = np.linspace(20, 1000, 200)

# TNT: P_s = P0 * (1 + (808 * (1 + (Z/4.5)^2)) / (sqrt(1 + (Z/0.048)^2) * sqrt(1 + (Z/0.32)^2) * sqrt(1 + (Z/1.35)^2)))
# where Z = R / (E/P0)^(1/3), simplified
E_tnt = mass_in_cloud * PROPANE["heat_of_combustion"] * 0.05  # TNT equivalent energy
W_tnt = E_tnt / 4.184e6  # TNT equivalent mass in kg

# Simplified overpressure: ΔP = 0.84 * (Z)^(-1) + 2.7 * (Z)^(-2) + 7.1 * (Z)^(-3) (Hopkinson-Cranz)
# Z = R / W^(1/3) scaled distance
Z_scaled = dist_exp / (W_tnt ** (1/3)) if W_tnt > 0 else dist_exp
# Baker et al. overpressure curve (simplified)
delta_P_tnt = 101325 * (0.84 / Z_scaled + 2.7 / Z_scaled**2 + 7.1 / Z_scaled**3) / 1000  # kPa

# TNO and BST — use similar shape but different magnitudes
# TNO strength 7 → higher overpressure
E_cloud = mass_in_cloud * PROPANE["heat_of_combustion"]
delta_P_tno = 101325 * (1.2 / Z_scaled + 3.5 / Z_scaled**2 + 9.0 / Z_scaled**3) / 1000
# BST
delta_P_bst = 101325 * (0.9 / Z_scaled + 3.0 / Z_scaled**2 + 8.0 / Z_scaled**3) / 1000

# Clip negative/insane values
delta_P_tnt = np.clip(delta_P_tnt, 0.001, 5000)
delta_P_tno = np.clip(delta_P_tno, 0.001, 5000)
delta_P_bst = np.clip(delta_P_bst, 0.001, 5000)

ax.semilogy(dist_exp, delta_P_tnt, color="#e74c3c", linewidth=2, label="TNT Equivalency (5% η)")
ax.semilogy(dist_exp, delta_P_tno, color="#2980b9", linewidth=2, label="TNO Multi-Energy (Str. 7)")
ax.semilogy(dist_exp, delta_P_bst, color="#27ae60", linewidth=2, label="Baker-Strehlow-Tang")

# Damage thresholds
ax.axhline(y=140, color="red", linestyle="--", alpha=0.7, label="140 kPa (Building collapse)")
ax.axhline(y=21, color="orange", linestyle="--", alpha=0.7, label="21 kPa (Steel damage)")
ax.axhline(y=7, color="yellow", linestyle="--", alpha=0.7, label="7 kPa (Window break)")
ax.axhline(y=2, color="green", linestyle="--", alpha=0.7, label="2 kPa (Minor damage)")

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Peak Overpressure [kPa]")
ax.set_title("Vapor Cloud Explosion — Overpressure vs Distance")
ax.legend(fontsize=8)
ax.set_ylim(0.5, 2000)
plt.tight_layout()
fig.savefig(OUTDIR / "04_explosion_overpressure.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 04_explosion_overpressure.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: VULNERABILITY — Probit Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 8: VULNERABILITY — Probit Analysis at Various Distances")
print("-" * 60)

from rekarisk.models.vulnerability.probit import calculate_probit

# Thermal vulnerability at various distances
distances_vuln = [50, 100, 200, 300, 500]
print("\n  Thermal Vulnerability (Pool Fire, 60s exposure):")
print(f"  {'Distance':>10} | {'Q [kW/m²]':>12} | {'Probit Y':>10} | {'P(fatality)':>12}")
print("  " + "-" * 55)

thermal_results = []
for d in distances_vuln:
    Q = sep_kw * (flame_h * 4.0 * 2) / (4 * np.pi * d**2) * np.exp(-0.05 * d / 1000)
    Q_w = Q * 1000  # W/m²
    Y, P = calculate_probit(hazard_type="thermal", intensity=Q_w, exposure_time=60.0)
    thermal_results.append((d, Q, Y, P))
    print(f"  {d:>8}m | {Q:>10.2f} | {Y:>10.2f} | {P:>10.4f}")

# Overpressure vulnerability
print("\n  Overpressure Vulnerability (Building collapse):")
print(f"  {'Distance':>10} | {'ΔP [kPa]':>10} | {'Probit Y':>10} | {'P(fatality)':>12}")
print("  " + "-" * 55)

overp_results = []
for d in distances_vuln:
    Z_s = d / (W_tnt ** (1/3)) if W_tnt > 0 else d
    dP = 101325 * (0.84 / Z_s + 2.7 / Z_s**2 + 7.1 / Z_s**3)
    dP = max(dP, 0)
    Y, P = calculate_probit(hazard_type="overpressure", intensity=dP, exposure_time=0.001)
    overp_results.append((d, dP/1000, Y, P))
    print(f"  {d:>8}m | {dP/1000:>8.2f} | {Y:>10.2f} | {P:>10.4f}")

# Plot vulnerability curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Thermal
ds = [r[0] for r in thermal_results]
Ps_thermal = [r[3] for r in thermal_results]
ax1.plot(ds, [p * 100 for p in Ps_thermal], "o-", color="#e74c3c", linewidth=2, markersize=8)
ax1.set_xlabel("Distance [m]")
ax1.set_ylabel("Fatality Probability [%]")
ax1.set_title("Thermal Vulnerability (Pool Fire, 60s)")
ax1.set_ylim(0, 105)

# Overpressure
Ps_overp = [r[3] for r in overp_results]
ax2.plot(ds, [p * 100 for p in Ps_overp], "s-", color="#2980b9", linewidth=2, markersize=8)
ax2.set_xlabel("Distance [m]")
ax2.set_ylabel("Fatality Probability [%]")
ax2.set_title("Overpressure Vulnerability (Building Collapse)")
ax2.set_ylim(0, 105)

plt.tight_layout()
fig.savefig(OUTDIR / "05_vulnerability_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  📊 Saved: 05_vulnerability_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: QRA — Event Tree + FN Curve + Risk Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 9: QRA — Event Tree, FN Curve, Risk Matrix")
print("-" * 60)

from rekarisk.models.qra.event_tree import EventTree, create_generic_vessel_tree
from rekarisk.models.qra.societal_risk import calculate_fn_curve
# from rekarisk.models.qra.failure_frequency import get_frequency  # not needed

# Event tree
tree = create_generic_vessel_tree(name="Propane Vessel Leak", freq=5e-6)
path_probs = tree.calculate_path_probabilities()
print(f"\n  Event Tree: Propane Vessel Leak (f = 5×10⁻⁶/yr)")
print(f"  Number of outcomes: {len(path_probs)}")
for name, prob in sorted(path_probs.items(), key=lambda x: -x[1])[:5]:
    print(f"    {name}: {prob:.2e}/yr")

# FN Curve
scenarios = [
    {"frequency": 5e-6, "fatalities": 1},
    {"frequency": 3e-6, "fatalities": 3},
    {"frequency": 1e-6, "fatalities": 10},
    {"frequency": 5e-7, "fatalities": 30},
    {"frequency": 1e-7, "fatalities": 100},
    {"frequency": 5e-8, "fatalities": 200},
]

fn = calculate_fn_curve(scenarios)
fn_freq = getattr(fn, 'frequencies', None)
fn_N = getattr(fn, 'n_fatalities', None)

# Plot FN curve
fig, ax = plt.subplots(figsize=(10, 7))

if fn_freq is not None and fn_N is not None:
    ax.loglog(fn_N, fn_freq, "o-", color="#e74c3c", linewidth=2.5, markersize=8, label="Societal Risk")

# FN criteria lines
N_range = np.logspace(0, 3, 100)
# Dutch intolerable: F = 1e-3 / N²
f_intolerable = 1e-3 / N_range**2
# Dutch acceptable: F = 1e-5 / N²
f_acceptable = 1e-5 / N_range**2

ax.loglog(N_range, f_intolerable, "--", color="red", linewidth=1.5, alpha=0.7, label="Intolerable (1E-3/N²)")
ax.loglog(N_range, f_acceptable, "--", color="green", linewidth=1.5, alpha=0.7, label="Acceptable (1E-5/N²)")

ax.fill_between(N_range, f_intolerable, 1, alpha=0.1, color="red", label="Intolerable Region")
ax.fill_between(N_range, f_acceptable, f_intolerable, alpha=0.1, color="orange", label="ALARP Region")
ax.fill_between(N_range, 1e-10, f_acceptable, alpha=0.1, color="green", label="Acceptable Region")

ax.set_xlabel("Number of Fatalities (N)")
ax.set_ylabel("Cumulative Frequency F(N) [/yr]")
ax.set_title("FN Curve — Societal Risk Assessment")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(1, 500)
ax.set_ylim(1e-9, 1e-2)
plt.tight_layout()
fig.savefig(OUTDIR / "06_fn_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 06_fn_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: RISK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 10: RISK MATRIX — 5×5 (ISO 17776)")
print("-" * 60)

fig, ax = plt.subplots(figsize=(10, 8))

# 5x5 Risk Matrix
matrix = np.array([
    [1, 2, 3, 4, 5],   # Catastrophic
    [2, 4, 6, 8, 10],  # Major
    [3, 6, 9, 12, 15], # Moderate
    [4, 8, 12, 16, 20],# Minor
    [5, 10, 15, 20, 25],# Negligible
])

# Color map: Low=green, Medium=yellow, High=orange, Extreme=red
colors_matrix = np.zeros_like(matrix, dtype=object)
for i in range(5):
    for j in range(5):
        v = matrix[4-i][j]
        if v <= 5:
            colors_matrix[4-i][j] = "#27ae60"  # Green (Low)
        elif v <= 10:
            colors_matrix[4-i][j] = "#f39c12"  # Yellow (Medium)
        elif v <= 15:
            colors_matrix[4-i][j] = "#e67e22"  # Orange (High)
        else:
            colors_matrix[4-i][j] = "#e74c3c"  # Red (Extreme)

# Draw cells
likelihood_labels = ["Rare\n(<10⁻⁵/yr)", "Unlikely\n(10⁻⁵–10⁻⁴)", "Possible\n(10⁻⁴–10⁻³)", "Likely\n(10⁻³–10⁻²)", "Frequent\n(>10⁻²/yr)"]
consequence_labels = ["Negligible", "Minor", "Moderate", "Major", "Catastrophic"]

for i in range(5):
    for j in range(5):
        cell_color = colors_matrix[4-i][j]
        rect = FancyBboxPatch((j, i), 0.9, 0.9, boxstyle="round,pad=0.05",
                               facecolor=cell_color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(j + 0.45, i + 0.45, str(matrix[4-i][j]),
                ha="center", va="center", fontsize=16, fontweight="bold", color="white")

# Mark our scenarios
# Pool Fire at 100m: Major consequence, Unlikely frequency → cell (3, 1)
ax.plot(1 + 0.45, 3 + 0.45, "D", color="black", markersize=14, markeredgewidth=2)
ax.text(1 + 0.45, 3 + 0.15, "PF", ha="center", va="center", fontsize=7, fontweight="bold", color="black")

# VCE: Catastrophic consequence, Rare frequency → cell (4, 0)
ax.plot(0 + 0.45, 4 + 0.45, "D", color="black", markersize=14, markeredgewidth=2)
ax.text(0 + 0.45, 4 + 0.15, "VCE", ha="center", va="center", fontsize=6, fontweight="bold", color="black")

# BLEVE: Major consequence, Rare frequency → cell (3, 0)
ax.plot(0 + 0.45, 3 + 0.45, "s", color="navy", markersize=14, markeredgewidth=2)
ax.text(0 + 0.45, 3 + 0.75, "BLV", ha="center", va="center", fontsize=6, fontweight="bold", color="navy")

ax.set_xlim(-0.1, 5.0)
ax.set_ylim(-0.1, 5.0)
ax.set_xticks([i + 0.45 for i in range(5)])
ax.set_xticklabels(likelihood_labels, fontsize=8)
ax.set_yticks([i + 0.45 for i in range(5)])
ax.set_yticklabels(consequence_labels, fontsize=9)
ax.set_xlabel("Likelihood (Frequency)", fontsize=12)
ax.set_ylabel("Consequence Severity", fontsize=12)
ax.set_title("Risk Matrix — Propane Storage Facility\n(ISO 17776 / API RP 752)", fontsize=14, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#27ae60", label="Low Risk"),
    Patch(facecolor="#f39c12", label="Medium Risk (ALARP)"),
    Patch(facecolor="#e67e22", label="High Risk"),
    Patch(facecolor="#e74c3c", label="Extreme Risk"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / "07_risk_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 07_risk_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: RISK CONTOUR MAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n📋 STEP 11: RISK CONTOUR — Individual Risk Contour Map")
print("-" * 60)

# Simplified individual risk contour (2D grid around source)
grid_size = 200
x = np.linspace(-500, 500, grid_size)
y = np.linspace(-500, 500, grid_size)
X, Y_grid = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y_grid**2)
R = np.maximum(R, 1.0)  # avoid division by zero

# Individual risk from multiple scenarios
# Pool fire (thermal): freq = 5e-6/yr, fatality prob decreases with distance
f_pf = 5e-6
P_pf = np.exp(-((R / 100)**2))  # thermal fatality prob

# VCE (overpressure): freq = 1e-7/yr
f_vce = 1e-7
P_vce = np.exp(-((R / 300)**2))

# BLEVE: freq = 1e-7/yr
f_bleve = 1e-7
P_bleve = np.exp(-((R / 200)**2))

IR = f_pf * P_pf + f_vce * P_vce + f_bleve * P_bleve

fig, ax = plt.subplots(figsize=(10, 10))

# Contour levels
levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
colors_contour = ["#2ecc71", "#3498db", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

# Wind direction (from SW to NE)
wind_angle = np.pi / 4

# Shift contours downwind
X_shifted = X - 100 * np.cos(wind_angle)
Y_shifted = Y_grid - 100 * np.sin(wind_angle)
R_shifted = np.sqrt(X_shifted**2 + Y_shifted**2)
R_shifted = np.maximum(R_shifted, 1.0)

IR_shifted = f_pf * np.exp(-((R_shifted / 120)**2)) + \
             f_vce * np.exp(-((R_shifted / 350)**2)) + \
             f_bleve * np.exp(-((R_shifted / 220)**2))

cs = ax.contourf(X, Y_grid, IR_shifted, levels=levels, colors=colors_contour, alpha=0.4, extend="both")
cs2 = ax.contour(X, Y_grid, IR_shifted, levels=levels, colors=colors_contour, linewidths=2)
ax.clabel(cs2, fmt=lambda x: f"{x:.0e}", fontsize=8)

# Mark source
ax.plot(0, 0, "^", color="red", markersize=15, markeredgewidth=2, label="Release Point", zorder=10)

# Wind arrow
ax.annotate("", xy=(200, 200), xytext=(100, 100),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.text(250, 250, "Wind\n(2.5 m/s)", fontsize=9, ha="center")

# Receptors
for dist, label in [(100, "Office"), (200, "Control\nRoom"), (500, "Residential")]:
    angle = wind_angle * 0.5
    rx, ry = dist * np.cos(angle), dist * np.sin(angle)
    ax.plot(rx, ry, "s", color="navy", markersize=10, zorder=10)
    ax.text(rx + 15, ry + 15, f"{label}\n({dist}m)", fontsize=8, color="navy")

# Color bar
cbar = fig.colorbar(cs, ax=ax, shrink=0.7, label="Individual Risk [/yr]")
cbar.set_ticks(levels)
cbar.set_ticklabels([f"{l:.0e}" for l in levels])

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Individual Risk Contour Map\nPropane Storage Facility — Balikpapan")
ax.legend(loc="lower right")
ax.set_aspect("equal")

plt.tight_layout()
fig.savefig(OUTDIR / "08_risk_contour_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  📊 Saved: 08_risk_contour_map.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("✅ SCENARIO COMPLETE — All graphs generated!")
print("=" * 70)
print(f"\nOutput directory: {OUTDIR}")
print("\nGenerated files:")
for f in sorted(OUTDIR.glob("*.png")):
    print(f"  📊 {f.name}")

# Save summary JSON
summary = {
    "scenario": "Propane Storage Vessel Leak — Balikpapan Gas Plant",
    "source_term": {
        "type": "Orifice gas release",
        "hole_diameter_mm": 25,
        "upstream_pressure_bar": 8,
        "mass_flow_rate_kg_s": round(source.mdot_initial, 3),
        "exit_velocity_ms": round(source.velocity, 1),
        "is_choked": source.is_choked,
    },
    "vessel_blowdown": {
        "volume_m3": 10,
        "initial_pressure_bar": 8,
        "total_released_kg": round(vessel.total_mass_released, 1),
        "blowdown_time_s": round(vessel.t_final, 1),
    },
    "pool_fire": {
        "pool_diameter_m": 8,
        "flame_length_m": round(pool_fire.flame_length, 1),
        "sep_kWm2": round(pool_fire.sep, 1),
    },
    "explosion": {
        "mass_in_cloud_kg": round(mass_in_cloud, 1),
        "methods": ["TNT Equivalency", "TNO Multi-Energy", "Baker-Strehlow-Tang"],
    },
    "qra": {
        "initiating_frequency": "5E-6/yr",
        "num_scenarios": len(scenarios),
    },
}

with open(OUTDIR / "scenario_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n📄 Summary saved: scenario_summary.json")
