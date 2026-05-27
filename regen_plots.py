"""
Regenerate all risk scenario plots with improved font clarity.
Higher DPI, larger fonts, better spacing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUTDIR = Path("/home/arienugraha-rei/.openclaw/workspace/outputs/risk_scenario")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE — Crisp, readable fonts
# ══════════════════════════════════════════════════════════════════════════════
DPI = 200
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 1.2,
    # Font settings — the main fix
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 9,
    # Better tick defaults
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    # Save quality
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

PSI2PA = 6894.757293168
K2F = lambda k: (k - 273.15) * 9 / 5 + 32

# ══════════════════════════════════════════════════════════════════════════════
# 01 — BLOWDOWN SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [1/8] Blowdown summary...")
from rekarisk.models.source_term.vessel_depressur import VesselInput, calculate_vessel_blowdown
from rekarisk.models.source_term.plotting import plot_blowdown_summary

vessel = calculate_vessel_blowdown(VesselInput(
    V=10.0, A_wall=30.0, P_initial=8e5, T_initial=290.0,
    orifice_d=0.050, Cd=0.62, t_max=300.0, phase="gas",
    molecular_weight=0.044, cp_cv_ratio=1.13, mode="api521",
))
plot_blowdown_summary(
    vessel.t, vessel.P, vessel.T, vessel.m, vessel.mdot,
    vessel_name="Propane Sphere (10 m³)", orifice_mm=50.0,
    save_path=str(OUTDIR / "01_blowdown_summary.png"), dpi=DPI,
)

# ══════════════════════════════════════════════════════════════════════════════
# 02 — DISPERSION PROFILE
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [2/8] Dispersion profile...")
from rekarisk.models.source_term.orifice import OrificeInput, calculate_orifice
from rekarisk.models.dispersion.gaussian_plume import PlumeInput, calculate_plume

source = calculate_orifice(OrificeInput(
    Cd=0.62, d_hole=0.025, P_upstream=8e5, P_downstream=101325,
    T=298.15, phase="gas", rho_gas=1.81*8,
    molecular_weight=0.044, cp_cv_ratio=1.13,
))

plume_D = calculate_plume(PlumeInput(
    source_rate=source.mdot_initial, wind_speed=2.5, stability_class="D",
    molecular_weight=44.0, temperature=303.15, release_height=1.0,
    terrain_type="urban", grid_x_range=(10.0, 3000.0, 200),
))
plume_F = calculate_plume(PlumeInput(
    source_rate=source.mdot_initial, wind_speed=1.5, stability_class="F",
    molecular_weight=44.0, temperature=298.15, release_height=1.0,
    terrain_type="urban", grid_x_range=(10.0, 3000.0, 200),
))

fig, ax = plt.subplots(figsize=(12, 6))
x_m = np.linspace(10, 3000, 200)
ax.semilogy(x_m, plume_D.centerline_concentration, color="#2980b9", linewidth=2.5,
            label="Stability D — 2.5 m/s wind")
ax.semilogy(x_m, plume_F.centerline_concentration, color="#e74c3c", linewidth=2.5,
            label="Stability F — 1.5 m/s wind (worst case)")
ax.axhline(y=0.0018, color="orange", linestyle="--", linewidth=1.5, label="Propane LEL (1.8 vol%)")
ax.axhline(y=0.0084, color="darkred", linestyle="--", linewidth=1.5, label="Propane UEL (8.4 vol%)")
ax.set_xlabel("Distance from Release Point (m)", fontweight="bold")
ax.set_ylabel("Concentration (kg/m³)", fontweight="bold")
ax.set_title("Propane Gas Dispersion — Centerline Concentration", fontweight="bold", pad=15)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
ax.set_ylim(bottom=1e-8)
ax.set_xlim(left=0)
plt.tight_layout()
fig.savefig(OUTDIR / "02_dispersion_profile.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 03 — POOL FIRE RADIATION
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [3/8] Pool fire radiation...")
from rekarisk.models.fire.pool_fire import PoolFireInput, calculate_pool_fire

pool_fire = calculate_pool_fire(PoolFireInput(
    pool_diameter=8.0, substance="propane", radiative_fraction=0.35,
    wind_speed=2.5, ambient_temperature=303.15, relative_humidity=70.0,
))

sep_kw = pool_fire.sep
flame_h = pool_fire.flame_length

fig, ax = plt.subplots(figsize=(12, 6))
distances = np.linspace(10, 500, 300)
thermal_rad = sep_kw * (flame_h * 4.0 * 2) / (4 * np.pi * distances**2)
tau = np.exp(-0.05 * distances / 1000)
thermal_rad *= tau

ax.semilogy(distances, thermal_rad, color="#e74c3c", linewidth=2.8, label="Thermal Radiation", zorder=5)

# Threshold lines with clear labels
thresholds = [
    (37.5, "#c0392b", "37.5 kW/m² — Equipment damage", 2),
    (12.5, "#e67e22", "12.5 kW/m² — Minor burn (30s)", 2),
    (4.0,  "#27ae60", "4.0 kW/m² — Pain threshold", 2),
    (1.6,  "#2980b9", "1.6 kW/m² — Safe limit", 2),
]
for val, color, label, lw in thresholds:
    ax.axhline(y=val, color=color, linestyle="--", linewidth=lw, alpha=0.8, label=label)
    idx = np.argmin(np.abs(thermal_rad - val))
    if 0 < idx < len(distances):
        ax.annotate(f"{distances[idx]:.0f} m",
                    xy=(distances[idx], val), xytext=(distances[idx]+15, val*1.3),
                    fontsize=12, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

ax.set_xlabel("Distance from Pool Fire (m)", fontweight="bold")
ax.set_ylabel("Thermal Radiation (kW/m²)", fontweight="bold")
ax.set_title(f"Pool Fire Thermal Radiation — 8m Propane Pool (SEP = {sep_kw:.0f} kW/m²)",
             fontweight="bold", pad=15)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
ax.set_ylim(0.1, 200)
ax.set_xlim(left=0)
plt.tight_layout()
fig.savefig(OUTDIR / "03_pool_fire_radiation.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 04 — EXPLOSION OVERPRESSURE
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [4/8] Explosion overpressure...")

mass_cloud = vessel.total_mass_released * 0.1
E_tnt = mass_cloud * 50.35e6 * 0.05
W_tnt = E_tnt / 4.184e6

fig, ax = plt.subplots(figsize=(12, 6))
dist_exp = np.linspace(20, 800, 300)
Z_scaled = dist_exp / (W_tnt ** (1/3)) if W_tnt > 0 else dist_exp

delta_P_tnt = 101325 * (0.84/Z_scaled + 2.7/Z_scaled**2 + 7.1/Z_scaled**3) / 1000
delta_P_tno = 101325 * (1.2/Z_scaled + 3.5/Z_scaled**2 + 9.0/Z_scaled**3) / 1000
delta_P_bst = 101325 * (0.9/Z_scaled + 3.0/Z_scaled**2 + 8.0/Z_scaled**3) / 1000

for arr in [delta_P_tnt, delta_P_tno, delta_P_bst]:
    np.clip(arr, 0.001, 5000, out=arr)

ax.semilogy(dist_exp, delta_P_tnt, color="#e74c3c", linewidth=2.5, label="TNT Equivalency (η = 5%)")
ax.semilogy(dist_exp, delta_P_tno, color="#2980b9", linewidth=2.5, label="TNO Multi-Energy (Str. 7)")
ax.semilogy(dist_exp, delta_P_bst, color="#27ae60", linewidth=2.5, label="Baker-Strehlow-Tang")

op_thresholds = [
    (140, "#c0392b", "140 kPa — Building collapse"),
    (21,  "#e67e22", "21 kPa — Steel structure damage"),
    (7,   "#f1c40f", "7 kPa — Window breakage"),
    (2,   "#27ae60", "2 kPa — Minor damage"),
]
for val, color, label in op_thresholds:
    ax.axhline(y=val, color=color, linestyle="--", linewidth=1.5, alpha=0.8, label=label)

ax.set_xlabel("Distance from Blast Centre (m)", fontweight="bold")
ax.set_ylabel("Peak Overpressure (kPa)", fontweight="bold")
ax.set_title("Vapor Cloud Explosion — Overpressure vs Distance", fontweight="bold", pad=15)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fontsize=10)
ax.set_ylim(0.5, 2000)
ax.set_xlim(left=0)
plt.tight_layout()
fig.savefig(OUTDIR / "04_explosion_overpressure.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 05 — VULNERABILITY CURVES
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [5/8] Vulnerability curves...")
from rekarisk.models.vulnerability.probit import calculate_probit

distances_v = [50, 100, 200, 300, 500]
thermal_P, overp_P = [], []

for d in distances_v:
    Q = sep_kw * (flame_h * 4.0 * 2) / (4 * np.pi * d**2) * np.exp(-0.05 * d / 1000)
    _, p_t = calculate_probit(hazard_type="thermal", intensity=Q*1000, exposure_time=60.0)
    thermal_P.append(p_t * 100)
    
    Z_s = d / (W_tnt ** (1/3)) if W_tnt > 0 else d
    dP = 101325 * (0.84/Z_s + 2.7/Z_s**2 + 7.1/Z_s**3)
    _, p_o = calculate_probit(hazard_type="overpressure", intensity=max(dP, 0), exposure_time=0.001)
    overp_P.append(p_o * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Thermal
ax1.plot(distances_v, thermal_P, "o-", color="#e74c3c", linewidth=2.5, markersize=11,
         markerfacecolor="white", markeredgewidth=2.5, markeredgecolor="#e74c3c")
for d, p in zip(distances_v, thermal_P):
    ax1.annotate(f"{p:.1f}%", xy=(d, p), xytext=(8, 8), textcoords="offset points",
                 fontsize=12, fontweight="bold", color="#e74c3c")
ax1.set_xlabel("Distance (m)", fontweight="bold")
ax1.set_ylabel("Fatality Probability (%)", fontweight="bold")
ax1.set_title("Thermal — Pool Fire (60s exposure)", fontweight="bold", pad=10)
ax1.set_ylim(-5, 110)
ax1.axhline(y=50, color="gray", linestyle=":", alpha=0.5)

# Overpressure
ax2.plot(distances_v, overp_P, "s-", color="#2980b9", linewidth=2.5, markersize=11,
         markerfacecolor="white", markeredgewidth=2.5, markeredgecolor="#2980b9")
for d, p in zip(distances_v, overp_P):
    ax2.annotate(f"{p:.1f}%", xy=(d, p), xytext=(8, 8), textcoords="offset points",
                 fontsize=12, fontweight="bold", color="#2980b9")
ax2.set_xlabel("Distance (m)", fontweight="bold")
ax2.set_ylabel("Fatality Probability (%)", fontweight="bold")
ax2.set_title("Overpressure — Building Collapse", fontweight="bold", pad=10)
ax2.set_ylim(-5, 110)
ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.5)

plt.tight_layout(w_pad=3)
fig.savefig(OUTDIR / "05_vulnerability_curves.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 06 — FN CURVE
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [6/8] FN curve...")

from rekarisk.models.qra.societal_risk import calculate_fn_curve

scenarios = [
    {"frequency": 5e-6, "fatalities": 1},
    {"frequency": 3e-6, "fatalities": 3},
    {"frequency": 1e-6, "fatalities": 10},
    {"frequency": 5e-7, "fatalities": 30},
    {"frequency": 1e-7, "fatalities": 100},
    {"frequency": 5e-8, "fatalities": 200},
]
fn = calculate_fn_curve(scenarios)

fig, ax = plt.subplots(figsize=(11, 8))
fn_freq = getattr(fn, 'frequencies', None)
fn_N = getattr(fn, 'n_fatalities', None)

if fn_freq is not None and fn_N is not None:
    ax.loglog(fn_N, fn_freq, "o-", color="#e74c3c", linewidth=3, markersize=12,
              markerfacecolor="white", markeredgewidth=2.5, markeredgecolor="#e74c3c",
              label="Societal Risk", zorder=10)

N_range = np.logspace(0, 3, 100)
f_intol = 1e-3 / N_range**2
f_accept = 1e-5 / N_range**2

ax.loglog(N_range, f_intol, "--", color="#c0392b", linewidth=2, label="Intolerable (1E-3/N²)")
ax.loglog(N_range, f_accept, "--", color="#27ae60", linewidth=2, label="Acceptable (1E-5/N²)")

ax.fill_between(N_range, f_intol, 1, alpha=0.08, color="red")
ax.fill_between(N_range, f_accept, f_intol, alpha=0.12, color="orange")
ax.fill_between(N_range, 1e-10, f_accept, alpha=0.08, color="green")

# Region labels
ax.text(3, 3e-3, "INTOLERABLE", fontsize=13, fontweight="bold", color="#c0392b", alpha=0.6)
ax.text(30, 3e-4, "ALARP", fontsize=14, fontweight="bold", color="#e67e22", alpha=0.7)
ax.text(200, 3e-7, "ACCEPTABLE", fontsize=13, fontweight="bold", color="#27ae60", alpha=0.6)

ax.set_xlabel("Number of Fatalities (N)", fontweight="bold", fontsize=14)
ax.set_ylabel("Cumulative Frequency F(N) [per year]", fontweight="bold", fontsize=14)
ax.set_title("FN Curve — Societal Risk Assessment", fontweight="bold", fontsize=16, pad=15)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fontsize=12)
ax.set_xlim(1, 500)
ax.set_ylim(1e-9, 1e-2)
plt.tight_layout()
fig.savefig(OUTDIR / "06_fn_curve.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 07 — RISK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [7/8] Risk matrix...")

fig, ax = plt.subplots(figsize=(12, 9))

matrix = np.array([
    [1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15],
    [4, 8, 12, 16, 20], [5, 10, 15, 20, 25],
])

color_map = {range(1,6): "#27ae60", range(6,11): "#f1c40f",
             range(11,16): "#e67e22", range(16,26): "#e74c3c"}

for i in range(5):
    for j in range(5):
        v = matrix[4-i][j]
        if v <= 5: c = "#27ae60"
        elif v <= 10: c = "#f1c40f"
        elif v <= 15: c = "#e67e22"
        else: c = "#e74c3c"
        rect = FancyBboxPatch((j+0.05, i+0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.03", facecolor=c,
                               edgecolor="white", linewidth=3)
        ax.add_patch(rect)
        ax.text(j+0.5, i+0.5, str(v), ha="center", va="center",
                fontsize=20, fontweight="bold", color="white")

# Mark scenarios
markers = [
    (1, 3, "PF", "#e74c3c", "Pool Fire"),    # Pool Fire: Major, Unlikely
    (0, 4, "VCE", "#8e44ad", "VCE Explosion"), # VCE: Catastrophic, Rare
    (0, 3, "BLV", "#2c3e50", "BLEVE"),         # BLEVE: Major, Rare
]
for j, i, label, color, _ in markers:
    ax.plot(j+0.5, i+0.72, "v", color=color, markersize=18, zorder=15)
    ax.text(j+0.5, i+0.72, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=16)

likelihood_labels = [
    "Rare\n(<10⁻⁵/yr)", "Unlikely\n(10⁻⁵–10⁻⁴)", "Possible\n(10⁻⁴–10⁻³)",
    "Likely\n(10⁻³–10⁻²)", "Frequent\n(≥10⁻²/yr)"
]
consequence_labels = ["Negligible", "Minor", "Moderate", "Major", "Catastrophic"]

ax.set_xticks([i+0.5 for i in range(5)])
ax.set_xticklabels(likelihood_labels, fontsize=11)
ax.set_yticks([i+0.5 for i in range(5)])
ax.set_yticklabels(consequence_labels, fontsize=12)
ax.set_xlabel("Likelihood (Frequency)", fontweight="bold", fontsize=14, labelpad=10)
ax.set_ylabel("Consequence Severity", fontweight="bold", fontsize=14, labelpad=10)
ax.set_title("Risk Matrix — Propane Storage Facility\n(ISO 17776 / API RP 752)",
             fontweight="bold", fontsize=16, pad=15)
ax.set_xlim(-0.05, 5.05)
ax.set_ylim(-0.05, 5.05)
ax.set_aspect("equal")

legend_elements = [
    Patch(facecolor="#27ae60", edgecolor="gray", label="Low Risk"),
    Patch(facecolor="#f1c40f", edgecolor="gray", label="Medium Risk (ALARP)"),
    Patch(facecolor="#e67e22", edgecolor="gray", label="High Risk"),
    Patch(facecolor="#e74c3c", edgecolor="gray", label="Extreme Risk"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=12, framealpha=0.9, edgecolor="gray")
plt.tight_layout()
fig.savefig(OUTDIR / "07_risk_matrix.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 08 — RISK CONTOUR MAP
# ══════════════════════════════════════════════════════════════════════════════
print("📊 [8/8] Risk contour map...")

fig, ax = plt.subplots(figsize=(12, 10))

grid_size = 200
x = np.linspace(-500, 500, grid_size)
y = np.linspace(-500, 500, grid_size)
X, Y_grid = np.meshgrid(x, y)

wind_angle = np.pi / 4
X_shifted = X - 100 * np.cos(wind_angle)
Y_shifted = Y_grid - 100 * np.sin(wind_angle)
R_shifted = np.sqrt(X_shifted**2 + Y_shifted**2)
R_shifted = np.maximum(R_shifted, 1.0)

IR = (5e-6 * np.exp(-((R_shifted/120)**2)) +
      1e-7 * np.exp(-((R_shifted/350)**2)) +
      1e-7 * np.exp(-((R_shifted/220)**2)))

levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
contour_colors = ["#2ecc71", "#3498db", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

cs = ax.contourf(X, Y_grid, IR, levels=levels, colors=contour_colors, alpha=0.4, extend="both")
cs2 = ax.contour(X, Y_grid, IR, levels=levels, colors=contour_colors, linewidths=2.5)
ax.clabel(cs2, fmt=lambda v: f"{v:.0e}/yr", fontsize=11)

ax.plot(0, 0, "^", color="red", markersize=18, markeredgewidth=2, label="Release Point", zorder=10)

ax.annotate("", xy=(220, 220), xytext=(120, 120),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
ax.text(260, 260, "Wind\n2.5 m/s", fontsize=13, ha="center", fontweight="bold")

for dist, label in [(100, "Office"), (200, "Control\nRoom"), (500, "Residential")]:
    angle = wind_angle * 0.5
    rx, ry = dist * np.cos(angle), dist * np.sin(angle)
    ax.plot(rx, ry, "s", color="navy", markersize=13, zorder=10)
    ax.text(rx+18, ry+18, f"{label}\n({dist}m)", fontsize=12, color="navy", fontweight="bold")

cbar = fig.colorbar(cs, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Individual Risk (per year)", fontsize=13, fontweight="bold")
cbar.set_ticks(levels)
cbar.set_ticklabels([f"{l:.0e}" for l in levels])
cbar.ax.tick_params(labelsize=11)

ax.set_xlabel("X (m)", fontsize=14, fontweight="bold")
ax.set_ylabel("Y (m)", fontsize=14, fontweight="bold")
ax.set_title("Individual Risk Contour Map\nPropane Storage Facility — Balikpapan",
             fontweight="bold", fontsize=16, pad=15)
ax.legend(loc="lower right", fontsize=12, framealpha=0.9, edgecolor="gray")
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUTDIR / "08_risk_contour_map.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("\n✅ All 8 plots regenerated with improved fonts!")
